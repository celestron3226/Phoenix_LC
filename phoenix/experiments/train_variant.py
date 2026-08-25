"""
train_variant.py
----------------
Trains the two ALTERNATIVE methods for the Phoenix land-cover comparison, on
the SAME 3000-tile data + SAME Chesapeake encoder + SAME albumentations as the
end-to-end Plan A (train_phoenix.py), so the three results are directly
comparable and only the METHOD differs.

  --method hybrid   (Week-4 style: DL detects, CHM rule structures)
      * DL target = 5 classes {Vegetation, Soil, Building, Asphalt, Water}
        (Tree/Shrub/Grass collapsed into Vegetation).
      * At prediction the CHM height rule re-splits Vegetation into
        Tree/Shrub/Grass -> final 7 classes.
      * Validation reports BOTH the 5-class DL mIoU and the final 7-class mIoU
        (height rule applied, scored against the real 7-class labels) so it is
        comparable to Plan A's headline number.
      * Input = NAIP + NDVI + CHM (+coord): full stack, like Plan A.

  --method planb    (Week-6 style: rule pre-labeling, then DL - weakly supervised)
      * TRAIN targets = rule pseudo-labels {Tree, Shrub, Grass, NonVeg}, generated
        on the fly from each tile's NDVI+CHM (NO manual labels used for training).
      * VALIDATION targets = the REAL manual labels, collapsed to the same 4
        classes -> an honest mIoU, comparable to Plan A / hybrid.
      * Input = NAIP (+coord) ONLY by default. NDVI/CHM are deliberately kept
        OUT of the input: the pseudo-labels are DERIVED from NDVI+CHM, so feeding
        them in lets the net trivially copy the rule instead of learning veg
        structure from imagery (the whole point of Plan B). Use --leak-features
        to include them anyway for an ablation.

Shared with Plan A via phoenix_common: normalize_stack, coord_channels,
load_chesapeake_encoder, and the class-scheme helpers.

Usage (Sol):
  python train_variant.py --method hybrid --bs 16 --lr 1e-4 --seed 1 --coord xy
  python train_variant.py --method planb  --bs 16 --lr 1e-4 --seed 1 --coord xy

Outputs in <train-root>/<method>_<arch>_bs<BS>_lr<LR>_seed<SEED>_coord-<MODE>/<ts>/
  best_unet_e<NNN>_miou<X.XXXX>.pth   (selection metric = the comparable mIoU)
  run_config.json  train_log.csv  training_curve.png  latest_checkpoint.pth
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp

# phoenix_common.py lives in ../training in this repo layout; make it importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from phoenix_common import (coord_channels, coord_num_channels, normalize_stack,
                            real_ndvi_chm, remap7_to5, remap7_to4,
                            hybrid_pred5_to_codes, rule_pseudolabel4,
                            load_chesapeake_encoder)

# =============================================================================
# ARGS   ---  path defaults below are placeholders; edit or pass via CLI (README)
# =============================================================================
parser = argparse.ArgumentParser(description="Phoenix hybrid / Plan-B training")
parser.add_argument("--method", required=True, choices=["hybrid", "planb"])
parser.add_argument("--bs", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--coord", type=str, default="xy", choices=["none", "xy", "sincos"])
parser.add_argument("--arch", type=str, default="resnet18",
                    choices=["resnet18", "resnet34", "resnet50"])
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--leak-features", action="store_true",
                    help="[planb ablation] feed NDVI/CHM into the input too "
                         "(defeats the point of Plan B; off by default)")
parser.add_argument("--encoder-ckpt", type=str,
                    default="/path/to/chesapeake_pretrain/best_encoder_e032_miou0.7219.pth")
parser.add_argument("--tile-root", type=str,
                    default="/path/to/Phoenix/Result/tiles")
parser.add_argument("--train-root", type=str,
                    default="/path/to/Phoenix/Result/training")
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--resume", type=str, default=None)
args = parser.parse_args()

# =============================================================================
# CONFIG
# =============================================================================
with open(os.path.join(args.tile_root, "config.json")) as f:
    CFG = json.load(f)

NAIP_CH = int(CFG["naip_band_count"])
IGNORE_INDEX = int(CFG["ignore_index"])
COORD_CH = coord_num_channels(args.coord)
BOUNDS = CFG["coord_bounds"]

if args.method == "hybrid":
    NUM_CLASSES = 5                              # DL output classes
    CLASS_NAMES = ["Veg", "Soil", "Building", "Asphalt", "Water"]
    FINAL_NAMES = ["Tree", "Shrub", "Grass", "Soil", "Building", "Asphalt", "Water"]
    USE_NDVI_CHM_INPUT = True
else:  # planb
    NUM_CLASSES = 4
    CLASS_NAMES = ["Tree", "Shrub", "Grass", "NonVeg"]
    FINAL_NAMES = CLASS_NAMES
    USE_NDVI_CHM_INPUT = bool(args.leak_features)

BASE_CH = (NAIP_CH + 2) if USE_NDVI_CHM_INPUT else NAIP_CH
IN_CHANNELS = BASE_CH + COORD_CH

WEIGHT_DECAY = 1e-4
DICE_WEIGHT = 0.7
MAX_CLASS_WEIGHT = 10.0

PARAM_DIR = os.path.join(
    args.train_root,
    f"{args.method}_{args.arch}_bs{args.bs}_lr{args.lr:g}_seed{args.seed}_coord-{args.coord}")
if args.resume:
    RUN_DIR = os.path.join(PARAM_DIR, args.resume)
    assert os.path.exists(RUN_DIR), f"Resume dir not found: {RUN_DIR}"
else:
    RUN_DIR = os.path.join(PARAM_DIR, datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    os.makedirs(RUN_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# SEEDING
# =============================================================================
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)


# =============================================================================
# AUGMENTATION -- identical parameters to train_chesapeake.py / Plan A
# =============================================================================
def build_geometric_augs(size):
    try:
        rrc = A.RandomResizedCrop(size=(size, size), scale=(0.25, 1.0),
                                  ratio=(1.0, 1.0), p=0.5)
    except TypeError:
        rrc = A.RandomResizedCrop(height=size, width=size, scale=(0.25, 1.0),
                                  ratio=(1.0, 1.0), p=0.5)
    return A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                      A.RandomRotate90(p=1.0), rrc])


def build_radiometric_aug():
    return A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.15, p=0.8)


# =============================================================================
# DATASET
#   returns (image tensor, dl_target, gt7) where
#     dl_target = the loss target (5-class remap for hybrid; 4-class pseudo-label
#                 for planb-train; 4-class real remap for planb-val)
#     gt7       = real 7-class labels (for the comparable metric); planb ignores it
# =============================================================================
class VariantTiles(Dataset):
    def __init__(self, npz_paths, cfg, method, coord_mode, use_ndvi_chm,
                 split, augment):
        self.paths = list(npz_paths)
        self.cfg = cfg
        self.method = method
        self.coord_mode = coord_mode
        self.use_ndvi_chm = use_ndvi_chm
        self.split = split                    # "train" | "val"
        self.augment = augment
        self.naip_ch = int(cfg["naip_band_count"])
        self.geo_tf = build_geometric_augs(cfg["tile_size"][0]) if augment else None
        self.rad_tf = build_radiometric_aug() if augment else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with np.load(self.paths[idx]) as z:
            raw = z["image"].astype(np.float32)      # (H,W,6) raw
            gt7 = z["mask"].astype(np.uint8)         # real 7-class internal
            tr = z["transform"]

        ndvi_real, chm_real = real_ndvi_chm(raw, self.cfg)

        # ---- DL loss target ----
        if self.method == "hybrid":
            target = remap7_to5(gt7)                 # 5-class from real labels
        else:  # planb
            if self.split == "train":
                target = rule_pseudolabel4(ndvi_real, chm_real, self.cfg)  # weak
            else:
                target = remap7_to4(gt7)             # real labels for honest val

        # ---- input channels ----
        norm = normalize_stack(raw, self.cfg)        # (H,W,6) normalized
        if self.use_ndvi_chm:
            img = norm                               # NAIP+NDVI+CHM
        else:
            img = norm[..., :self.naip_ch]           # NAIP only
        if self.coord_mode != "none":
            cc = coord_channels(tr, img.shape[0], img.shape[1],
                                self.cfg["coord_bounds"], self.coord_mode)
            img = np.concatenate([img, cc], axis=2)

        # carry raw CHM as an extra plane so val can apply the height rule after
        # the same geometric augment (kept co-registered). Stored as a channel,
        # stripped before the model sees it.
        chm_plane = chm_real[..., None].astype(np.float32)
        stack = np.concatenate([img, chm_plane], axis=2)

        # ---- augment (geometry on full stack + both label maps jointly) ----
        if self.augment:
            masks = np.stack([target, gt7], axis=-1)     # (H,W,2)
            out = self.geo_tf(image=stack, mask=masks)
            stack, masks = out["image"], out["mask"]
            target, gt7 = masks[..., 0], masks[..., 1]
            naip = np.ascontiguousarray(stack[..., :self.naip_ch])
            stack[..., :self.naip_ch] = self.rad_tf(image=naip)["image"]

        img = stack[..., :-1]
        chm_out = stack[..., -1]
        img = np.transpose(img, (2, 0, 1))
        return (torch.from_numpy(np.ascontiguousarray(img, np.float32)),
                torch.from_numpy(np.ascontiguousarray(target.astype(np.int64))),
                torch.from_numpy(np.ascontiguousarray(gt7.astype(np.int64))),
                torch.from_numpy(np.ascontiguousarray(chm_out.astype(np.float32))))


# =============================================================================
# LOSS / METRIC
# =============================================================================
def soft_dice_loss(logits, targets, num_classes, ignore_index, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    valid = (targets != ignore_index)
    t = targets.clone(); t[~valid] = 0
    onehot = F.one_hot(t, num_classes).permute(0, 3, 1, 2).float()
    v = valid.unsqueeze(1).float()
    inter = (probs * v * onehot).sum(dim=(0, 2, 3))
    union = (probs * v).sum(dim=(0, 2, 3)) + (onehot * v).sum(dim=(0, 2, 3))
    return 1.0 - ((2 * inter + eps) / (union + eps)).mean()


def make_dice(num_classes, ignore_index):
    try:
        from segmentation_models_pytorch.losses import DiceLoss
        return DiceLoss(mode="multiclass", ignore_index=ignore_index)
    except Exception:
        print("[WARN] smp DiceLoss unavailable -> masked soft dice")
        return lambda lo, ta: soft_dice_loss(lo, ta, num_classes, ignore_index)


def update_confusion(conf, pred, target, num_classes):
    with torch.no_grad():
        k = (target >= 0) & (target < num_classes)
        idx = num_classes * target[k].to(torch.int64) + pred[k]
        conf += torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)
    return conf


def ious_from_confusion(conf):
    inter = torch.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    iou = inter.float() / union.clamp(min=1).float()
    miou = iou[union > 0].mean().item() if (union > 0).any() else 0.0
    return iou, miou


def class_weights(counts):
    counts = np.maximum(np.asarray(counts, np.float64), 1.0)
    w = counts.sum() / (len(counts) * counts)
    return np.minimum(w / w.mean(), MAX_CLASS_WEIGHT).astype(np.float32)


def compute_dl_weights(paths, method, cfg, num_classes):
    """Class weights over the DL TRAIN target (5-class remap or 4-class pseudo)."""
    counts = np.zeros(num_classes, np.float64)
    for p in paths:
        with np.load(p) as z:
            raw = z["image"].astype(np.float32); gt7 = z["mask"].astype(np.uint8)
        if method == "hybrid":
            t = remap7_to5(gt7)
        else:
            ndvi, chm = real_ndvi_chm(raw, cfg)
            t = rule_pseudolabel4(ndvi, chm, cfg)
        for c in range(num_classes):
            counts[c] += int((t == c).sum())
    return class_weights(counts), counts


# =============================================================================
# MAIN
# =============================================================================
def main():
    seed_everything(args.seed)
    train_tiles = sorted(glob.glob(os.path.join(args.tile_root, "tiles", "train", "*.npz")))
    val_tiles = sorted(glob.glob(os.path.join(args.tile_root, "tiles", "val", "*.npz")))
    if not train_tiles or not val_tiles:
        raise SystemExit("No tiles -- run prepare_tiles_phoenix.py first.")

    print("=" * 60)
    print(f"[INFO] method={args.method} device={DEVICE}")
    print(f"[INFO] arch={args.arch} bs={args.bs} lr={args.lr:g} seed={args.seed} "
          f"epochs={args.epochs} coord={args.coord}")
    print(f"[INFO] in_channels={IN_CHANNELS} "
          f"(NAIP {NAIP_CH}{' + NDVI + CHM' if USE_NDVI_CHM_INPUT else ''} + coord {COORD_CH})")
    print(f"[INFO] DL classes={NUM_CLASSES} ({CLASS_NAMES})")
    if args.method == "hybrid":
        print(f"[INFO] final classes after height rule = 7 ({FINAL_NAMES})")
    else:
        print(f"[INFO] train=rule pseudo-labels | val=real labels (4-class)")
    print(f"[INFO] train={len(train_tiles)} val={len(val_tiles)} | run: {RUN_DIR}")
    print("=" * 60)

    g = torch.Generator(); g.manual_seed(args.seed)
    dl_kw = dict(num_workers=args.workers, pin_memory=(DEVICE.type == "cuda"),
                 worker_init_fn=worker_init_fn, generator=g,
                 persistent_workers=(args.workers > 0))
    dl_tr = DataLoader(VariantTiles(train_tiles, CFG, args.method, args.coord,
                                    USE_NDVI_CHM_INPUT, "train", augment=True),
                       batch_size=args.bs, shuffle=True, drop_last=True, **dl_kw)
    dl_va = DataLoader(VariantTiles(val_tiles, CFG, args.method, args.coord,
                                    USE_NDVI_CHM_INPUT, "val", augment=False),
                       batch_size=args.bs, shuffle=False, **dl_kw)

    model = smp.Unet(encoder_name=args.arch, encoder_weights=None,
                     in_channels=IN_CHANNELS, classes=NUM_CLASSES).to(DEVICE)
    model = load_chesapeake_encoder(model, args.encoder_ckpt, NAIP_CH, IN_CHANNELS)

    w, counts = compute_dl_weights(train_tiles, args.method, CFG, NUM_CLASSES)
    print(f"[INFO] DL-target pixel counts: {dict(zip(CLASS_NAMES, counts.astype(int)))}")
    print(f"[INFO] class weights: {dict(zip(CLASS_NAMES, [round(float(x),3) for x in w]))}")
    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEVICE),
                                  ignore_index=IGNORE_INDEX)
    dice_loss = make_dice(NUM_CLASSES, IGNORE_INDEX)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    # run_config.json -- everything predict_variant.py needs
    with open(os.path.join(RUN_DIR, "run_config.json"), "w") as f:
        json.dump({"method": args.method, "arch": args.arch, "coord": args.coord,
                   "in_channels": IN_CHANNELS, "dl_classes": NUM_CLASSES,
                   "dl_class_names": CLASS_NAMES, "final_class_names": FINAL_NAMES,
                   "use_ndvi_chm_input": USE_NDVI_CHM_INPUT,
                   "tile_root": args.tile_root, "bs": args.bs, "lr": args.lr,
                   "seed": args.seed, "epochs": args.epochs,
                   "encoder_ckpt": args.encoder_ckpt}, f, indent=2)

    # metric class count for the COMPARABLE headline mIoU
    metric_classes = 7 if args.method == "hybrid" else 4
    metric_names = FINAL_NAMES

    start_epoch, best_metric = 1, -1.0
    latest = os.path.join(RUN_DIR, "latest_checkpoint.pth")
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location=DEVICE)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1; best_metric = ck["best_metric"]
        print(f"[RESUME] from epoch {start_epoch}, best={best_metric:.4f}")

    log_csv = os.path.join(RUN_DIR, "train_log.csv")
    FIELDS = (["epoch", "lr", "train_loss", "val_loss", "val_miou_dl", "val_miou_final"]
              + [f"IoU_{n}" for n in metric_names]
              + ["train_time_sec", "val_time_sec", "epoch_time_sec", "end_timestamp"])
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    run_start = datetime.now()
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time(); cur_lr = optimizer.param_groups[0]["lr"]

        # ---- train ----
        model.train(); running = 0.0
        for x, target, _gt7, _chm in dl_tr:
            x = x.to(DEVICE, non_blocking=True)
            y = target.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                logits = model(x)
                loss = ce_loss(logits, y) + DICE_WEIGHT * dice_loss(logits, y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running += loss.item()
        train_loss = running / max(len(dl_tr), 1); t_train = time.time() - t0

        # ---- validate ----
        tv = time.time(); model.eval()
        conf_dl = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
        conf_fin = torch.zeros(metric_classes, metric_classes, dtype=torch.int64)
        vrun = 0.0
        with torch.no_grad():
            for x, target, gt7, chm in dl_va:
                x = x.to(DEVICE, non_blocking=True)
                y = target.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    logits = model(x)
                    vrun += (ce_loss(logits, y) + DICE_WEIGHT * dice_loss(logits, y)).item()
                pred = torch.argmax(logits, dim=1).cpu()
                conf_dl = update_confusion(conf_dl, pred.flatten(), target.flatten(), NUM_CLASSES)

                # ---- comparable final metric ----
                if args.method == "hybrid":
                    # apply height rule -> codes 1..7 -> internal 0..6, score vs real gt7
                    p5 = pred.numpy().astype(np.uint8)
                    chm_np = chm.numpy()
                    codes = np.stack([hybrid_pred5_to_codes(p5[i], chm_np[i], CFG)
                                      for i in range(p5.shape[0])])
                    fin_pred = torch.from_numpy(codes.astype(np.int64) - 1)
                    fin_gt = gt7
                else:
                    fin_pred = pred                 # 4-class DL prediction
                    fin_gt = target                 # val target is real 4-class
                conf_fin = update_confusion(conf_fin, fin_pred.flatten(),
                                            fin_gt.flatten(), metric_classes)
        val_loss = vrun / max(len(dl_va), 1); t_val = time.time() - tv

        _, miou_dl = ious_from_confusion(conf_dl)
        iou_fin, miou_fin = ious_from_confusion(conf_fin)
        print(f"[E{epoch:03d}] lr={cur_lr:.2e} | loss={train_loss:.4f}/{val_loss:.4f} | "
              f"mIoU_dl({NUM_CLASSES})={miou_dl:.4f} | mIoU_final({metric_classes})={miou_fin:.4f} | "
              f"IoU={[round(v,3) for v in iou_fin.tolist()]}")

        if miou_fin > best_metric:
            best_metric = miou_fin
            tag = f"e{epoch:03d}_miou{miou_fin:.4f}"
            torch.save(model.state_dict(), os.path.join(RUN_DIR, f"best_unet_{tag}.pth"))
            print(f"        ^ new best ({metric_classes}-class mIoU) -> {tag}")

        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "best_metric": best_metric}, latest)

        row = {"epoch": epoch, "lr": cur_lr, "train_loss": train_loss,
               "val_loss": val_loss, "val_miou_dl": miou_dl, "val_miou_final": miou_fin,
               "train_time_sec": round(t_train, 1), "val_time_sec": round(t_val, 1),
               "epoch_time_sec": round(time.time() - t0, 1),
               "end_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        for n, v in zip(metric_names, iou_fin.tolist()):
            row[f"IoU_{n}"] = v
        with open(log_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        scheduler.step()

    dur = datetime.now() - run_start
    print(f"\n[TOTAL] runtime: {dur} ({dur.total_seconds()/3600:.2f} hr)")
    print(f"[DONE] best {metric_classes}-class mIoU = {best_metric:.4f}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep, trl, val, mf = [], [], [], []
        with open(log_csv) as f:
            for r in csv.DictReader(f):
                ep.append(int(r["epoch"])); trl.append(float(r["train_loss"]))
                val.append(float(r["val_loss"])); mf.append(float(r["val_miou_final"]))
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(ep, trl, label="train loss"); ax1.plot(ep, val, label="val loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(loc="upper left")
        ax2 = ax1.twinx(); ax2.plot(ep, mf, "g-", label=f"val mIoU ({metric_classes}cls)")
        ax2.set_ylabel("mIoU"); ax2.legend(loc="upper right")
        fig.tight_layout(); fig.savefig(os.path.join(RUN_DIR, "training_curve.png"), dpi=150)
    except Exception as e:
        print(f"[WARN] plot failed: {e}")


if __name__ == "__main__":
    main()
