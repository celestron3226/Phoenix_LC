"""
train_phoenix.py
----------------
Fine-tune a U-Net (ResNet18 encoder pretrained on ChesapeakeCVPR) on the 3000
manually-labeled Phoenix tiles. 7 classes, Dataset-A philosophy (no shadow
class; shadows were annotated as the underlying surface).

Design (mirrors train_chesapeake.py where it matters):
  1. Augmentation via albumentations, IDENTICAL parameters to
     train_chesapeake.py: H/V flip, RandomRotate90(p=1.0),
     RandomResizedCrop(scale=(0.25,1.0), ratio=(1,1), p=0.5),
     RandomBrightnessContrast(brightness 0.05 / contrast 0.15, p=0.8).
     The geometric ops run on the FULL channel stack + mask (so NDVI/CHM/coord
     channels stay co-registered); brightness/contrast is applied to the 4
     NAIP channels ONLY (jittering CHM metres or coordinates is meaningless).
  2. Coordinate encoding via --coord {none, xy, sincos} (see phoenix_common).
     Channels are built from each tile's REAL affine transform BEFORE
     augmentation, so flips/rotations/crops transform them together with the
     imagery -- the network always sees geometrically consistent coords.
  3. Chesapeake encoder transfer: loads "encoder."-prefixed keys, copies
     conv1 weights for the 4 NAIP channels, random-inits the extra
     NDVI/CHM/coord input channels (same loader logic as the local train.py).
  4. Class-weighted CE (ignore 255) + 0.7 * Dice (ignore 255), AdamW,
     CosineAnnealingLR over --epochs, AMP.
  5. Full seeding, resume support, cumulative best checkpoints, train_log.csv,
     training curve png. RUN_DIR is built from the parameters.

Usage:
  python train_phoenix.py --bs 16 --lr 1e-4 --seed 1 --coord xy
  python train_phoenix.py --bs 16 --lr 1e-4 --seed 1 --coord none      # ablation
  python train_phoenix.py --bs 16 --lr 1e-4 --seed 1 --coord xy --epochs 2   # timing test
  python train_phoenix.py --bs 16 --lr 1e-4 --seed 1 --coord xy --resume 2026-07-21_090000

Outputs in <TRAIN_ROOT>/<arch>_bs<BS>_lr<LR>_seed<SEED>_coord-<MODE>/<timestamp>/:
  best_unet_e<NNN>_miou<X.XXXX>.pth     full U-Net at each new best
  best_encoder_e<NNN>_miou<X.XXXX>.pth  encoder-only ("encoder." prefix)
  latest_checkpoint.pth                 resume checkpoint
  run_config.json                       everything predict_phoenix.py needs
  train_log.csv / training_curve.png
"""

import argparse
import csv
import glob
import json
import os
import random
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

from phoenix_common import coord_channels, coord_num_channels, normalize_stack

# =============================================================================
# ARGS   ---  path defaults below are placeholders; edit or pass via CLI (README)
# =============================================================================
parser = argparse.ArgumentParser(description="Phoenix 7-class U-Net fine-tuning")
parser.add_argument("--bs", type=int, required=True, help="batch size")
parser.add_argument("--lr", type=float, required=True, help="peak learning rate")
parser.add_argument("--seed", type=int, required=True, help="random seed")
parser.add_argument("--coord", type=str, default="xy",
                    choices=["none", "xy", "sincos"],
                    help="coordinate encoding mode (default xy)")
parser.add_argument("--arch", type=str, default="resnet18",
                    choices=["resnet18", "resnet34", "resnet50"])
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--encoder-ckpt", type=str,
                    default="/path/to/chesapeake_pretrain/best_encoder_e032_miou0.7219.pth",
                    help="Chesapeake pretrained encoder ('encoder.' prefixed keys)")
parser.add_argument("--tile-root", type=str,
                    default="/path/to/Phoenix/Result/tiles")
parser.add_argument("--train-root", type=str,
                    default="/path/to/Phoenix/Result/training")
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--resume", type=str, default=None,
                    help="timestamp folder of an existing run to resume")
args = parser.parse_args()

# =============================================================================
# CONFIG / RUN DIR
# =============================================================================
with open(os.path.join(args.tile_root, "config.json")) as f:
    CFG = json.load(f)

NUM_CLASSES = int(CFG["num_classes"])
CLASS_NAMES = CFG["class_names"]
IGNORE_INDEX = int(CFG["ignore_index"])
NAIP_CH = int(CFG["naip_band_count"])
BASE_CH = int(CFG["base_in_channels"])            # NAIP + NDVI + CHM
COORD_CH = coord_num_channels(args.coord)
IN_CHANNELS = BASE_CH + COORD_CH
BOUNDS = CFG["coord_bounds"]

WEIGHT_DECAY = 1e-4
DICE_WEIGHT = 0.7
MAX_CLASS_WEIGHT = 10.0    # cap inverse-frequency weights (Water is rare)

PARAM_DIR = os.path.join(
    args.train_root,
    f"{args.arch}_bs{args.bs}_lr{args.lr:g}_seed{args.seed}_coord-{args.coord}")
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# AUGMENTATION -- identical parameters to train_chesapeake.py
# =============================================================================
def build_geometric_augs(size):
    """Geometric part of the Chesapeake pipeline; applied to the FULL channel
    stack (NAIP+NDVI+CHM+coords) jointly with the mask. Masks use NEAREST
    automatically. Coord channels are linear ramps of real coordinates, so
    crop-resize interpolates them EXACTLY and flips/rotations keep them
    consistent with the imagery."""
    try:
        rrc = A.RandomResizedCrop(size=(size, size), scale=(0.25, 1.0),
                                  ratio=(1.0, 1.0), p=0.5)
    except TypeError:
        rrc = A.RandomResizedCrop(height=size, width=size, scale=(0.25, 1.0),
                                  ratio=(1.0, 1.0), p=0.5)
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1.0),
        rrc,
    ])


def build_radiometric_aug():
    """RandomBrightnessContrast exactly as in train_chesapeake.py, but applied
    to the 4 NAIP channels only (NDVI/CHM/coords must not be jittered)."""
    return A.RandomBrightnessContrast(brightness_limit=0.05,
                                      contrast_limit=0.15, p=0.8)


# =============================================================================
# DATASET
# =============================================================================
class PhoenixTiles(Dataset):
    def __init__(self, npz_paths, cfg, coord_mode, augment):
        self.paths = list(npz_paths)
        self.cfg = cfg
        self.coord_mode = coord_mode
        self.augment = augment
        self.geo_tf = None
        self.rad_tf = None
        if augment:
            h, w = cfg["tile_size"]
            assert h == w, "geometric augs assume square tiles"
            self.geo_tf = build_geometric_augs(h)
            self.rad_tf = build_radiometric_aug()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with np.load(self.paths[idx]) as z:
            img = z["image"].astype(np.float32)          # (H,W,6) raw values
            mask = z["mask"].astype(np.uint8)            # (H,W)
            tr = z["transform"]

        img = normalize_stack(img, self.cfg)             # all channels -> ~[0,1]
        if self.coord_mode != "none":
            cc = coord_channels(tr, img.shape[0], img.shape[1],
                                self.cfg["coord_bounds"], self.coord_mode)
            img = np.concatenate([img, cc], axis=2)

        if self.augment:
            out = self.geo_tf(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
            # radiometric jitter on NAIP bands only
            naip = np.ascontiguousarray(img[..., :NAIP_CH])
            img[..., :NAIP_CH] = self.rad_tf(image=naip)["image"]

        img = np.transpose(img, (2, 0, 1))
        return (torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(mask.astype(np.int64))))


# =============================================================================
# CHESAPEAKE ENCODER TRANSFER
# =============================================================================
def load_chesapeake_encoder(model, ckpt_path, in_channels):
    """Copy 'encoder.'-prefixed weights; conv1 gets the pretrained 4 NAIP
    channels, extra channels (NDVI, CHM, coords) stay randomly initialized."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Chesapeake encoder not found: {ckpt_path}\n"
            f"(refusing to silently train from random init)")
    print(f"[INFO] Loading Chesapeake encoder: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else None
    if sd is None:
        raise ValueError("Unexpected checkpoint format")

    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    if not enc_sd:
        raise ValueError("No 'encoder.*' keys in checkpoint -- wrong file? "
                         "Use best_encoder_*.pth, not a bare state dict.")

    model_sd = model.encoder.state_dict()
    new_sd = {}
    for k, v in enc_sd.items():
        if k not in model_sd:
            continue
        if k == "conv1.weight":
            w = model_sd[k].clone()                       # (64, in_ch, 7, 7) random
            n = min(NAIP_CH, v.shape[1], in_channels)
            w[:, :n] = v[:, :n]
            # scale so total input magnitude stays comparable despite extra chans
            new_sd[k] = w * (v.shape[1] / float(in_channels))
            print(f"[INFO] conv1: {n} NAIP channels from pretrained, "
                  f"{in_channels - n} extra channels random-init "
                  f"(rescaled x{v.shape[1] / float(in_channels):.2f})")
        else:
            new_sd[k] = v
    res = model.encoder.load_state_dict(new_sd, strict=False)
    missing = res.missing_keys if res is not None else []
    unexpected = res.unexpected_keys if res is not None else []
    print(f"[INFO] encoder loaded: {len(new_sd)} tensors "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return model


# =============================================================================
# LOSS / METRIC
# =============================================================================
def soft_dice_loss(logits, targets, num_classes, ignore_index, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    valid = (targets != ignore_index)
    t = targets.clone()
    t[~valid] = 0
    onehot = F.one_hot(t, num_classes).permute(0, 3, 1, 2).float()
    v = valid.unsqueeze(1).float()
    inter = (probs * v * onehot).sum(dim=(0, 2, 3))
    union = (probs * v).sum(dim=(0, 2, 3)) + (onehot * v).sum(dim=(0, 2, 3))
    return 1.0 - ((2 * inter + eps) / (union + eps)).mean()


def make_dice():
    try:
        from segmentation_models_pytorch.losses import DiceLoss
        return DiceLoss(mode="multiclass", ignore_index=IGNORE_INDEX)
    except Exception:
        print("[WARN] smp DiceLoss unavailable/incompatible -> masked soft dice")
        return lambda lo, ta: soft_dice_loss(lo, ta, NUM_CLASSES, IGNORE_INDEX)


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


# =============================================================================
# CLASS WEIGHTS (inverse frequency from config counts, capped)
# =============================================================================
def class_weights_from_cfg(cfg):
    counts = np.array([cfg["class_pixel_counts"][n] for n in CLASS_NAMES], dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (len(counts) * counts)     # inverse freq, mean ~1
    w = np.minimum(w / w.mean(), MAX_CLASS_WEIGHT)
    return w.astype(np.float32)


# =============================================================================
# MAIN
# =============================================================================
def main():
    seed_everything(args.seed)

    train_tiles = sorted(glob.glob(os.path.join(args.tile_root, "tiles", "train", "*.npz")))
    val_tiles = sorted(glob.glob(os.path.join(args.tile_root, "tiles", "val", "*.npz")))
    if not train_tiles or not val_tiles:
        raise SystemExit("No tiles found -- run prepare_tiles_phoenix.py first.")

    print("=" * 60)
    print(f"[INFO] device={DEVICE}")
    print(f"[INFO] arch={args.arch} bs={args.bs} lr={args.lr:g} seed={args.seed} "
          f"epochs={args.epochs} coord={args.coord}")
    print(f"[INFO] in_channels={IN_CHANNELS} (NAIP {NAIP_CH} + NDVI 1 + CHM 1 + coord {COORD_CH})")
    print(f"[INFO] train={len(train_tiles)} val={len(val_tiles)} tiles")
    print(f"[INFO] run dir: {RUN_DIR}")
    print("=" * 60)

    g = torch.Generator(); g.manual_seed(args.seed)
    dl_kw = dict(num_workers=args.workers, pin_memory=(DEVICE.type == "cuda"),
                 worker_init_fn=worker_init_fn, generator=g,
                 persistent_workers=(args.workers > 0))
    dl_tr = DataLoader(PhoenixTiles(train_tiles, CFG, args.coord, augment=True),
                       batch_size=args.bs, shuffle=True, drop_last=True, **dl_kw)
    dl_va = DataLoader(PhoenixTiles(val_tiles, CFG, args.coord, augment=False),
                       batch_size=args.bs, shuffle=False, **dl_kw)

    model = smp.Unet(encoder_name=args.arch, encoder_weights=None,
                     in_channels=IN_CHANNELS, classes=NUM_CLASSES).to(DEVICE)
    model = load_chesapeake_encoder(model, args.encoder_ckpt, IN_CHANNELS)

    w = class_weights_from_cfg(CFG)
    print(f"[INFO] class weights: {dict(zip(CLASS_NAMES, [round(float(x), 3) for x in w]))}")
    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEVICE),
                                  ignore_index=IGNORE_INDEX)
    dice_loss = make_dice()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    # ---- run_config.json: everything prediction needs to rebuild the model ----
    with open(os.path.join(RUN_DIR, "run_config.json"), "w") as f:
        json.dump({"arch": args.arch, "coord": args.coord,
                   "in_channels": IN_CHANNELS, "num_classes": NUM_CLASSES,
                   "class_names": CLASS_NAMES, "tile_root": args.tile_root,
                   "bs": args.bs, "lr": args.lr, "seed": args.seed,
                   "epochs": args.epochs, "encoder_ckpt": args.encoder_ckpt,
                   "class_weights": [float(x) for x in w]}, f, indent=2)

    # ---- resume ----
    start_epoch, best_miou = 1, -1.0
    latest = os.path.join(RUN_DIR, "latest_checkpoint.pth")
    if args.resume and os.path.exists(latest):
        print(f"[RESUME] {latest}")
        ck = torch.load(latest, map_location=DEVICE)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_miou = ck["best_miou"]
        print(f"[RESUME] from epoch {start_epoch}, best mIoU={best_miou:.4f}")

    log_csv = os.path.join(RUN_DIR, "train_log.csv")
    FIELDS = (["epoch", "lr", "train_loss", "val_loss", "val_miou"]
              + [f"IoU_{n}" for n in CLASS_NAMES]
              + ["train_time_sec", "val_time_sec", "epoch_time_sec", "end_timestamp"])
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    run_start = datetime.now()
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        cur_lr = optimizer.param_groups[0]["lr"]

        # ---- train ----
        model.train()
        running = 0.0
        for x, y in dl_tr:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                logits = model(x)
                loss = ce_loss(logits, y) + DICE_WEIGHT * dice_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        train_loss = running / max(len(dl_tr), 1)
        t_train = time.time() - t0

        # ---- validate ----
        tv = time.time()
        model.eval()
        conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=DEVICE)
        vrun = 0.0
        with torch.no_grad():
            for x, y in dl_va:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    logits = model(x)
                    vrun += (ce_loss(logits, y) + DICE_WEIGHT * dice_loss(logits, y)).item()
                pred = torch.argmax(logits, dim=1)
                conf = update_confusion(conf, pred.flatten(), y.flatten(), NUM_CLASSES)
        val_loss = vrun / max(len(dl_va), 1)
        t_val = time.time() - tv

        iou, miou = ious_from_confusion(conf.cpu())
        print(f"[E{epoch:03d}] lr={cur_lr:.2e} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | mIoU={miou:.4f} | "
              f"IoU={[round(v, 3) for v in iou.tolist()]}")

        # ---- best checkpoints (cumulative) ----
        if miou > best_miou:
            best_miou = miou
            tag = f"e{epoch:03d}_miou{miou:.4f}"
            torch.save(model.state_dict(), os.path.join(RUN_DIR, f"best_unet_{tag}.pth"))
            enc_sd = {f"encoder.{k}": v for k, v in model.encoder.state_dict().items()}
            torch.save(enc_sd, os.path.join(RUN_DIR, f"best_encoder_{tag}.pth"))
            print(f"        ^ new best -> {tag}")

        # ---- latest checkpoint (resume) ----
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "best_miou": best_miou}, latest)

        # ---- CSV ----
        row = {"epoch": epoch, "lr": cur_lr, "train_loss": train_loss,
               "val_loss": val_loss, "val_miou": miou,
               "train_time_sec": round(t_train, 1), "val_time_sec": round(t_val, 1),
               "epoch_time_sec": round(time.time() - t0, 1),
               "end_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        for n, v in zip(CLASS_NAMES, iou.tolist()):
            row[f"IoU_{n}"] = v
        with open(log_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

        scheduler.step()

    dur = datetime.now() - run_start
    print(f"\n[TOTAL] runtime: {dur} ({dur.total_seconds() / 3600:.2f} hr)")
    print(f"[DONE] best val mIoU = {best_miou:.4f}")

    # ---- training curve ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep, trl, val, mi = [], [], [], []
        with open(log_csv) as f:
            for r in csv.DictReader(f):
                ep.append(int(r["epoch"])); trl.append(float(r["train_loss"]))
                val.append(float(r["val_loss"])); mi.append(float(r["val_miou"]))
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(ep, trl, label="train loss"); ax1.plot(ep, val, label="val loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(loc="upper left")
        ax2 = ax1.twinx(); ax2.plot(ep, mi, "g-", label="val mIoU"); ax2.set_ylabel("mIoU")
        ax2.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(RUN_DIR, "training_curve.png"), dpi=150)
        print(f"[DONE] curve -> {os.path.join(RUN_DIR, 'training_curve.png')}")
    except Exception as e:
        print(f"[WARN] plot failed: {e}")


if __name__ == "__main__":
    main()
