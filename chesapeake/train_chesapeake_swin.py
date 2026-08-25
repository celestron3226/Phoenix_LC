"""
train_chesapeake_swin.py
------------------------
train_chesapeake.py extended with --arch swin_b: pretrain a Swin-v2-B U-Net
on ChesapeakeCVPR so its encoder can transfer to the Phoenix experiments.
This fills the last cell of the 2x2 design (architecture x pretraining):
swin_b + chesapeake vs swin_b + satlas_aerial becomes a clean comparison
where ONLY the pretraining source differs.

Everything from the resnet version is kept identical: exhaustive
ShuffledGridSampler / GridGeoSampler, albumentations augs, CE(ignore 0)
+ 0.7 * Dice, AdamW + cosine, AMP, resume support, cumulative best
checkpoints, encoder saved with the "encoder." key prefix.

Swin specifics:
  - torchvision swin_v2_b (same implementation as the Week 12 Phoenix
    SwinUnet, so parameter names transfer without any key mapping).
  - Starts from ImageNet weights (Swin_V2_B_Weights.IMAGENET1K_V1), the
    same policy as the resnet runs (ENCODER_WEIGHTS="imagenet").
  - Patch embed is expanded 3ch -> 4ch (NAIP RGBN) with the same rule the
    resnet conv1 expansion uses: copy pretrained channels, keep the rest
    random, rescale by old_ch / new_ch.
  - Custom 4-block U-Net decoder + bilinear x2 head (Swin has no stride-2
    feature), identical to the Week 12 SwinUnet.
  - --accum N for gradient accumulation. Swin-B at bs128 / 256px may OOM
    even on an 80 GB A100; use --bs 64 --accum 2 to keep the EFFECTIVE
    batch at 128. The run folder then says bs64x2 so it stays traceable.

IMPORTANT (Sol): compute nodes may not reach the internet. Pre-download
the ImageNet weights ONCE on a login node before submitting:
  python -c "from torchvision.models import swin_v2_b, Swin_V2_B_Weights; \
             swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)"
(caches to ~/.cache/torch/hub/checkpoints)

Usage:
  python train_chesapeake_swin.py --arch swin_b --bs 128 --lr 1e-4 --seed 1
  python train_chesapeake_swin.py --arch swin_b --bs 64 --accum 2 --lr 1e-4 --seed 1
  python train_chesapeake_swin.py --arch swin_b --bs 128 --lr 1e-4 --seed 1 --epochs 2   # timing test
  python train_chesapeake_swin.py --arch swin_b --bs 128 --lr 1e-4 --seed 1 --resume 2026-08-01_090000

Outputs in <TRAIN_ROOT>/<arch>_bs<BS>[x<ACCUM>]_lr<LR>_seed<SEED>/<timestamp>/:
  best_unet_e<NNN>_miou<X.XXXX>.pth      full model at each new best
  best_encoder_e<NNN>_miou<X.XXXX>.pth   encoder-only, keys prefixed "encoder."
  latest_checkpoint.pth / train_log.csv / training_curve.png
"""

import argparse
import csv
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
from torch.utils.data import DataLoader
import albumentations as A
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

from torchgeo.datasets import ChesapeakeCVPR, stack_samples
from torchgeo.samplers import GridGeoSampler

try:
    import torchvision
    from torchvision.models import swin_v2_b, Swin_V2_B_Weights
    HAS_TV_SWIN = True
except ImportError:
    HAS_TV_SWIN = False


# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="ChesapeakeCVPR U-Net pretraining (+swin_b)")
parser.add_argument("--bs", type=int, required=True, help="per-step batch size")
parser.add_argument("--lr", type=float, required=True, help="peak learning rate")
parser.add_argument("--seed", type=int, required=True, help="random seed")
parser.add_argument("--arch", type=str, default="resnet18",
                    choices=["resnet18", "resnet34", "resnet50", "swin_b"],
                    help="encoder backbone (default resnet18)")
parser.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation steps; effective batch = bs * accum "
                         "(use --bs 64 --accum 2 if swin_b OOMs at bs 128)")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--resume", type=str, default=None,
                    help="timestamp folder of an existing run to resume")
args = parser.parse_args()

if args.arch == "swin_b" and not HAS_TV_SWIN:
    raise SystemExit("[FATAL] --arch swin_b needs torchvision >= 0.13 "
                     "(swin_v2_b not found). Check the pytorch_gpu env.")


# =============================================================================
# CONFIG   ---  >>> EDIT THESE PATHS FOR YOUR ENVIRONMENT (see README) <<<
# =============================================================================
ROOT = "/path/to/chesapeake_cvpr/cvpr_chesapeake_landcover"
TRAIN_ROOT = "/path/to/chesapeake_cvpr/Dataset_training"

# bs64x2 in the folder name when accumulating, so runs stay distinguishable
BS_TAG = f"{args.bs}" if args.accum == 1 else f"{args.bs}x{args.accum}"
PARAM_DIR = os.path.join(TRAIN_ROOT,
                         f"{args.arch}_bs{BS_TAG}_lr{args.lr:g}_seed{args.seed}")
if args.resume:
    RUN_DIR = os.path.join(PARAM_DIR, args.resume)
    assert os.path.exists(RUN_DIR), f"Resume dir not found: {RUN_DIR}"
else:
    RUN_DIR = os.path.join(PARAM_DIR, datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    os.makedirs(RUN_DIR, exist_ok=True)

CLASS_SET = 7
IGNORE_INDEX = 0

CLASS_NAMES = [
    "nodata",             # 0  (raw 15 remapped here; ignore_index)
    "water",              # 1
    "tree_canopy",        # 2
    "low_vegetation",     # 3
    "barren",             # 4
    "impervious_other",   # 5
    "impervious_road",    # 6
]
assert len(CLASS_NAMES) == CLASS_SET

PATCH_SIZE = 256
STRIDE = 256

ENCODER_NAME = args.arch
ENCODER_WEIGHTS = "imagenet"

ALL_STATES = ["de", "md", "ny", "pa", "va", "wv"]
TRAIN_SPLITS = [f"{s}-train" for s in ALL_STATES]
VAL_SPLITS = [f"{s}-val" for s in ALL_STATES]

NUM_WORKERS = 20
WEIGHT_DECAY = 1e-4
DICE_WEIGHT = 0.7

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
# SWIN-V2-B ENCODER + U-NET DECODER (identical to the Week 12 Phoenix SwinUnet)
# =============================================================================
class SwinV2BEncoder(nn.Module):
    """torchvision swin_v2_b as a 4-scale feature extractor.
    Stage taps at features 1/3/5/7 -> strides 4/8/16/32,
    channels 128/256/512/1024. Outputs permuted NHWC -> NCHW."""
    OUT_CHANNELS = (128, 256, 512, 1024)
    TAP_INDICES = (1, 3, 5, 7)

    def __init__(self, imagenet=False):
        super().__init__()
        weights = Swin_V2_B_Weights.IMAGENET1K_V1 if imagenet else None
        m = swin_v2_b(weights=weights)
        self.features = m.features

    def expand_patch_embed(self, in_channels, verbose=True):
        """Same rule as smp's conv1 expansion: copy pretrained channels into
        the first slots, keep extras random, rescale by old_ch / new_ch."""
        old = self.features[0][0]              # Conv2d(3, 128, k=4, s=4)
        if in_channels == old.in_channels:
            return
        new = nn.Conv2d(in_channels, old.out_channels,
                        kernel_size=old.kernel_size, stride=old.stride,
                        padding=old.padding, bias=(old.bias is not None))
        with torch.no_grad():
            n = min(old.in_channels, in_channels)
            new.weight[:, :n] = old.weight[:, :n]
            new.weight.mul_(old.in_channels / float(in_channels))
            if old.bias is not None:
                new.bias.copy_(old.bias)
        self.features[0][0] = new
        if verbose:
            print(f"[INIT] patch-embed: {n} ch pretrained, "
                  f"{in_channels - n} ch random "
                  f"(rescaled x{old.in_channels / float(in_channels):.2f})")

    def forward(self, x):
        feats = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in self.TAP_INDICES:
                feats.append(x.permute(0, 3, 1, 2).contiguous())
        return feats


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class SwinUnet(nn.Module):
    """U-Net over torchvision Swin-v2-B: 4 up-blocks (32->16->8->4->2) and a
    bilinear x2 head to input resolution. model.encoder mirrors smp.Unet's
    attribute layout so the encoder checkpointing below works unchanged."""

    def __init__(self, in_channels, num_classes, imagenet=False):
        super().__init__()
        self.encoder = SwinV2BEncoder(imagenet=imagenet)
        c4, c8, c16, c32 = SwinV2BEncoder.OUT_CHANNELS
        self.b1 = DecoderBlock(c32, c16, 256)
        self.b2 = DecoderBlock(256, c8, 128)
        self.b3 = DecoderBlock(128, c4, 64)
        self.b4 = DecoderBlock(64, 0, 32)
        self.head = nn.Conv2d(32, num_classes, 3, padding=1)
        self._in_channels = in_channels

    def finalize_input_channels(self, verbose=True):
        self.encoder.expand_patch_embed(self._in_channels, verbose=verbose)

    def forward(self, x):
        f4, f8, f16, f32 = self.encoder(x)
        d = self.b1(f32, f16)
        d = self.b2(d, f8)
        d = self.b3(d, f4)
        d = self.b4(d)
        logits = self.head(d)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear",
                             align_corners=False)


def build_model(arch, in_channels, num_classes):
    if arch == "swin_b":
        # ImageNet start, mirroring ENCODER_WEIGHTS="imagenet" for resnets.
        model = SwinUnet(in_channels, num_classes, imagenet=True)
        model.finalize_input_channels()        # 3ch imagenet -> 4ch NAIP RGBN
        return model
    return smp.Unet(encoder_name=arch, encoder_weights=ENCODER_WEIGHTS,
                    in_channels=in_channels, classes=num_classes)


# =============================================================================
# SAMPLERS
# =============================================================================
class ShuffledGridSampler(GridGeoSampler):
    """Exhaustive grid; iteration order reshuffled each epoch (full coverage,
    per Dr. Tong's guidance). Uses the global torch RNG -> seeded."""

    def __iter__(self):
        items = list(super().__iter__())
        order = torch.randperm(len(items))
        for i in order:
            yield items[i]


# =============================================================================
# DATA
# =============================================================================
def fit_to(t, size):
    h, w = t.shape[-2], t.shape[-1]
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    t = t[..., top:top + size, left:left + size]
    h, w = t.shape[-2], t.shape[-1]
    return F.pad(t, (0, size - w, 0, size - h))


def build_augmentations(size):
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
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.15,
                                   p=0.8),
    ])


class FitSample:
    def __init__(self, size, augment=False):
        self.size = size
        self.augment = augment
        self.tf = build_augmentations(size) if augment else None

    def __call__(self, sample):
        img = sample["image"].float() / 255.0
        msk = sample["mask"].long()
        msk[msk > 6] = 0
        img = fit_to(img, self.size)
        msk = fit_to(msk, self.size)

        if self.tf is not None:
            img_np = img.permute(1, 2, 0).contiguous().numpy()
            msk_np = msk.squeeze(0).numpy().astype(np.uint8)
            out = self.tf(image=img_np, mask=msk_np)
            img = torch.from_numpy(out["image"]).permute(2, 0, 1).contiguous().float()
            msk = torch.from_numpy(out["mask"]).unsqueeze(0).long()

        sample["image"] = img
        sample["mask"] = msk
        return sample


class RobustChesapeakeCVPR(ChesapeakeCVPR):
    def __getitem__(self, query):
        try:
            return super().__getitem__(query)
        except Exception:
            sample = {
                "image": torch.zeros(4, PATCH_SIZE, PATCH_SIZE),
                "mask": torch.zeros(1, PATCH_SIZE, PATCH_SIZE, dtype=torch.long),
                "crs": self.crs,
                "bounds": query,
            }
            if self.transforms is not None:
                sample = self.transforms(sample)
            return sample


def make_loader(splits, batch_size, num_workers, train, seed):
    ds = RobustChesapeakeCVPR(
        root=ROOT,
        splits=splits,
        layers=["naip-new", "lc"],
        transforms=FitSample(PATCH_SIZE, augment=train),
        download=False,
    )
    if train:
        sampler = ShuffledGridSampler(ds, size=PATCH_SIZE, stride=STRIDE)
    else:
        sampler = GridGeoSampler(ds, size=PATCH_SIZE, stride=STRIDE)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, sampler=sampler,
        collate_fn=stack_samples, num_workers=num_workers,
        generator=g, worker_init_fn=worker_init_fn,
    )


# =============================================================================
# METRIC
# =============================================================================
def update_confusion(conf, pred, target, num_classes):
    with torch.no_grad():
        k = (target >= 0) & (target < num_classes)
        idx = num_classes * target[k].to(torch.int64) + pred[k]
        conf += torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return conf


def iou_from_confusion(conf):
    inter = torch.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    iou = inter.float() / union.clamp(min=1).float()
    return iou, union


def compute_mious(conf):
    iou7, union7 = iou_from_confusion(conf)
    miou7 = iou7[union7 > 0].mean().item()

    conf6 = conf.clone()
    conf6[IGNORE_INDEX] = 0
    iou6, union6 = iou_from_confusion(conf6)
    valid = union6 > 0
    valid[IGNORE_INDEX] = False
    miou6 = iou6[valid].mean().item()
    return iou7, miou7, miou6


# =============================================================================
# MAIN
# =============================================================================
def main():
    seed_everything(args.seed)

    print("=" * 60)
    print(f"[INFO] device={DEVICE}")
    print(f"[INFO] arch={args.arch} bs={args.bs} accum={args.accum} "
          f"(effective batch {args.bs * args.accum}) lr={args.lr:g} "
          f"seed={args.seed} epochs={args.epochs}")
    print(f"[INFO] sampler=exhaustive grid (train: shuffled order) | "
          f"patch={PATCH_SIZE} stride={STRIDE}")
    print(f"[INFO] resume={args.resume}")
    print(f"[INFO] run dir: {RUN_DIR}")
    print("=" * 60)

    train_loader = make_loader(TRAIN_SPLITS, args.bs, NUM_WORKERS, train=True, seed=args.seed)
    val_loader = make_loader(VAL_SPLITS, args.bs, NUM_WORKERS, train=False, seed=args.seed)
    print(f"[INFO] train: {len(train_loader)} steps/epoch | val: {len(val_loader)} steps/epoch")

    sample = next(iter(train_loader))
    in_channels = sample["image"].shape[1]
    print(f"[INFO] in_channels={in_channels} (4=NAIP RGB+NIR expected)")

    model = build_model(args.arch, in_channels, CLASS_SET).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] model params: {n_params:.1f}M")

    ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    dice_loss = DiceLoss(mode="multiclass", ignore_index=IGNORE_INDEX)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    start_epoch = 1
    best_miou = -1.0
    latest_ckpt_path = os.path.join(RUN_DIR, "latest_checkpoint.pth")

    if args.resume and os.path.exists(latest_ckpt_path):
        print(f"[RESUME] Loading {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt["best_miou"]
        print(f"[RESUME] Continuing from epoch {start_epoch}, "
              f"best_miou_6cls so far = {best_miou:.4f}")

    log_csv = os.path.join(RUN_DIR, "train_log.csv")
    FIELDNAMES = (["epoch", "lr", "train_loss", "val_loss", "val_miou_7cls", "val_miou_6cls"]
                  + [f"IoU_{n}" for n in CLASS_NAMES]
                  + ["train_time_sec", "val_time_sec", "epoch_time_sec", "end_timestamp"])
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    run_start = datetime.now()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- train (with gradient accumulation) ----
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            x = batch["image"].to(DEVICE)
            y = batch["mask"].squeeze(1).long().to(DEVICE)

            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                logits = model(x)
                loss = (ce_loss(logits, y) + DICE_WEIGHT * dice_loss(logits, y)) / args.accum
            scaler.scale(loss).backward()
            if (i + 1) % args.accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += loss.item() * args.accum
        # flush a leftover partial accumulation window at epoch end
        if len(train_loader) % args.accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        train_loss = running / max(len(train_loader), 1)
        train_time_sec = time.time() - epoch_t0

        # ---- validate ----
        val_t0 = time.time()
        model.eval()
        conf = torch.zeros(CLASS_SET, CLASS_SET, dtype=torch.int64, device=DEVICE)
        val_running = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(DEVICE)
                y = batch["mask"].squeeze(1).long().to(DEVICE)
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    logits = model(x)
                    val_running += (ce_loss(logits, y) + DICE_WEIGHT * dice_loss(logits, y)).item()
                pred = torch.argmax(logits, dim=1)
                conf = update_confusion(conf, pred.flatten(), y.flatten(), CLASS_SET)
        val_loss = val_running / max(len(val_loader), 1)
        val_time_sec = time.time() - val_t0

        iou7, miou7, miou6 = compute_mious(conf.cpu())

        print(f"[E{epoch:03d}] lr={current_lr:.2e} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | mIoU_7cls={miou7:.4f} | mIoU_6cls={miou6:.4f} | "
              f"per-class IoU={[round(v, 3) for v in iou7.tolist()]}")

        if miou6 > best_miou:
            best_miou = miou6
            tag = f"e{epoch:03d}_miou{miou6:.4f}"
            torch.save(model.state_dict(),
                       os.path.join(RUN_DIR, f"best_unet_{tag}.pth"))
            # "encoder." prefix: works for BOTH smp.Unet and SwinUnet since
            # both expose model.encoder. For swin the keys become
            # "encoder.features.*", which the Phoenix loader strips.
            enc_sd = {f"encoder.{k}": v for k, v in model.encoder.state_dict().items()}
            torch.save(enc_sd, os.path.join(RUN_DIR, f"best_encoder_{tag}.pth"))
            print(f"        ^ new best (mIoU_6cls={best_miou:.4f}) -> saved as {tag}")

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_miou": best_miou,
        }, latest_ckpt_path)

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_miou_7cls": miou7,
            "val_miou_6cls": miou6,
            "train_time_sec": round(train_time_sec, 1),
            "val_time_sec": round(val_time_sec, 1),
            "epoch_time_sec": round(time.time() - epoch_t0, 1),
            "end_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        for name, v in zip(CLASS_NAMES, iou7.tolist()):
            row[f"IoU_{name}"] = v
        with open(log_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

        scheduler.step()

    run_end = datetime.now()
    total_dur = run_end - run_start
    print(f"\n[TOTAL] start : {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[TOTAL] end   : {run_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[TOTAL] runtime: {total_dur}  ({total_dur.total_seconds() / 3600:.2f} hr)")
    print(f"\n[DONE] best val mIoU_6cls = {best_miou:.4f}")

    try:
        from plot_training_log import plot_run
        plot_run(RUN_DIR)
    except Exception as e:
        print(f"[WARN] plot generation failed: {e}")


if __name__ == "__main__":
    main()
