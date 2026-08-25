# encoder_experiment_week13.py   (Week 13)
# =============================================================================
# Experiment 3 -- COORD x PATCH-SIZE grid, on top of the Week 12 encoder arms
# (Phoenix 7-class, Windows local).
#
# Four conditions per (arch, init, seed):
#   coord-none_128 : no coordinate channels, 128 tiles
#   coord-none_256 : no coordinate channels, 256 tiles
#   coord-xy_128   : CoordConv xy channels,  128 tiles
#   coord-xy_256   : CoordConv xy channels,  256 tiles
#
# Tiles (flat layout: train/ val/ config.json directly under the size folder):
#   C:\Users\Insoo\Desktop\2026 Summer\Dr. Tong\Week 13\Tile\128\
#   C:\Users\Insoo\Desktop\2026 Summer\Dr. Tong\Week 13\Tile\256\
# (the old "tiles\train" nesting is also auto-detected, so folders copied
#  straight from Sol work unchanged)
#
# Results:
#   C:\Users\Insoo\Desktop\2026 Summer\Dr. Tong\Week 13\Train\
#       coord-none_128\<init>_<arch>_bs4_seed<N>\<timestamp>\...
#       coord-none_256\...
#       coord-xy_128\...
#       coord-xy_256\...
#       results.csv           one row per arch x init x coord x patch x seed
#       results_summary.csv   mean +/- std across seeds per condition
#
# RESUME: a run whose (arch, init, coord, patch, seed) already has a row in
# results.csv is SKIPPED, so the grid can be stopped and relaunched freely.
# Delete the row (or use --rerun) to redo a run.
#
# Arms are inherited from Week 12 (same checkpoints, same loaders):
#   resnet18 : random | imagenet | rsc | chesapeake
#   resnet50 : random | imagenet | chesapeake | satlas_s2
#   swin_b   : random | imagenet | chesapeake | satlas_aerial | satlas_s2
#
# Fixed (identical to Week 11/12): bs=4, epochs=100, lr=1e-4, AdamW(wd=1e-4),
# CosineAnnealingLR, CE(class_weight, ignore 255) + 0.7*Dice, albumentations
# aug (geom on full stack, radiometric on NAIP only).
#
# Usage:
#   # smoke test first (one arm, one condition, 2 epochs):
#   python encoder_experiment_week13.py --archs swin_b --inits chesapeake \
#       --seeds 1 --patches 128 --coords xy --epochs 2
#   # full grid (156 runs, stop/relaunch whenever):
#   python encoder_experiment_week13.py
#   # per-arch chunks:
#   python encoder_experiment_week13.py --archs swin_b
#   python encoder_experiment_week13.py --archs resnet50 resnet18
#
# NOTE: phoenix_common.py must sit in the SAME folder as this file.
# =============================================================================
from pathlib import Path
import argparse
import csv
import json
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

try:
    import torchvision
    from torchvision.models import swin_v2_b, Swin_V2_B_Weights
except ImportError as e:
    raise SystemExit(
        "[FATAL] torchvision is required for the swin_b arms "
        "(pip install torchvision matching your torch version): " + str(e))

# phoenix_common.py lives in ../training in this repo layout; make it importable.
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from phoenix_common import coord_channels, coord_num_channels, normalize_stack

# =============================================================================
# PATHS (Windows local, Week 13)
# =============================================================================
WEEK11_ROOT = Path(r"C:\Users\Insoo\Desktop\2026 Summer\Dr. Tong\Week 11\Encoder_experiment_PHX")
WEEK12_ROOT = Path(r"C:\Users\Insoo\Desktop\2026 Summer\Dr. Tong\Week 12\Encoder_experiment")
TILE_BASE   = Path(r"C:\Users\Insoo\Desktop\2026 Summer\Dr. Tong\Week 13\Tile")
TRAIN_ROOT  = Path(r"C:\Users\Insoo\Desktop\2026 Summer\Dr. Tong\Week 13\Train")
RESULT_ROOT = TRAIN_ROOT                      # results.csv lives here

# Checkpoints per architecture. 'random'/'imagenet' need no file.
CKPT_BY_ARCH = {
    "resnet50": {
        # OUR resnet50 land-cover encoder ('encoder.'-prefixed keys)
        "chesapeake": WEEK11_ROOT / "best_encoder_e025_miou0.7288.pth",
        # SatlasPretrain Sentinel-2 ResNet50 (10 m RGB) -> resolution mismatch arm
        "satlas_s2":  WEEK12_ROOT / "sentinel2_resnet50_si_rgb.pth",
    },
    "resnet18": {
        "rsc":        Path(r"E:\2026 Spring\Dr. Tong\Week7\unet-resnet18.pt"),
        "chesapeake": WEEK11_ROOT / "best_encoder_e032_miou0.7219.pth",
    },
    "swin_b": {
        # SatlasPretrain aerial (0.5-2 m/px) -> the foundation-model arm
        "satlas_aerial": WEEK12_ROOT / "aerial_swinb_si.pth",
        # SatlasPretrain Sentinel-2 (10 m RGB) -> same family, wrong resolution
        "satlas_s2":     WEEK12_ROOT / "sentinel2_swinb_si_rgb.pth",
        # OUR Swin-v2-B pretrained on ChesapeakeCVPR (NAIP RGBN, 4ch)
        "chesapeake":    TRAIN_ROOT / "Swin_best_encoder_e039_miou0.7386.pth",
    },
}

# Default arms per architecture (used when --inits is not given).
DEFAULT_INITS = {
    "resnet50": ["random", "imagenet", "chesapeake", "satlas_s2"],
    "resnet18": ["random", "imagenet", "rsc", "chesapeake"],
    "swin_b":   ["random", "imagenet", "chesapeake", "satlas_aerial", "satlas_s2"],
}
ALL_INITS = ["random", "imagenet", "rsc", "chesapeake", "satlas_aerial", "satlas_s2"]
ALL_ARCHS = ["swin_b", "resnet50", "resnet18"]

# The experiment grid axes.
ALL_PATCHES = [128, 256]
ALL_COORDS  = ["none", "xy"]

# =============================================================================
# FIXED EXPERIMENT CONFIG (identical to Week 11/12)
# =============================================================================
BS           = 4
EPOCHS       = 100
LR           = 1e-4
NUM_WORKERS  = 0           # Windows-safe
WEIGHT_DECAY = 1e-4
DICE_WEIGHT  = 0.7
MAX_CLASS_WEIGHT = 10.0

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = (DEVICE.type == "cuda")


def condition_name(coord, patch):
    return f"coord-{coord}_{patch}"


# =============================================================================
# AUGMENTATION (identical params to train_phoenix.py / Week 11)
# =============================================================================
def build_geometric_augs(size):
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
    return A.RandomBrightnessContrast(brightness_limit=0.05,
                                      contrast_limit=0.15, p=0.8)


# =============================================================================
# DATASET (identical to Week 11/12; coord_mode is now a per-run parameter)
# =============================================================================
class PhoenixTiles(Dataset):
    def __init__(self, npz_paths, cfg, coord_mode, augment):
        self.paths = list(npz_paths)
        self.cfg = cfg
        self.coord_mode = coord_mode
        self.augment = augment
        self.naip_ch = int(cfg["naip_band_count"])
        self.geo_tf = self.rad_tf = None
        if augment:
            h, w = cfg["tile_size"]
            assert h == w, "geometric augs assume square tiles"
            self.geo_tf = build_geometric_augs(h)
            self.rad_tf = build_radiometric_aug()

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _read_mask(z):
        if "mask" in z:
            return z["mask"]
        if "mask4" in z:
            return z["mask4"]
        raise KeyError("npz has neither 'mask' nor 'mask4'")

    def __getitem__(self, idx):
        with np.load(self.paths[idx]) as z:
            img = z["image"].astype(np.float32)
            mask = self._read_mask(z).astype(np.uint8)
            tr = z["transform"] if "transform" in z else None

        img = normalize_stack(img, self.cfg)
        if self.coord_mode != "none":
            if tr is None:
                raise KeyError(f"'transform' missing in {self.paths[idx]}")
            cc = coord_channels(tr, img.shape[0], img.shape[1],
                                self.cfg["coord_bounds"], self.coord_mode)
            img = np.concatenate([img, cc], axis=2)

        if self.augment:
            out = self.geo_tf(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
            naip = np.ascontiguousarray(img[..., :self.naip_ch])
            img[..., :self.naip_ch] = self.rad_tf(image=naip)["image"]

        img = np.transpose(img, (2, 0, 1))
        return (torch.from_numpy(np.ascontiguousarray(img, np.float32)),
                torch.from_numpy(np.ascontiguousarray(mask.astype(np.int64))))


# =============================================================================
# GENERIC CHECKPOINT KEY HANDLING (identical to Week 12)
# =============================================================================
PREFIX_CANDIDATES = ["", "encoder.", "backbone.backbone.", "backbone.resnet.",
                     "backbone.", "module.", "model."]


def strip_best_prefix(sd, ref_keys):
    best_prefix, best_stripped, best_n = "", dict(sd), -1
    for p in PREFIX_CANDIDATES:
        if p:
            stripped = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
        else:
            stripped = dict(sd)
        n = sum(1 for k in stripped if k in ref_keys)
        if n > best_n:
            best_prefix, best_stripped, best_n = p, stripped, n
    return best_prefix, best_stripped, best_n


def load_raw_state_dict(ckpt_path):
    ckpt = torch.load(Path(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    return None


# =============================================================================
# RESNET ENCODER LOADER (identical to Week 12)
# =============================================================================
def load_encoder_ckpt(model, ckpt_path, naip_ch, in_channels, verbose=True):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"[INIT] checkpoint NOT found: {ckpt_path} -> SKIP this arm")
        return False

    full_sd = load_raw_state_dict(ckpt_path)
    if full_sd is None:
        print(f"[INIT] unexpected checkpoint format -> SKIP")
        return False

    model_sd = model.encoder.state_dict()
    prefix, enc_sd, n_hit = strip_best_prefix(full_sd, set(model_sd.keys()))
    if n_hit == 0:
        print(f"[INIT] no matching encoder keys in {ckpt_path.name} "
              f"(tried prefixes {PREFIX_CANDIDATES}) -> SKIP")
        return False
    if verbose and prefix:
        print(f"[INIT] key prefix detected: '{prefix}'")

    new_sd = {}
    conv1_note = "conv1: not found in checkpoint"
    for k, v in enc_sd.items():
        if k not in model_sd:
            continue
        if k == "conv1.weight":
            w = model_sd[k].clone()
            n = min(naip_ch, v.shape[1], in_channels)
            if v.shape[2:] != w.shape[2:]:
                continue
            w[:, :n] = v[:, :n]
            new_sd[k] = w * (v.shape[1] / float(in_channels))
            conv1_note = (f"conv1: {n} ch from pretrained ({v.shape[1]}ch ckpt), "
                          f"{in_channels - n} extra random "
                          f"(rescaled x{v.shape[1] / float(in_channels):.2f})")
        else:
            if v.shape != model_sd[k].shape:
                continue
            new_sd[k] = v

    n_model, n_match = len(model_sd), len(new_sd)
    ratio = n_match / max(n_model, 1)
    if verbose:
        print(f"[INIT] source = {ckpt_path.name}")
        print(f"[INIT] encoder tensors matched: {n_match}/{n_model} "
              f"({ratio:.0%})   |   {conv1_note}")
    if ratio < 0.5:
        print(f"[INIT] WARNING: only {ratio:.0%} matched -> architecture "
              f"mismatch. SKIP this arm.")
        return False

    model.encoder.load_state_dict(new_sd, strict=False)
    return True


# =============================================================================
# SWIN-V2-B ENCODER (torchvision) + U-NET STYLE DECODER (identical to Week 12)
# =============================================================================
class SwinV2BEncoder(nn.Module):
    """torchvision swin_v2_b as a 4-scale feature extractor.

    features layout: 0 patch-embed, 1 stage1, 2 merge, 3 stage2, 4 merge,
    5 stage3, 6 merge, 7 stage4. Outputs are NHWC; we collect after stages
    1/3/5/7 and permute to NCHW -> strides 4/8/16/32, channels 128/256/512/1024.
    (128 input works too: torchvision zeroes the window shift when a stage's
    feature map is smaller than the attention window.)
    """
    OUT_CHANNELS = (128, 256, 512, 1024)
    TAP_INDICES = (1, 3, 5, 7)

    def __init__(self, imagenet=False):
        super().__init__()
        weights = Swin_V2_B_Weights.IMAGENET1K_V1 if imagenet else None
        m = swin_v2_b(weights=weights)
        self.features = m.features            # keys: 'features.*'

    def expand_patch_embed(self, in_channels, verbose=True):
        """Replace the 3ch patch-embed conv with an in_channels version.
        Same rule as the ResNet conv1 loader: copy pretrained channels into
        the first slots, keep the rest random, rescale by (old_ch / new_ch).
        Call this AFTER loading any pretrained checkpoint."""
        old = self.features[0][0]             # Conv2d(3, 128, k=4, s=4)
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
            print(f"[INIT] patch-embed: {n} ch from pretrained, "
                  f"{in_channels - n} extra random "
                  f"(rescaled x{old.in_channels / float(in_channels):.2f})")

    def forward(self, x):
        feats = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in self.TAP_INDICES:
                feats.append(x.permute(0, 3, 1, 2).contiguous())
        return feats                          # [f4, f8, f16, f32]


class DecoderBlock(nn.Module):
    """Upsample x2, concat skip, two Conv-BN-ReLU. Mirrors smp's UnetDecoder
    block so the Swin decoder stays as close as possible to the ResNet runs."""

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
    """U-Net over the torchvision Swin-v2-B encoder (4 up-blocks + final x2)."""

    def __init__(self, in_channels, num_classes, imagenet=False):
        super().__init__()
        self.encoder = SwinV2BEncoder(imagenet=imagenet)
        c4, c8, c16, c32 = SwinV2BEncoder.OUT_CHANNELS
        self.b1 = DecoderBlock(c32, c16, 256)   # 1/32 -> 1/16
        self.b2 = DecoderBlock(256, c8, 128)    # 1/16 -> 1/8
        self.b3 = DecoderBlock(128, c4, 64)     # 1/8  -> 1/4
        self.b4 = DecoderBlock(64, 0, 32)       # 1/4  -> 1/2 (no skip)
        self.head = nn.Conv2d(32, num_classes, 3, padding=1)
        self._in_channels = in_channels

    def finalize_input_channels(self, verbose=True):
        # called after any checkpoint loading
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


PATCH_EMBED_KEY = "features.0.0.weight"       # Conv2d(in_ch, 128, 4, 4)


def load_swin_ckpt(swin_encoder, ckpt_path, verbose=True):
    """Load a swin checkpoint into SwinV2BEncoder (Satlas 3ch or our 4ch
    Chesapeake; the encoder is resized to the checkpoint's channel count
    first so the pretrained NIR filter carries across). Identical to Week 12."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"[INIT] checkpoint NOT found: {ckpt_path} -> SKIP this arm")
        return False

    full_sd = load_raw_state_dict(ckpt_path)
    if full_sd is None:
        print(f"[INIT] unexpected checkpoint format -> SKIP")
        return False

    ref_sd = swin_encoder.state_dict()        # keys: 'features.*'
    prefix, stripped, n_hit = strip_best_prefix(full_sd, set(ref_sd.keys()))

    # resize patch embed to the checkpoint's channel count before matching
    w = stripped.get(PATCH_EMBED_KEY)
    if w is not None and w.ndim == 4:
        ckpt_ch = int(w.shape[1])
        cur_ch = swin_encoder.features[0][0].in_channels
        if ckpt_ch != cur_ch:
            swin_encoder.expand_patch_embed(ckpt_ch, verbose=False)
            ref_sd = swin_encoder.state_dict()
        if verbose:
            print(f"[INIT] checkpoint patch-embed is {ckpt_ch}ch "
                  f"({'RGBN, NIR included' if ckpt_ch >= 4 else 'RGB only, NIR stays random'})")

    matched = {k: v for k, v in stripped.items()
               if k in ref_sd and v.shape == ref_sd[k].shape}
    ratio = len(matched) / max(len(ref_sd), 1)
    if verbose:
        print(f"[INIT] source = {ckpt_path.name} (prefix '{prefix}')")
        print(f"[INIT] swin tensors matched: {len(matched)}/{len(ref_sd)} "
              f"({ratio:.0%})")
    if ratio < 0.5:
        print(f"[INIT] WARNING: only {ratio:.0%} matched -> key layout "
              f"mismatch with torchvision swin_v2_b. SKIP this arm.")
        return False

    swin_encoder.load_state_dict(matched, strict=False)
    return True


# =============================================================================
# MODEL FACTORY (arch is a parameter now, not a global)
# =============================================================================
def build_model(arch, init, in_channels, num_classes, naip_ch):
    print(f"\n[INIT] mode = {init} | arch = {arch}")
    ckpts = CKPT_BY_ARCH[arch]

    # ---------------- Swin-v2-B arms ----------------
    if arch == "swin_b":
        if init == "random":
            model = SwinUnet(in_channels, num_classes, imagenet=False)
            print("[INIT] source = none (random init, by design)")
        elif init == "imagenet":
            model = SwinUnet(in_channels, num_classes, imagenet=True)
            print("[INIT] source = torchvision Swin_V2_B IMAGENET1K_V1")
        elif init in ("satlas_aerial", "satlas_s2", "chesapeake"):
            ckpt = ckpts.get(init)
            if ckpt is None:
                print(f"[INIT] arm '{init}' has no checkpoint for swin_b -> SKIP")
                return None
            model = SwinUnet(in_channels, num_classes, imagenet=False)
            if not load_swin_ckpt(model.encoder, ckpt):
                return None
        else:
            print(f"[INIT] arm '{init}' not applicable to swin_b -> SKIP")
            return None
        model.finalize_input_channels()
        return model.to(DEVICE)

    # ---------------- ResNet arms (smp U-Net) ----------------
    if init == "random":
        model = smp.Unet(encoder_name=arch, encoder_weights=None,
                         in_channels=in_channels, classes=num_classes)
        print("[INIT] source = none (random init, by design)")
        return model.to(DEVICE)

    if init == "imagenet":
        model = smp.Unet(encoder_name=arch, encoder_weights="imagenet",
                         in_channels=in_channels, classes=num_classes)
        print("[INIT] source = smp built-in ImageNet (conv1 3ch->Nch expanded by smp)")
        return model.to(DEVICE)

    if init in ("rsc", "chesapeake", "satlas_s2"):
        ckpt = ckpts.get(init)
        if ckpt is None:
            print(f"[INIT] arm '{init}' has no checkpoint for arch={arch} -> SKIP")
            return None
        model = smp.Unet(encoder_name=arch, encoder_weights=None,
                         in_channels=in_channels, classes=num_classes)
        if not load_encoder_ckpt(model, ckpt, int(naip_ch), in_channels):
            return None
        return model.to(DEVICE)

    print(f"[INIT] arm '{init}' not applicable to {arch} -> SKIP")
    return None


# =============================================================================
# LOSS / METRIC / WEIGHTS (identical to Week 11/12)
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


def update_confusion(conf, pred, target, num_classes):
    k = (target >= 0) & (target < num_classes)
    idx = num_classes * target[k].to(torch.int64) + pred[k]
    conf += torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)
    return conf


def ious_from_confusion(conf):
    inter = torch.diag(conf).float()
    union = (conf.sum(0) + conf.sum(1) - torch.diag(conf)).float()
    iou = torch.where(union > 0, inter / union.clamp(min=1),
                      torch.full_like(inter, float("nan")))
    valid = union > 0
    miou = float(iou[valid].mean()) if bool(valid.any()) else float("nan")
    return iou, miou


def class_weights_from_cfg(cfg, class_names):
    counts = np.array([cfg["class_pixel_counts"][n] for n in class_names], dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (len(counts) * counts)
    w = np.minimum(w / w.mean(), MAX_CLASS_WEIGHT)
    return w.astype(np.float32)


# =============================================================================
# ONE TRAINING RUN (Week 12 loop; coord/patch are parameters now)
# =============================================================================
def train_one(arch, init, coord, patch, cfg, train_tiles, val_tiles, epochs, seed):
    num_classes = int(cfg["num_classes"])
    class_names = cfg["class_names"]
    ignore_index = int(cfg["ignore_index"])
    naip_ch = int(cfg["naip_band_count"])
    in_channels = int(cfg["base_in_channels"]) + coord_num_channels(coord)
    shrub_idx = class_names.index("Shrub") if "Shrub" in class_names else None

    seed_everything(seed)
    model = build_model(arch, init, in_channels, num_classes, naip_ch)
    if model is None:
        return None

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = (TRAIN_ROOT / condition_name(coord, patch)
               / f"{init}_{arch}_bs{BS}_seed{seed}" / stamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    dl_tr = DataLoader(PhoenixTiles(train_tiles, cfg, coord, augment=True),
                       batch_size=BS, shuffle=True, drop_last=False,
                       num_workers=NUM_WORKERS, pin_memory=AMP_ENABLED)
    dl_va = DataLoader(PhoenixTiles(val_tiles, cfg, coord, augment=False),
                       batch_size=BS, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=AMP_ENABLED)

    w = class_weights_from_cfg(cfg, class_names)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEVICE), ignore_index=ignore_index)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[RUN] {condition_name(coord, patch)} | init={init} arch={arch} seed={seed} "
          f"in_ch={in_channels} params={n_params:.1f}M epochs={epochs} "
          f"train={len(train_tiles)} val={len(val_tiles)} -> {run_dir}")

    csv_path = run_dir / "train_log.csv"
    cols = (["epoch", "lr", "train_loss", "val_loss", "mIoU"]
            + [f"IoU_{n}" for n in class_names] + ["epoch_time_sec", "timestamp"])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(cols)

    best_miou, best_epoch, best_iou = -1.0, -1, None
    best_shrub, best_shrub_epoch = -1.0, -1

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        cur_lr = opt.param_groups[0]["lr"]

        model.train()
        tr_loss = 0.0
        for x, y in dl_tr:
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                logits = model(x)
                loss = ce(logits, y) + DICE_WEIGHT * soft_dice_loss(
                    logits, y, num_classes, ignore_index)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tr_loss += loss.item()
        tr_loss /= max(len(dl_tr), 1)

        model.eval()
        va_loss = 0.0
        conf = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=DEVICE)
        with torch.no_grad():
            for x, y in dl_va:
                x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                    logits = model(x)
                    va_loss += (ce(logits, y) + DICE_WEIGHT * soft_dice_loss(
                        logits, y, num_classes, ignore_index)).item()
                conf = update_confusion(conf, torch.argmax(logits, 1).flatten(),
                                        y.flatten(), num_classes)
        va_loss /= max(len(dl_va), 1)

        iou, miou = ious_from_confusion(conf.cpu())
        sched.step()
        dt = time.time() - t0
        shrub_iou = float(iou[shrub_idx]) if shrub_idx is not None else float("nan")

        print(f"[{condition_name(coord, patch)}/{init}/{arch}/s{seed}] "
              f"ep {epoch:03d}/{epochs} lr={cur_lr:.2e} "
              f"train={tr_loss:.4f} val={va_loss:.4f} mIoU={miou:.4f} "
              f"Shrub={shrub_iou:.4f}")

        with csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, f"{cur_lr:.8g}", f"{tr_loss:.6f}", f"{va_loss:.6f}", f"{miou:.6f}"]
                + [f"{float(iou[i]):.6f}" for i in range(num_classes)]
                + [f"{dt:.1f}", datetime.now().isoformat(timespec="seconds")])

        if miou == miou and miou > best_miou:
            best_miou, best_epoch = miou, epoch
            best_iou = {n: float(iou[i]) for i, n in enumerate(class_names)}
            torch.save(model.state_dict(), run_dir / "model_best.pth")
        if shrub_iou == shrub_iou and shrub_iou > best_shrub:
            best_shrub, best_shrub_epoch = shrub_iou, epoch

    summary = {
        "experiment": "coord_patch_grid_week13", "init": init, "arch": arch,
        "coord": coord, "patch": patch, "bs": BS, "lr": LR, "seed": seed,
        "epochs": epochs, "in_channels": in_channels,
        "n_train": len(train_tiles), "n_val": len(val_tiles),
        "class_names": class_names, "best_epoch": best_epoch, "best_miou": best_miou,
        "best_iou_per_class": best_iou, "best_shrub_iou": best_shrub,
        "best_shrub_epoch": best_shrub_epoch,
        "ckpt": str(CKPT_BY_ARCH[arch].get(init, "")),
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] {condition_name(coord, patch)}/{init}/{arch}: "
          f"best mIoU={best_miou:.4f} @ ep{best_epoch} | "
          f"best Shrub={best_shrub:.4f} @ ep{best_shrub_epoch}")

    del model, opt, scaler
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return summary


# =============================================================================
# RESULTS AGGREGATION (one global CSV with coord + patch columns)
# =============================================================================
def results_cols(class_names):
    return (["init", "arch", "coord", "patch", "bs", "seed", "best_epoch", "mIoU"]
            + [f"IoU_{n}" for n in class_names]
            + ["best_shrub_iou", "best_shrub_epoch", "run_dir"])


def read_results():
    results_csv = RESULT_ROOT / "results.csv"
    if not results_csv.exists():
        return []
    with results_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def already_done(rows, arch, init, coord, patch, seed):
    for r in rows:
        if (r.get("arch") == arch and r.get("init") == init
                and r.get("coord") == coord and str(r.get("patch")) == str(patch)
                and str(r.get("seed")) == str(seed)):
            return True
    return False


def append_results(summary, class_names):
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    results_csv = RESULT_ROOT / "results.csv"
    cols = results_cols(class_names)
    per = summary.get("best_iou_per_class") or {}
    row = {"init": summary["init"], "arch": summary["arch"], "coord": summary["coord"],
           "patch": summary["patch"], "bs": summary["bs"], "seed": summary["seed"],
           "best_epoch": summary["best_epoch"], "mIoU": round(summary["best_miou"], 6),
           "best_shrub_iou": round(summary["best_shrub_iou"], 6),
           "best_shrub_epoch": summary["best_shrub_epoch"], "run_dir": summary["run_dir"]}
    for n in class_names:
        row[f"IoU_{n}"] = round(per.get(n, float("nan")), 6)
    new = not results_csv.exists()
    with results_csv.open("a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        if new:
            wr.writeheader()
        wr.writerow(row)
    print(f"[RESULT] appended -> {results_csv}")


def write_summary_table(class_names):
    rows = read_results()
    if not rows:
        return

    metrics = ["mIoU"] + [f"IoU_{n}" for n in class_names] + ["best_shrub_iou"]
    groups = {}
    for r in rows:
        key = (r.get("arch", ""), r["init"], r.get("coord", ""), str(r.get("patch", "")))
        groups.setdefault(key, []).append(r)

    out = []
    for (arch, init, coord, patch), sub in groups.items():
        rec = {"arch": arch, "init": init, "coord": coord, "patch": patch,
               "n_seeds": len(sub),
               "seeds": "|".join(sorted({str(r.get("seed", "")) for r in sub}))}
        for m in metrics:
            vals = np.array([float(r[m]) for r in sub
                             if r.get(m) not in (None, "")], dtype=float)
            rec[f"{m}_mean"] = round(float(np.nanmean(vals)), 4) if vals.size else float("nan")
            rec[f"{m}_std"] = round(float(np.nanstd(vals, ddof=0)), 4) if vals.size else float("nan")
        out.append(rec)
    out.sort(key=lambda r: (r["arch"], r["init"], r["patch"], r["coord"]))

    summary_csv = RESULT_ROOT / "results_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wr.writeheader()
        wr.writerows(out)

    print("\n" + "=" * 96)
    print(f"{'arch':<10}{'init':<16}{'coord':<8}{'patch':<7}{'n':<4}"
          f"{'mIoU (mean+/-std)':<24}{'Shrub (mean+/-std)':<24}")
    print("-" * 96)
    for r in out:
        print(f"{r['arch']:<10}{r['init']:<16}{r['coord']:<8}{r['patch']:<7}{r['n_seeds']:<4}"
              f"{r['mIoU_mean']:.4f} +/- {r['mIoU_std']:.4f}       "
              f"{r['best_shrub_iou_mean']:.4f} +/- {r['best_shrub_iou_std']:.4f}")
    print("=" * 96)
    print(f"[RESULT] summary -> {summary_csv}")


def print_table():
    rows = read_results()
    if not rows:
        return
    print("\n" + "=" * 88)
    print(f"{'arch':<12}{'init':<16}{'coord':<8}{'patch':<7}{'seed':<6}"
          f"{'mIoU':<10}{'Shrub IoU':<12}{'best_ep':<8}")
    print("-" * 88)
    for r in rows:
        print(f"{r.get('arch',''):<12}{r['init']:<16}{r.get('coord',''):<8}"
              f"{r.get('patch',''):<7}{r.get('seed',''):<6}{float(r['mIoU']):<10.4f}"
              f"{float(r['best_shrub_iou']):<12.4f}{r['best_epoch']:<8}")
    print("=" * 88)


# =============================================================================
# TILE DISCOVERY (flat Week 13 layout, with fallback to the Sol layout)
# =============================================================================
def find_split_dir(tile_root, split):
    flat = tile_root / split                      # Week 13 local layout
    if flat.is_dir():
        return flat
    nested = tile_root / "tiles" / split          # layout as produced on Sol
    if nested.is_dir():
        return nested
    return None


def load_tileset(patch):
    tile_root = TILE_BASE / str(patch)
    cfg_path = tile_root / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"[FATAL] config.json not found: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    for key in ("num_classes", "class_names", "ignore_index", "naip_band_count",
                "base_in_channels", "tile_size", "class_pixel_counts"):
        if key not in cfg:
            raise SystemExit(f"[FATAL] {cfg_path} missing required key: '{key}'")

    tr_dir = find_split_dir(tile_root, "train")
    va_dir = find_split_dir(tile_root, "val")
    if tr_dir is None or va_dir is None:
        raise SystemExit(f"[FATAL] no train/val folders under {tile_root}")
    train_tiles = sorted(tr_dir.glob("tile_*.npz"))
    val_tiles = sorted(va_dir.glob("tile_*.npz"))
    if not train_tiles or not val_tiles:
        raise SystemExit(f"[FATAL] no tiles under {tile_root}")

    with np.load(train_tiles[0]) as z:
        keys = set(z.keys())
    if "image" not in keys:
        raise SystemExit(f"[FATAL] {patch}: npz has no 'image' key (found {sorted(keys)})")
    if "mask" not in keys and "mask4" not in keys:
        raise SystemExit(f"[FATAL] {patch}: npz has no 'mask'/'mask4' key")
    if "transform" not in keys:
        raise SystemExit(f"[FATAL] {patch}: npz has no 'transform' (needed for coord=xy)")

    ts = cfg.get("tile_size")
    if ts and int(ts[0]) != int(patch):
        print(f"[WARN] {patch}: config tile_size={ts} does not match folder name")
    return cfg, train_tiles, val_tiles


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Week 13 coord x patch-size grid (Phoenix)")
    ap.add_argument("--archs", nargs="+", choices=ALL_ARCHS, default=ALL_ARCHS)
    ap.add_argument("--inits", nargs="+", default=None, choices=ALL_INITS,
                    help="override arms (default: per-arch set)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--patches", type=int, nargs="+", choices=ALL_PATCHES,
                    default=ALL_PATCHES)
    ap.add_argument("--coords", nargs="+", choices=ALL_COORDS, default=ALL_COORDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--rerun", action="store_true",
                    help="run even if the combo already has a row in results.csv")
    args = ap.parse_args()

    # tile sets, one per patch size
    tilesets = {p: load_tileset(p) for p in args.patches}
    class_names = tilesets[args.patches[0]][0]["class_names"]
    for p in args.patches:
        if "coord_bounds" not in tilesets[p][0] and "xy" in args.coords:
            raise SystemExit(f"[FATAL] coord=xy requested but {p} config.json "
                             f"has no 'coord_bounds'")

    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    for coord in args.coords:
        for p in args.patches:
            (TRAIN_ROOT / condition_name(coord, p)).mkdir(parents=True, exist_ok=True)

    # build the job list: seed outermost so the full grid completes at seed 1
    # before seeds 2 and 3 begin
    jobs = []
    for seed in args.seeds:
        for arch in args.archs:
            inits = args.inits if args.inits is not None else DEFAULT_INITS[arch]
            for init in inits:
                if init not in ("random", "imagenet") and init not in CKPT_BY_ARCH[arch]:
                    continue          # arm not available for this arch
                for patch in args.patches:
                    for coord in args.coords:
                        jobs.append((seed, arch, init, patch, coord))

    done_rows = read_results()
    todo = [j for j in jobs if args.rerun or
            not already_done(done_rows, j[1], j[2], j[4], j[3], j[0])]

    print("=" * 88)
    print(f"[INFO] device={DEVICE} torch={torch.__version__} "
          f"torchvision={torchvision.__version__}")
    if DEVICE.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    for p in args.patches:
        cfg, tr, va = tilesets[p]
        print(f"[INFO] patch {p}: train={len(tr)} val={len(va)} "
              f"(in_ch base={cfg['base_in_channels']})")
    print(f"[INFO] archs={args.archs} coords={args.coords} patches={args.patches} "
          f"seeds={args.seeds} bs={BS} epochs={args.epochs}")
    print(f"[INFO] jobs: {len(jobs)} total, {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to run")
    print(f"[INFO] results -> {RESULT_ROOT}")
    print("=" * 88)

    t_all = time.time()
    for k, (seed, arch, init, patch, coord) in enumerate(todo, 1):
        print(f"\n{'#'*88}\n# [{k}/{len(todo)}] {condition_name(coord, patch)} | "
              f"{arch} | init={init} | seed={seed}\n{'#'*88}")
        cfg, train_tiles, val_tiles = tilesets[patch]
        s = train_one(arch, init, coord, patch, cfg, train_tiles, val_tiles,
                      args.epochs, seed)
        if s is None:
            print(f"[RUN] arm '{init}' ({arch}, seed {seed}) skipped.")
            continue
        append_results(s, class_names)
        print_table()

    print(f"\n[RUN] total elapsed: {(time.time()-t_all)/60:.1f} min")
    print_table()
    write_summary_table(class_names)


if __name__ == "__main__":
    main()
