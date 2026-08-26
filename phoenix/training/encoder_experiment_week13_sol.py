# encoder_experiment_week13_sol.py   (Week 13, Sol)
# =============================================================================
# Experiment 3 -- PATCH-SIZE comparison on top of the Week 12 encoder arms
# (Phoenix 7-class). Sol / SLURM version.
#
# The coordinate-channel axis of the local Week 13 grid has been REMOVED.
# Only the "coord none" conditions are run here, i.e. two conditions per
# (arch, init, seed):
#   coord-none_128 : 128 px tiles, no coordinate channels
#   coord-none_256 : 256 px tiles, no coordinate channels
#
# Everything lives in the folder that holds this file (any name works,
# e.g. /path/to/Phoenix_LC/train/08.24(script,result)):
#       <here>/   this file + phoenix_common.py + the sbatch script
#       <here>/result/   one folder per run + logs + results.csv (see below)
#
# Tiles: the sets written by prepare_tiles_training.py v3 on Sol
#   /path/to/Phoenix/scripts/Tile/256/tiles_<stamp>_n<NNNN>/
#   /path/to/Phoenix/scripts/Tile/128/tiles_<stamp>_n<NNNN>/
#   Pinned to the tiles_2026-08-04_211537_* sets (the ones the local Week 13
#   runs used); change with --tile-set <substring>, or pass '' for the newest.
#   A flat layout with train/ val/ config.json directly under
#   <TILE_BASE>/<size> also works.
#
# Checkpoints: /path/to/weights/ (matched by file name)
#       Resnet18_chesapeake.pth        resnet18 chesapeake
#       Resnet18_rsc.pt                resnet18 rsc
#       Resnet50_chesapeake.pth        resnet50 chesapeake
#       sentinel2_resnet50_si_rgb.pth  resnet50 satlas_s2
#       Swinb_chesapeake.pth           swin_b   chesapeake
#       aerial_swinb_si.pth            swin_b   satlas_aerial
#       sentinel2_swinb_si_rgb.pth     swin_b   satlas_s2
#   If a file is not in that folder, the Chesapeake training tree
#   (/path/to/chesapeake_cvpr/Dataset_training) is searched by name,
#   so our own encoders do not need to be copied.
#
# Results (same layout as the local Week 13 Train folder, so the two can be
# merged later):
#   <here>/result/
#       coord-none_128/<init>_<arch>_bs4_seed<N>/<timestamp>/  model_best.pth,
#                                                              train_log.csv,
#                                                              summary.json
#       coord-none_256/...
#       logs/<condition>_<init>_<arch>_seed<N>_<jobid>.out     SLURM log per run
#       results.csv           one row per arch x init x patch x seed
#       results_summary.csv   mean +/- std across seeds per condition
#
# results.csv is REBUILT from the summary.json files (never appended), so any
# number of SLURM jobs can run in parallel without corrupting it.
# Run `--aggregate` at any time to refresh the two csv files.
#
# RESUME: a run whose (arch, init, patch, seed) already has a summary.json is
# SKIPPED, so the grid can be stopped and relaunched freely. Delete the run
# folder (or use --rerun) to redo a run.
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
# ONE SLURM JOB PER RUN: encoder_experiment_week13_sol.sh is a job array
# (--array=0-77); task k runs job k of the list printed by --list-jobs via
# --task-id/--n-tasks. Every run gets its own SLURM log, result folder and
# summary.json. (`--make-sbatch` alternatively writes one sbatch file per run
# into <here>/sbatch/ plus a submit_all.sh.)
#
# IMPORTANT (Sol): compute nodes may not reach the internet. Pre-download the
# ImageNet weights ONCE on a login node before submitting:
#   python encoder_experiment_week13_sol.py --predownload
#
# Usage (from the folder holding this file, env pytorch_gpu):
#   python encoder_experiment_week13_sol.py --check                # inputs present?
#   python encoder_experiment_week13_sol.py --list-jobs            # show the 78 runs
#   mkdir -p slurm_out && sbatch --array=0-77%8 --time=03:00:00 encoder_experiment_week13_sol.sh
#   # smoke test in an interactive GPU session (2 epochs):
#   python encoder_experiment_week13_sol.py --archs swin_b --inits chesapeake \
#       --seeds 1 --patches 128 --epochs 2
#   # rebuild results.csv / results_summary.csv from the finished runs
#   python encoder_experiment_week13_sol.py --aggregate
#
# NOTE: phoenix_common.py must sit in the SAME folder as this file.
# =============================================================================
from pathlib import Path
import argparse
import csv
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

try:
    import torchvision
    from torchvision.models import swin_v2_b, Swin_V2_B_Weights
except ImportError as e:
    raise SystemExit(
        "[FATAL] torchvision is required for the swin_b arms "
        "(pip install torchvision matching your torch version): " + str(e))

from phoenix_common import normalize_stack

# =============================================================================
# PATHS (Sol). All of these can be overridden on the command line.
# =============================================================================
# Base = the folder this file lives in (whatever it is called), so nothing
# below depends on the folder name. Override with --train-root if needed.
BASE_ROOT   = Path(__file__).resolve().parent
SCRIPT_DIR  = BASE_ROOT                       # this file, phoenix_common.py, sbatch/
TRAIN_ROOT  = BASE_ROOT / "result"            # run folders + results.csv + logs/
# Tile sets written by prepare_tiles_training.py v3 on Sol:
#   <TILE_BASE>/256/tiles_<stamp>_n<NNNN>/{tiles/train, tiles/val, config.json}
#   <TILE_BASE>/128/tiles_<stamp>_n<NNNN>/...
# The tiles_* folder matching TILE_SET (below) is used, or the newest one.
TILE_BASE   = Path("/path/to/Phoenix/scripts/Tile")
CKPT_ROOT   = Path("/path/to/weights")   # all checkpoint files, by name
CHESAPEAKE_TRAIN_ROOT = Path("/path/to/chesapeake_cvpr/Dataset_training")

# Checkpoint FILE NAMES per architecture. 'random'/'imagenet' need no file.
# First name = the file as uploaded to /path/to/weights; the other
# names are the original local / Chesapeake-training file names (fallbacks).
CKPT_NAME_BY_ARCH = {
    "resnet50": {
        # OUR resnet50 land-cover encoder ('encoder.'-prefixed keys)
        "chesapeake": ["Resnet50_chesapeake.pth", "best_encoder_e025_miou0.7288.pth"],
        # SatlasPretrain Sentinel-2 ResNet50 (10 m RGB) -> resolution mismatch arm
        "satlas_s2":  ["sentinel2_resnet50_si_rgb.pth"],
    },
    "resnet18": {
        "rsc":        ["Resnet18_rsc.pt", "unet-resnet18.pt"],
        "chesapeake": ["Resnet18_chesapeake.pth", "best_encoder_e032_miou0.7219.pth"],
    },
    "swin_b": {
        # SatlasPretrain aerial (0.5-2 m/px) -> the foundation-model arm
        "satlas_aerial": ["aerial_swinb_si.pth"],
        # SatlasPretrain Sentinel-2 (10 m RGB) -> same family, wrong resolution
        "satlas_s2":     ["sentinel2_swinb_si_rgb.pth"],
        # OUR Swin-v2-B pretrained on ChesapeakeCVPR (NAIP RGBN, 4ch)
        "chesapeake":    ["Swinb_chesapeake.pth", "Swin_best_encoder_e039_miou0.7386.pth",
                          "best_encoder_e039_miou0.7386.pth"],
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

# The experiment grid axis.
ALL_PATCHES = [128, 256]

# SLURM settings used by --make-sbatch (Sol).
SLURM = {
    "env":       "pytorch_gpu",
    "account":   "YOUR_SLURM_ACCOUNT",
    "partition": "public",
    "qos":       "public",
    "gres":      "gpu:1",
    "cpus":      8,
    "mem":       "64G",
    "time":      "1-12:00:00",
    "mail":      "YOUR_EMAIL@example.com",
}

# =============================================================================
# FIXED EXPERIMENT CONFIG (identical to Week 11/12)
# =============================================================================
BS           = 4
EPOCHS       = 100
LR           = 1e-4
NUM_WORKERS  = 0           # 0 = same DataLoader / RNG path as the local Week 13 runs
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


def worker_init_fn(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def condition_name(patch):
    # same folder name as the local Week 13 runs (coord axis is fixed to none)
    return f"coord-none_{patch}"


def resolve_ckpt(arch, init):
    """Return the checkpoint Path for (arch, init), or None if the arm has no
    file. Looks in CKPT_ROOT first, then searches the Chesapeake training tree
    by file name (our own encoders live there already)."""
    names = CKPT_NAME_BY_ARCH.get(arch, {}).get(init)
    if names is None:
        return None
    if isinstance(names, str):
        names = [names]
    for name in names:
        direct = CKPT_ROOT / name
        if direct.exists():
            return direct
    if CHESAPEAKE_TRAIN_ROOT.is_dir():
        for name in names:
            hits = sorted(CHESAPEAKE_TRAIN_ROOT.glob(f"**/{name}"))
            if hits:
                return hits[0]
    return CKPT_ROOT / names[0]        # non-existent path; the loader prints SKIP


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
# DATASET (identical to Week 11/12, without coordinate channels)
# =============================================================================
class PhoenixTiles(Dataset):
    def __init__(self, npz_paths, cfg, augment):
        self.paths = list(npz_paths)
        self.cfg = cfg
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

        img = normalize_stack(img, self.cfg)

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
        print(f"[INIT] source = {ckpt_path}")
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
        print(f"[INIT] source = {ckpt_path} (prefix '{prefix}')")
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

    # ---------------- Swin-v2-B arms ----------------
    if arch == "swin_b":
        if init == "random":
            model = SwinUnet(in_channels, num_classes, imagenet=False)
            print("[INIT] source = none (random init, by design)")
        elif init == "imagenet":
            model = SwinUnet(in_channels, num_classes, imagenet=True)
            print("[INIT] source = torchvision Swin_V2_B IMAGENET1K_V1")
        elif init in ("satlas_aerial", "satlas_s2", "chesapeake"):
            ckpt = resolve_ckpt(arch, init)
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
        ckpt = resolve_ckpt(arch, init)
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
# ONE TRAINING RUN (Week 12 loop; patch is a parameter now)
# =============================================================================
def train_one(arch, init, patch, cfg, train_tiles, val_tiles, epochs, seed, workers):
    num_classes = int(cfg["num_classes"])
    class_names = cfg["class_names"]
    ignore_index = int(cfg["ignore_index"])
    naip_ch = int(cfg["naip_band_count"])
    in_channels = int(cfg["base_in_channels"])
    shrub_idx = class_names.index("Shrub") if "Shrub" in class_names else None

    seed_everything(seed)
    model = build_model(arch, init, in_channels, num_classes, naip_ch)
    if model is None:
        return None

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = (TRAIN_ROOT / condition_name(patch)
               / f"{init}_{arch}_bs{BS}_seed{seed}" / stamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    if workers == 0:
        # EXACTLY the local Week 13 DataLoader call: shuffle draws from the
        # global torch RNG seeded by seed_everything, augmentation randomness
        # comes from the main process. Keeps the RNG stream identical to local.
        dl_tr = DataLoader(PhoenixTiles(train_tiles, cfg, augment=True),
                           batch_size=BS, shuffle=True, drop_last=False,
                           num_workers=0, pin_memory=AMP_ENABLED)
        dl_va = DataLoader(PhoenixTiles(val_tiles, cfg, augment=False),
                           batch_size=BS, shuffle=False,
                           num_workers=0, pin_memory=AMP_ENABLED)
    else:
        # multi-process loading (faster): per-worker seeding so the 8 workers
        # do not share one numpy RNG state; shuffle order pinned to the seed
        g = torch.Generator(); g.manual_seed(seed)
        dl_kw = dict(num_workers=workers, pin_memory=AMP_ENABLED,
                     worker_init_fn=worker_init_fn, generator=g,
                     persistent_workers=True)
        dl_tr = DataLoader(PhoenixTiles(train_tiles, cfg, augment=True),
                           batch_size=BS, shuffle=True, drop_last=False, **dl_kw)
        dl_va = DataLoader(PhoenixTiles(val_tiles, cfg, augment=False),
                           batch_size=BS, shuffle=False, **dl_kw)

    w = class_weights_from_cfg(cfg, class_names)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEVICE), ignore_index=ignore_index)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[RUN] {condition_name(patch)} | init={init} arch={arch} seed={seed} "
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

        print(f"[{condition_name(patch)}/{init}/{arch}/s{seed}] "
              f"ep {epoch:03d}/{epochs} lr={cur_lr:.2e} "
              f"train={tr_loss:.4f} val={va_loss:.4f} mIoU={miou:.4f} "
              f"Shrub={shrub_iou:.4f} ({dt:.0f}s)", flush=True)

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

    ckpt = resolve_ckpt(arch, init)
    summary = {
        "experiment": "coord-none_patch_week13_sol", "init": init, "arch": arch,
        "patch": patch, "bs": BS, "lr": LR, "seed": seed,
        "epochs": epochs, "in_channels": in_channels,
        "n_train": len(train_tiles), "n_val": len(val_tiles),
        "class_names": class_names, "best_epoch": best_epoch, "best_miou": best_miou,
        "best_iou_per_class": best_iou, "best_shrub_iou": best_shrub,
        "best_shrub_epoch": best_shrub_epoch,
        "ckpt": str(ckpt) if ckpt is not None else "",
        "run_dir": str(run_dir),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    # summary.json marks the run as DONE; write atomically so a killed job
    # never leaves a half-written file behind
    tmp = run_dir / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(tmp, run_dir / "summary.json")
    print(f"[DONE] {condition_name(patch)}/{init}/{arch}: "
          f"best mIoU={best_miou:.4f} @ ep{best_epoch} | "
          f"best Shrub={best_shrub:.4f} @ ep{best_shrub_epoch}")

    del model, opt, scaler
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return summary


# =============================================================================
# RESULTS AGGREGATION
# results.csv is rebuilt from every summary.json under TRAIN_ROOT, so parallel
# SLURM tasks never append to the same file.
# =============================================================================
def scan_summaries():
    """Return all finished runs as summary dicts (newest run wins if the same
    (arch, init, patch, seed) was trained more than once)."""
    out = {}
    for p in sorted(TRAIN_ROOT.glob("coord-none_*/*/*/summary.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"[WARN] unreadable summary {p}: {e}")
            continue
        key = (s.get("arch"), s.get("init"), str(s.get("patch")), str(s.get("seed")))
        out[key] = s
    return out


def already_done(done, arch, init, patch, seed):
    return (arch, init, str(patch), str(seed)) in done


def results_cols(class_names):
    return (["init", "arch", "patch", "bs", "seed", "best_epoch", "mIoU"]
            + [f"IoU_{n}" for n in class_names]
            + ["best_shrub_iou", "best_shrub_epoch", "run_dir"])


def summary_to_row(s, class_names):
    per = s.get("best_iou_per_class") or {}
    row = {"init": s["init"], "arch": s["arch"], "patch": s["patch"],
           "bs": s["bs"], "seed": s["seed"], "best_epoch": s["best_epoch"],
           "mIoU": round(s["best_miou"], 6),
           "best_shrub_iou": round(s["best_shrub_iou"], 6),
           "best_shrub_epoch": s["best_shrub_epoch"], "run_dir": s["run_dir"]}
    for n in class_names:
        row[f"IoU_{n}"] = round(per.get(n, float("nan")), 6)
    return row


def atomic_write_csv(path, fieldnames, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)
    os.replace(tmp, path)


def write_results(class_names):
    """Rebuild results.csv from the summary.json files. Returns the rows."""
    done = scan_summaries()
    rows = [summary_to_row(s, class_names) for s in done.values()]
    rows.sort(key=lambda r: (r["arch"], r["init"], int(r["patch"]), int(r["seed"])))
    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    if rows:
        atomic_write_csv(TRAIN_ROOT / "results.csv", results_cols(class_names), rows)
        print(f"[RESULT] results.csv rebuilt ({len(rows)} rows) -> {TRAIN_ROOT / 'results.csv'}")
    return rows


def write_summary_table(class_names, rows):
    if not rows:
        return

    metrics = ["mIoU"] + [f"IoU_{n}" for n in class_names] + ["best_shrub_iou"]
    groups = {}
    for r in rows:
        key = (r["arch"], r["init"], str(r["patch"]))
        groups.setdefault(key, []).append(r)

    out = []
    for (arch, init, patch), sub in groups.items():
        rec = {"arch": arch, "init": init, "patch": patch,
               "n_seeds": len(sub),
               "seeds": "|".join(sorted({str(r.get("seed", "")) for r in sub}))}
        for m in metrics:
            vals = np.array([float(r[m]) for r in sub
                             if r.get(m) not in (None, "")], dtype=float)
            rec[f"{m}_mean"] = round(float(np.nanmean(vals)), 4) if vals.size else float("nan")
            rec[f"{m}_std"] = round(float(np.nanstd(vals, ddof=0)), 4) if vals.size else float("nan")
        out.append(rec)
    out.sort(key=lambda r: (r["arch"], r["init"], int(r["patch"])))

    summary_csv = TRAIN_ROOT / "results_summary.csv"
    atomic_write_csv(summary_csv, list(out[0].keys()), out)

    print("\n" + "=" * 88)
    print(f"{'arch':<10}{'init':<16}{'patch':<7}{'n':<4}"
          f"{'mIoU (mean+/-std)':<24}{'Shrub (mean+/-std)':<24}")
    print("-" * 88)
    for r in out:
        print(f"{r['arch']:<10}{r['init']:<16}{r['patch']:<7}{r['n_seeds']:<4}"
              f"{r['mIoU_mean']:.4f} +/- {r['mIoU_std']:.4f}       "
              f"{r['best_shrub_iou_mean']:.4f} +/- {r['best_shrub_iou_std']:.4f}")
    print("=" * 88)
    print(f"[RESULT] summary -> {summary_csv}")


def print_table(rows):
    if not rows:
        return
    print("\n" + "=" * 80)
    print(f"{'arch':<12}{'init':<16}{'patch':<7}{'seed':<6}"
          f"{'mIoU':<10}{'Shrub IoU':<12}{'best_ep':<8}")
    print("-" * 80)
    for r in rows:
        print(f"{r['arch']:<12}{r['init']:<16}{r['patch']:<7}{r['seed']:<6}"
              f"{float(r['mIoU']):<10.4f}{float(r['best_shrub_iou']):<12.4f}"
              f"{r['best_epoch']:<8}")
    print("=" * 80)


# =============================================================================
# TILE DISCOVERY (flat layout, with fallback to the prepare_tiles layout)
# =============================================================================
def find_split_dir(tile_root, split):
    flat = tile_root / split                      # <root>/train
    if flat.is_dir():
        return flat
    nested = tile_root / "tiles" / split          # <root>/tiles/train
    if nested.is_dir():
        return nested
    return None


# Substring of the tiles_<stamp>_n<NNNN> folder to use. Pinned to the set the
# local Week 13 runs used (tiles_2026-08-04_211537_n0940 for 128; the 256 set
# of the same prepare run shares the stamp). None = newest folder.
TILE_SET = "2026-08-04_211537"


def resolve_tile_root(patch):
    """<TILE_BASE>/<patch> itself if it holds config.json, else the newest
    (or --tile-set matching) tiles_* subfolder that has a config.json."""
    base = TILE_BASE / str(patch)
    if (base / "config.json").exists():
        return base
    cands = sorted(d for d in base.glob("tiles_*") if (d / "config.json").exists())
    if TILE_SET:
        cands = [d for d in cands if TILE_SET in d.name]
    if not cands:
        raise SystemExit(f"[FATAL] no tile set with config.json under {base}"
                         + (f" matching '{TILE_SET}'" if TILE_SET else ""))
    if len(cands) > 1:
        print(f"[INFO] {patch}: {len(cands)} tile sets under {base}, using newest: {cands[-1].name}")
    return cands[-1]


def load_tileset(patch):
    tile_root = resolve_tile_root(patch)
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

    ts = cfg.get("tile_size")
    if ts and int(ts[0]) != int(patch):
        print(f"[WARN] {patch}: config tile_size={ts} does not match folder name")
    return cfg, train_tiles, val_tiles


# =============================================================================
# PRE-DOWNLOAD (run once on a login node; compute nodes may be offline)
# =============================================================================
def check_inputs(archs, inits_override, patches):
    """Print where every needed input is (or is not) found. No training."""
    ok = True
    print("[CHECK] tiles")
    for p in patches:
        try:
            root = resolve_tile_root(p)
        except SystemExit as e:
            print(f"  MISSING {TILE_BASE / str(p)}  ({e})")
            ok = False
            continue
        tr, va = find_split_dir(root, "train"), find_split_dir(root, "val")
        n_tr = len(list(tr.glob("tile_*.npz"))) if tr else 0
        n_va = len(list(va.glob("tile_*.npz"))) if va else 0
        cfg_ok = (root / "config.json").exists()
        good = cfg_ok and n_tr > 0 and n_va > 0
        ok &= good
        print(f"  {'OK ' if good else 'MISSING'} {root}  config={cfg_ok} train={n_tr} val={n_va}")
    print("[CHECK] checkpoints")
    for arch in archs:
        inits = inits_override if inits_override is not None else DEFAULT_INITS[arch]
        for init in inits:
            if init in ("random", "imagenet"):
                continue
            if init not in CKPT_NAME_BY_ARCH[arch]:
                continue
            path = resolve_ckpt(arch, init)
            found = path is not None and path.exists()
            ok &= found
            print(f"  {'OK ' if found else 'MISSING'} {arch:<9} {init:<14} -> {path}")
    print("[CHECK] phoenix_common.py:", "OK" if (Path(__file__).parent / "phoenix_common.py").exists()
          else "MISSING (must sit next to this script)")
    print("[CHECK]", "all inputs found" if ok else "some inputs are MISSING (see above)")


def predownload():
    print("[PRE] torchvision Swin_V2_B IMAGENET1K_V1 ...")
    swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
    for arch in ("resnet18", "resnet50"):
        print(f"[PRE] smp {arch} imagenet ...")
        smp.Unet(encoder_name=arch, encoder_weights="imagenet", in_channels=3, classes=1)
    print("[PRE] done (cached under ~/.cache/torch/hub/checkpoints)")


# =============================================================================
# SBATCH GENERATION: one file per run, one SLURM job per run
# =============================================================================
def job_tag(seed, arch, init, patch):
    return f"{condition_name(patch)}_{init}_{arch}_seed{seed}"


def write_sbatch_files(jobs, epochs, workers, rerun, script_dir, train_root):
    sb_dir = script_dir / "sbatch"
    log_dir = train_root / "logs"
    sb_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    script_name = Path(__file__).name

    files = []
    for (seed, arch, init, patch) in jobs:
        tag = job_tag(seed, arch, init, patch)
        extra = " --rerun" if rerun else ""
        body = f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={SLURM['account']}
#SBATCH --partition={SLURM['partition']}
#SBATCH --qos={SLURM['qos']}
#SBATCH --gres={SLURM['gres']}
#SBATCH --cpus-per-task={SLURM['cpus']}
#SBATCH --mem={SLURM['mem']}
#SBATCH --time={SLURM['time']}
#SBATCH --output={log_dir}/{tag}_%j.out
#SBATCH --error={log_dir}/{tag}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={SLURM['mail']}
# generated by {script_name} --make-sbatch : one run = one job
set -euo pipefail
module purge
module load mamba/latest
source activate {SLURM['env']}
cd {script_dir}
echo "[SLURM] job ${{SLURM_JOB_ID}} ({tag}) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python {script_name} \\
    --archs {arch} --inits {init} --patches {patch} --seeds {seed} \\
    --epochs {epochs} --workers {workers} \\
    --tile-base {TILE_BASE} --train-root {train_root} --ckpt-root {CKPT_ROOT}{extra}
"""
        path = sb_dir / f"{tag}.sh"
        path.write_text(body, encoding="utf-8")
        files.append(path)

    submit = sb_dir / "submit_all.sh"
    lines = ["#!/bin/bash", f"# submit every generated run ({len(files)} jobs)",
             f"cd {sb_dir}"]
    lines += [f"sbatch {f.name}" for f in files]
    submit.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[SBATCH] wrote {len(files)} job files -> {sb_dir}")
    print(f"[SBATCH] submit with:  bash {submit}")
    return files


# =============================================================================
# MAIN
# =============================================================================
def main():
    global TILE_BASE, TRAIN_ROOT, CKPT_ROOT, TILE_SET

    ap = argparse.ArgumentParser(
        description="Week 13 patch-size experiment (Phoenix, Sol)")
    ap.add_argument("--archs", nargs="+", choices=ALL_ARCHS, default=ALL_ARCHS)
    ap.add_argument("--inits", nargs="+", default=None, choices=ALL_INITS,
                    help="override arms (default: per-arch set)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--patches", type=int, nargs="+", choices=ALL_PATCHES,
                    default=ALL_PATCHES)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--rerun", action="store_true",
                    help="run even if the combo already has a summary.json")
    ap.add_argument("--tile-base", type=str, default=str(TILE_BASE))
    ap.add_argument("--tile-set", type=str, default=TILE_SET,
                    help="substring of the tiles_<stamp>_n<NNNN> folder to use "
                         "(default: %(default)s; pass '' for the newest)")
    ap.add_argument("--train-root", type=str, default=str(TRAIN_ROOT))
    ap.add_argument("--ckpt-root", type=str, default=str(CKPT_ROOT))
    ap.add_argument("--task-id", type=int, default=None,
                    help="this task's index (default: SLURM_ARRAY_TASK_ID or 0)")
    ap.add_argument("--n-tasks", type=int, default=None,
                    help="number of parallel tasks (default: SLURM_ARRAY_TASK_COUNT or 1)")
    ap.add_argument("--list-jobs", action="store_true",
                    help="print the job grid and exit")
    ap.add_argument("--make-sbatch", action="store_true",
                    help="write one sbatch file per run into script/sbatch/ and exit")
    ap.add_argument("--script-dir", type=str, default=str(SCRIPT_DIR),
                    help="folder holding this script (used by --make-sbatch)")
    ap.add_argument("--aggregate", action="store_true",
                    help="rebuild results.csv / results_summary.csv and exit")
    ap.add_argument("--predownload", action="store_true",
                    help="download ImageNet weights (login node) and exit")
    ap.add_argument("--check", action="store_true",
                    help="verify tiles / checkpoints exist and exit")
    ap.add_argument("--job-tag", action="store_true",
                    help="print the tag of this task's single job (for sbatch log names) and exit")
    args = ap.parse_args()

    TILE_BASE = Path(args.tile_base)
    TILE_SET = args.tile_set or None
    TRAIN_ROOT = Path(args.train_root)
    CKPT_ROOT = Path(args.ckpt_root)

    if args.predownload:
        predownload()
        return
    if args.check:
        check_inputs(args.archs, args.inits, args.patches)
        return

    # build the job list: seed outermost so the full grid completes at seed 1
    # before seeds 2 and 3 begin
    jobs = []
    for seed in args.seeds:
        for arch in args.archs:
            inits = args.inits if args.inits is not None else DEFAULT_INITS[arch]
            for init in inits:
                if init not in ("random", "imagenet") and init not in CKPT_NAME_BY_ARCH[arch]:
                    continue          # arm not available for this arch
                for patch in args.patches:
                    jobs.append((seed, arch, init, patch))

    # SLURM array slicing: task k of n takes jobs[k::n]
    task_id = args.task_id if args.task_id is not None else int(
        os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n_tasks = args.n_tasks if args.n_tasks is not None else int(
        os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    if not (0 <= task_id < n_tasks):
        raise SystemExit(f"[FATAL] task-id {task_id} out of range for n-tasks {n_tasks}")
    my_jobs = jobs[task_id::n_tasks]

    if args.job_tag:
        # used by the sbatch script to name the log file; no tiles needed
        print("_".join(job_tag(*j) for j in my_jobs) if my_jobs else "empty")
        return

    # tile sets, one per patch size
    tilesets = {p: load_tileset(p) for p in args.patches}
    class_names = tilesets[args.patches[0]][0]["class_names"]

    if args.aggregate:
        rows = write_results(class_names)
        print_table(rows)
        write_summary_table(class_names, rows)
        return

    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    for p in args.patches:
        (TRAIN_ROOT / condition_name(p)).mkdir(parents=True, exist_ok=True)

    done = scan_summaries()

    if args.list_jobs:
        print(f"[JOBS] {len(jobs)} total, split over {n_tasks} task(s)")
        for k, (seed, arch, init, patch) in enumerate(jobs):
            flag = "done" if already_done(done, arch, init, patch, seed) else "todo"
            print(f"  [{k:3d}] task {k % n_tasks:2d} | {job_tag(seed, arch, init, patch):<44} {flag}")
        return

    if args.make_sbatch:
        pending = [j for j in jobs if args.rerun or
                   not already_done(done, j[1], j[2], j[3], j[0])]
        print(f"[SBATCH] {len(jobs)} jobs in grid, {len(jobs) - len(pending)} already done, "
              f"{len(pending)} sbatch files to write")
        write_sbatch_files(pending, args.epochs, args.workers, args.rerun,
                           Path(args.script_dir), TRAIN_ROOT)
        return

    todo = [j for j in my_jobs if args.rerun or
            not already_done(done, j[1], j[2], j[3], j[0])]

    print("=" * 88)
    print(f"[INFO] device={DEVICE} torch={torch.__version__} "
          f"torchvision={torchvision.__version__}")
    if DEVICE.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    for p in args.patches:
        cfg, tr, va = tilesets[p]
        print(f"[INFO] patch {p}: train={len(tr)} val={len(va)} "
              f"(in_ch={cfg['base_in_channels']}) <- {tr[0].parent.parent}")
    print(f"[INFO] archs={args.archs} patches={args.patches} "
          f"seeds={args.seeds} bs={BS} epochs={args.epochs} workers={args.workers}")
    print(f"[INFO] task {task_id}/{n_tasks}: {len(my_jobs)} of {len(jobs)} jobs, "
          f"{len(my_jobs) - len(todo)} already done, {len(todo)} to run")
    print(f"[INFO] tiles   -> {TILE_BASE}")
    print(f"[INFO] ckpts   -> {CKPT_ROOT}")
    print(f"[INFO] results -> {TRAIN_ROOT}")
    print("=" * 88, flush=True)

    t_all = time.time()
    for k, (seed, arch, init, patch) in enumerate(todo, 1):
        print(f"\n{'#'*88}\n# [{k}/{len(todo)}] {condition_name(patch)} | "
              f"{arch} | init={init} | seed={seed}\n{'#'*88}", flush=True)
        cfg, train_tiles, val_tiles = tilesets[patch]
        s = train_one(arch, init, patch, cfg, train_tiles, val_tiles,
                      args.epochs, seed, args.workers)
        if s is None:
            print(f"[RUN] arm '{init}' ({arch}, seed {seed}) skipped.")
            continue
        rows = write_results(class_names)
        print_table(rows)

    print(f"\n[RUN] total elapsed: {(time.time()-t_all)/60:.1f} min")
    rows = write_results(class_names)
    print_table(rows)
    write_summary_table(class_names, rows)


if __name__ == "__main__":
    main()
