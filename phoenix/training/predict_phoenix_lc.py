"""
predict_phoenix_lc.py
---------------------
Per-quad land cover prediction from the 256px prediction tiles built by
prepare_tiles_prediction.py. One output raster per NAIP quad.

Model: satlas_aerial + Swin-V2-B U-Net (Week 13 FINAL), coord-none,
6-channel input (NAIP RGBN + NDVI + CHM), trained on 256px patches.
The model classes below are copied verbatim from predict_local_swin.py /
train_phoenix.py so state dict keys match. Do not edit them.

Inputs:
    /path/to/Phoenix_LC/tile/config.json       normalization keys
    /path/to/Phoenix_LC/tile/<stem>/meta.json  quad size/crs/grid
    /path/to/Phoenix_LC/tile/<stem>/tile_rRRRRR_cCCCCC.npz
    /path/to/Phoenix_LC/train/model_best.pth

Output:
    /path/to/Phoenix_LC/predict/lc_per_quad/landcover_<stem>.tif
    uint8 codes 1..7 (1 Tree 2 Shrub 3 Grass 4 Soil 5 Building 6 Asphalt
    7 Water), 0 = nodata, LZW, same grid/CRS as the source quad.

Notes:
    * Tiles are pure non-overlapping 256px (no padding), as decided. Class
      boundaries may show faint seams at tile edges; accepted trade-off.
    * Edge tiles were zero padded to 256 at prep time; the prediction is
      cropped back to valid_h x valid_w before writing.
    * Preprocessing (normalize_stack) uses the SAME naip_scale / chm_clip_m
      that training used, read from the tile config.json.
    * Resume-safe: a quad whose output tif already exists is skipped, and
      the tif is written to a temp name then renamed, so a killed job never
      leaves a quad that LOOKS finished.

Usage:
    python predict_phoenix_lc.py            # all quads serially
    (SLURM array: SLURM_ARRAY_TASK_ID picks one quad per task)
"""

import glob
import json
import os
import re

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.windows import Window
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_v2_b

# =============================================================================
# PATHS / SETTINGS
# =============================================================================
TILE_ROOT = "/path/to/Phoenix_LC/tile"
MODEL_PATH = "/path/to/Phoenix_LC/train/model_best.pth"
OUT_DIR = "/path/to/Phoenix_LC/predict/lc_per_quad"

COORD_MODE = "none"        # trained without coord channels -> 4 + 2 = 6 ch
NUM_CLASSES = 7            # codes 1..7: Tree Shrub Grass Soil Building Asphalt Water
CHIP = 256                 # tile size, must match the training patch size
BATCH = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REQUIRED_CFG_KEYS = ("naip_band_count", "naip_scale", "chm_clip_m")


# =============================================================================
# MODEL (copied from predict_local_swin.py, must stay identical)
# =============================================================================
class SwinV2BEncoder(nn.Module):
    """torchvision swin_v2_b as a 4-scale feature extractor.

    Taps after stages 1/3/5/7 -> strides 4/8/16/32,
    channels 128/256/512/1024. Outputs permuted NHWC -> NCHW."""
    OUT_CHANNELS = (128, 256, 512, 1024)
    TAP_INDICES = (1, 3, 5, 7)

    def __init__(self):
        super().__init__()
        m = swin_v2_b(weights=None)     # weights come from the checkpoint
        self.features = m.features

    def expand_patch_embed(self, in_channels, verbose=True):
        """Replace the 3ch patch-embed conv with an in_channels version.
        Must be called BEFORE load_state_dict here, so the module shape
        matches the 6-channel weights stored in the checkpoint."""
        old = self.features[0][0]
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
            print(f"[INIT] patch-embed expanded to {in_channels} channels")

    def forward(self, x):
        feats = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in self.TAP_INDICES:
                feats.append(x.permute(0, 3, 1, 2).contiguous())
        return feats                     # [f4, f8, f16, f32]


class DecoderBlock(nn.Module):
    """Upsample x2, concat skip, two Conv-BN-ReLU. Mirrors smp's UnetDecoder."""

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
    """U-Net over torchvision Swin-V2-B: 4 up-blocks + bilinear x2 head."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.encoder = SwinV2BEncoder()
        c4, c8, c16, c32 = SwinV2BEncoder.OUT_CHANNELS
        self.b1 = DecoderBlock(c32, c16, 256)   # 1/32 -> 1/16
        self.b2 = DecoderBlock(256, c8, 128)    # 1/16 -> 1/8
        self.b3 = DecoderBlock(128, c4, 64)     # 1/8  -> 1/4
        self.b4 = DecoderBlock(64, 0, 32)       # 1/4  -> 1/2 (no skip)
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


# =============================================================================
# from phoenix_common.py (byte-identical to training preprocessing)
# =============================================================================
def normalize_stack(img_hwc, cfg):
    img = img_hwc.astype(np.float32, copy=True)
    nb = int(cfg["naip_band_count"])
    img[..., :nb] = img[..., :nb] / max(float(cfg["naip_scale"]), 1.0)
    ndvi = img[..., nb]
    if bool(cfg.get("ndvi_auto_rescale", True)):
        finite = ndvi[np.isfinite(ndvi)]
        if finite.size and (finite.max() > 2.0 or finite.min() < -2.0):
            ndvi = ndvi / 10000.0
    img[..., nb] = (np.clip(ndvi, -1.0, 1.0) + 1.0) / 2.0
    clip = float(cfg["chm_clip_m"])
    img[..., nb + 1] = np.clip(img[..., nb + 1], 0.0, clip) / clip
    return np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def check_against_summary(in_channels):
    """Cross-check this script's assumptions against the run's summary.json.
    Cheapest guard against pointing MODEL_PATH at the wrong run."""
    path = os.path.join(os.path.dirname(MODEL_PATH), "summary.json")
    if not os.path.exists(path):
        print(f"[WARN] no summary.json next to the checkpoint, skipping "
              f"cross-check ({path})")
        return

    with open(path, encoding="utf-8") as f:
        s = json.load(f)

    print(f"[INFO] run      {s.get('arch')} / {s.get('init')} / "
          f"coord={s.get('coord')} / patch={s.get('patch')} / "
          f"seed={s.get('seed')}")

    problems = []
    if int(s.get("in_channels", in_channels)) != in_channels:
        problems.append(f"in_channels: summary={s['in_channels']} "
                        f"script={in_channels}")
    if s.get("coord") is not None and s["coord"] != COORD_MODE:
        problems.append(f"coord: summary={s['coord']} script={COORD_MODE}")
    if len(s.get("class_names", [])) not in (0, NUM_CLASSES):
        problems.append(f"classes: summary={len(s['class_names'])} "
                        f"script={NUM_CLASSES}")
    if problems:
        raise SystemExit("[FATAL] script does not match the trained run:\n  "
                         + "\n  ".join(problems))

    if s.get("patch") and int(s["patch"]) != CHIP:
        print(f"[WARN] trained on {s['patch']}px patches but CHIP={CHIP}")


# =============================================================================
# MAIN
# =============================================================================
TILE_RE = re.compile(r"tile_r(\d+)_c(\d+)\.npz$")


def quad_dirs():
    """Finished quad tile folders (meta.json present), sorted by stem."""
    dirs = []
    for d in sorted(glob.glob(os.path.join(TILE_ROOT, "m_*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "meta.json")):
            dirs.append(d)
    return dirs


def predict_quad(model, cfg, quad_dir):
    stem = os.path.basename(quad_dir)
    out_path = os.path.join(OUT_DIR, f"landcover_{stem}.tif")
    if os.path.exists(out_path):
        print(f"[SKIP] {stem}: output exists")
        return

    with open(os.path.join(quad_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    W, H = int(meta["width"]), int(meta["height"])
    transform = Affine(*meta["transform"])
    crs = CRS.from_string(meta["crs"])

    tiles = sorted(glob.glob(os.path.join(quad_dir, "tile_*.npz")))
    if len(tiles) != int(meta["n_tiles"]):
        raise SystemExit(f"[FATAL] {stem}: {len(tiles)} npz files but "
                         f"meta says {meta['n_tiles']}; tile prep incomplete?")
    print(f"[INFO] {stem}: {W}x{H}, {len(tiles)} tiles")

    profile = {"driver": "GTiff", "height": H, "width": W, "count": 1,
               "dtype": "uint8", "crs": crs, "transform": transform,
               "nodata": 0, "compress": "LZW", "tiled": True,
               "BIGTIFF": "IF_SAFER"}

    tmp_path = out_path + ".part.tif"
    n_done = 0
    with rasterio.open(tmp_path, "w", **profile) as dst:
        batch_imgs, batch_infos = [], []

        def flush():
            nonlocal n_done
            if not batch_imgs:
                return
            x = torch.from_numpy(np.stack(batch_imgs)).to(DEVICE)
            with torch.no_grad(), torch.amp.autocast(
                    "cuda", enabled=(DEVICE.type == "cuda")):
                pred = torch.argmax(model(x), dim=1).cpu().numpy().astype(np.uint8)
            for i, (r0, c0, vh, vw) in enumerate(batch_infos):
                lc = (pred[i, :vh, :vw] + 1).astype(np.uint8)
                dst.write(lc, 1, window=Window(c0, r0, vw, vh))
            n_done += len(batch_imgs)
            batch_imgs.clear()
            batch_infos.clear()
            if n_done % (BATCH * 50) == 0:
                print(f"[.. ] {n_done}/{len(tiles)} tiles")

        for tp in tiles:
            m = TILE_RE.search(os.path.basename(tp))
            if not m:
                raise SystemExit(f"[FATAL] unexpected tile name: {tp}")
            r0, c0 = int(m.group(1)), int(m.group(2))
            with np.load(tp) as z:
                img = z["image"]                      # (256, 256, 6) raw
                vh, vw = int(z["valid_h"]), int(z["valid_w"])
            img = normalize_stack(img, cfg)
            # COORD_MODE == "none": no coord channels appended
            batch_imgs.append(np.transpose(img, (2, 0, 1)).astype(np.float32))
            batch_infos.append((r0, c0, vh, vw))
            if len(batch_imgs) == BATCH:
                flush()
        flush()

    os.replace(tmp_path, out_path)
    print(f"[DONE] {stem} -> {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg_path = os.path.join(TILE_ROOT, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if not all(k in cfg for k in REQUIRED_CFG_KEYS):
        raise SystemExit(f"[FATAL] {cfg_path} missing keys {REQUIRED_CFG_KEYS}")
    nb = int(cfg["naip_band_count"])
    in_channels = nb + 2                       # NAIP(4) + NDVI + CHM = 6

    print(f"[INFO] config   {cfg_path}")
    print(f"[INFO] naip_scale={cfg['naip_scale']}  chm_clip_m={cfg['chm_clip_m']}")
    print(f"[INFO] device={DEVICE}  in_channels={in_channels}  "
          f"classes={NUM_CLASSES}  coord={COORD_MODE}")
    check_against_summary(in_channels)

    quads = quad_dirs()
    if not quads:
        raise SystemExit(f"[FATAL] no finished quad folders under {TILE_ROOT}")

    # SLURM array mode: one quad per task, same sorted-index scheme as prep.
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        idx = int(task_id)
        if idx >= len(quads):
            print(f"[INFO] array task {idx}: only {len(quads)} quads, "
                  f"nothing to do")
            return
        quads = [quads[idx]]
        print(f"[INFO] array task {idx} -> {os.path.basename(quads[0])}")

    model = SwinUnet(in_channels, NUM_CLASSES)
    # Order matters: expand the patch embed to 6 channels FIRST, then load
    # the checkpoint, which was saved with the 6-channel patch embed.
    model.finalize_input_channels()
    ck = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(ck, dict) and "model" in ck:      # wrapped checkpoint
        ck = ck["model"]
    missing, unexpected = model.load_state_dict(ck, strict=False)
    if missing or unexpected:
        # Anything listed here means an architecture mismatch. Stop and check
        # rather than predicting garbage silently.
        raise RuntimeError(f"state dict mismatch:\n  missing={missing}\n"
                           f"  unexpected={unexpected}")
    model.to(DEVICE).eval()

    for qd in quads:
        predict_quad(model, cfg, qd)

    print("[DONE] all assigned quads finished")


if __name__ == "__main__":
    main()
