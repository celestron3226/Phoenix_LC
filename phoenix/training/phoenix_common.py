"""
phoenix_common.py
-----------------
Shared normalization + coordinate-encoding code for train_phoenix.py and
predict_phoenix.py. Keeping this in ONE place guarantees that training and
city-wide prediction use byte-identical preprocessing (the classic silent
killer in this pipeline is train/predict normalization drift).

Coordinate encoding (--coord):
  none    : no coordinate channels.
  xy      : 2 extra channels = per-pixel real EPSG:3857 x,y normalized to
            [0,1] over the Phoenix Council_District bounds (CoordConv-style).
            Simple, robust, and identical at train and prediction time.
  sincos  : multi-scale Fourier features of the same normalized coords:
            sin/cos(2^k * pi * u) for k in FREQS, for u in {x,y}
            -> 4*len(FREQS) channels. Lets the network express periodic /
            multi-scale spatial priors (block structure, irrigation grids).

Why not SatCLIP here: SatCLIP's location encoder is trained GLOBALLY; across
a single metro area (~60 km) its embedding is nearly constant, so it adds
almost no intra-city signal while adding a dependency + a frozen 256-d vector
to plumb through the U-Net. Local normalized coordinates carry strictly more
within-Phoenix information. (Easy to add later as another --coord mode.)
"""

import numpy as np

SINCOS_FREQS = (0, 1, 2, 3)   # 2^k * pi -> wavelengths ~ city, half, quarter, eighth


def coord_num_channels(mode):
    if mode == "none":
        return 0
    if mode == "xy":
        return 2
    if mode == "sincos":
        return 4 * len(SINCOS_FREQS)
    raise ValueError(f"unknown coord mode: {mode}")


def coord_channels(transform_abcdef, height, width, bounds, mode):
    """Per-pixel coordinate channels, float32 (H, W, C).

    transform_abcdef: (a, b, c, d, e, f) affine coefficients of THIS tile /
                      window (x = c + (col+0.5)*a, y = f + (row+0.5)*e for
                      axis-aligned rasters, which all Phoenix data are).
    bounds: dict with xmin/ymin/xmax/ymax (EPSG:3857, from config.json).
    """
    if mode == "none":
        return None
    a, b, c, d, e, f = [float(v) for v in transform_abcdef]
    cols = (np.arange(width, dtype=np.float64) + 0.5)
    rows = (np.arange(height, dtype=np.float64) + 0.5)
    xs = c + cols * a          # (W,)
    ys = f + rows * e          # (H,)  (e is negative for north-up rasters)

    nx = (xs - bounds["xmin"]) / max(bounds["xmax"] - bounds["xmin"], 1e-9)
    ny = (ys - bounds["ymin"]) / max(bounds["ymax"] - bounds["ymin"], 1e-9)
    gx = np.broadcast_to(nx[None, :], (height, width))
    gy = np.broadcast_to(ny[:, None], (height, width))

    if mode == "xy":
        ch = np.stack([gx, gy], axis=-1)
    elif mode == "sincos":
        feats = []
        for k in SINCOS_FREQS:
            w = (2.0 ** k) * np.pi
            feats += [np.sin(w * gx), np.cos(w * gx),
                      np.sin(w * gy), np.cos(w * gy)]
        ch = np.stack(feats, axis=-1)
    else:
        raise ValueError(f"unknown coord mode: {mode}")
    return ch.astype(np.float32)


def normalize_stack(img_hwc, cfg):
    """Normalize the raw 6-channel stack (NAIP r,g,b,nir + NDVI + CHM), HWC.

    NAIP: / naip_scale (255 for uint8)          -> [0, 1]
    NDVI: auto-rescale if stored *10000, clip to [-1,1], map -> [0, 1]
    CHM : clip [0, chm_clip_m] m, / chm_clip_m  -> [0, 1]

    Identical to the Week16 local pipeline so results stay comparable.
    """
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


# =============================================================================
# CLASS-SCHEME HELPERS  (shared by Plan A / hybrid / Plan B)
# =============================================================================
# prepare_tiles_training.py stores internal 7-class ids (code - 1):
#   0 Tree  1 Shrub  2 Grass  3 Soil  4 Building  5 Asphalt  6 Water
# Output CODES (what predictions/GeoTIFFs use) are internal + 1, i.e. 1..7.
VEG7_INTERNAL = (0, 1, 2)          # Tree, Shrub, Grass
IGNORE = 255


def real_ndvi_chm(img_hwc_raw, cfg):
    """Recover REAL NDVI (-1..1) and CHM (metres) from the RAW tile channels,
    applying the same NDVI auto-rescale that normalize_stack uses. Used by the
    height rule and the Plan-B pseudo-labeler, which need physical units."""
    nb = int(cfg["naip_band_count"])
    ndvi = np.asarray(img_hwc_raw[..., nb], dtype=np.float32)
    if bool(cfg.get("ndvi_auto_rescale", True)):
        finite = ndvi[np.isfinite(ndvi)]
        if finite.size and (finite.max() > 2.0 or finite.min() < -2.0):
            ndvi = ndvi / 10000.0
    ndvi = np.clip(np.nan_to_num(ndvi, nan=0.0), -1.0, 1.0)
    chm = np.asarray(img_hwc_raw[..., nb + 1], dtype=np.float32)
    chm = np.nan_to_num(chm, nan=0.0, posinf=0.0, neginf=0.0)
    chm[chm < -1000] = 0.0
    return ndvi, chm


def remap7_to5(mask, ignore=IGNORE):
    """7-class internal -> 5-class {0 Veg, 1 Soil, 2 Building, 3 Asphalt, 4 Water}.
    Tree/Shrub/Grass collapse into a single 'Vegetation' class (the DL target for
    the hybrid method; the CHM height rule re-splits it at prediction time)."""
    out = np.full_like(mask, ignore)
    m = mask != ignore
    out[m & np.isin(mask, [0, 1, 2])] = 0
    out[m & (mask == 3)] = 1
    out[m & (mask == 4)] = 2
    out[m & (mask == 5)] = 3
    out[m & (mask == 6)] = 4
    return out


def remap7_to4(mask, ignore=IGNORE):
    """7-class internal -> 4-class {0 Tree, 1 Shrub, 2 Grass, 3 NonVeg}.
    Soil/Building/Asphalt/Water all collapse to NonVeg. Used to score Plan B
    (whose rule ceiling is 4 classes) against the real manual labels."""
    out = np.full_like(mask, ignore)
    m = mask != ignore
    out[m & (mask == 0)] = 0
    out[m & (mask == 1)] = 1
    out[m & (mask == 2)] = 2
    out[m & np.isin(mask, [3, 4, 5, 6])] = 3
    return out


def hybrid_pred5_to_codes(pred5, chm_m, cfg):
    """Combine a 5-class DL prediction with the CHM height rule -> final CODES 1..7.
    Vegetation pixels (pred5==0) are split by height:
        CHM <  th_grass_m            -> Grass (3)
        th_grass_m <= CHM < th_tree_m-> Shrub (2)
        CHM >= th_tree_m             -> Tree  (1)
    Non-veg predictions map straight through: Soil4 Building5 Asphalt6 Water7."""
    g = float(cfg.get("th_grass_m", 0.3))
    t = float(cfg.get("th_tree_m", 1.5))
    codes = np.zeros(pred5.shape, dtype=np.uint8)
    veg = pred5 == 0
    codes[veg & (chm_m < g)] = 3
    codes[veg & (chm_m >= g) & (chm_m < t)] = 2
    codes[veg & (chm_m >= t)] = 1
    codes[pred5 == 1] = 4
    codes[pred5 == 2] = 5
    codes[pred5 == 3] = 6
    codes[pred5 == 4] = 7
    return codes


def rule_pseudolabel4(ndvi, chm_m, cfg):
    """Plan-B weak labels from physical rules -> 4-class internal
    {0 Tree, 1 Shrub, 2 Grass, 3 NonVeg}. Every pixel is labeled (no ignore):
        NDVI >= ndvi_veg_min  AND  CHM >  th_tree_m         -> Tree
        NDVI >= ndvi_veg_min  AND  th_grass_m<=CHM<=th_tree -> Shrub
        NDVI >= ndvi_veg_min  AND  CHM <  th_grass_m        -> Grass
        else                                                -> NonVeg"""
    vmin = float(cfg.get("ndvi_veg_min", 0.2))
    g = float(cfg.get("th_grass_m", 0.3))
    t = float(cfg.get("th_tree_m", 1.5))
    veg = ndvi >= vmin
    lbl = np.full(ndvi.shape, 3, dtype=np.uint8)     # NonVeg
    lbl[veg & (chm_m < g)] = 2                        # Grass
    lbl[veg & (chm_m >= g) & (chm_m <= t)] = 1        # Shrub
    lbl[veg & (chm_m > t)] = 0                        # Tree
    return lbl


# =============================================================================
# CHESAPEAKE ENCODER TRANSFER (shared by all training scripts)
# =============================================================================
def load_chesapeake_encoder(model, ckpt_path, naip_ch, in_channels, verbose=True):
    """Copy 'encoder.'-prefixed weights into model.encoder; conv1 gets the
    pretrained NAIP channels, any extra input channels (NDVI/CHM/coords) stay
    randomly initialized and the copied block is rescaled so total input
    magnitude stays comparable. Raises if the checkpoint has no encoder keys."""
    import os
    import torch
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Chesapeake encoder not found: {ckpt_path} "
            f"(refusing to silently train from random init)")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else None
    if sd is None:
        raise ValueError("Unexpected checkpoint format")
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    if not enc_sd:
        raise ValueError("No 'encoder.*' keys in checkpoint -- use best_encoder_*.pth")

    model_sd = model.encoder.state_dict()
    new_sd = {}
    for k, v in enc_sd.items():
        if k not in model_sd:
            continue
        if k == "conv1.weight":
            w = model_sd[k].clone()
            n = min(naip_ch, v.shape[1], in_channels)
            w[:, :n] = v[:, :n]
            new_sd[k] = w * (v.shape[1] / float(in_channels))
            if verbose:
                print(f"[INFO] conv1: {n} NAIP channels from pretrained, "
                      f"{in_channels - n} extra random-init "
                      f"(rescaled x{v.shape[1] / float(in_channels):.2f})")
        else:
            new_sd[k] = v
    res = model.encoder.load_state_dict(new_sd, strict=False)
    missing = res.missing_keys if res is not None else []
    unexpected = res.unexpected_keys if res is not None else []
    if verbose:
        print(f"[INFO] encoder loaded: {len(new_sd)} tensors "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return model
