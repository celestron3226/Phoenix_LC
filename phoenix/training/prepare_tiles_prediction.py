"""
prepare_tiles_prediction.py
---------------------------
Cut the full Phoenix metro NAIP + NDVI + CHM stacks into 256x256 PREDICTION
tiles (.npz). No labels involved. One folder per NAIP quad.

Inputs (all three already cut to the same NAIP grid per quad):
    NAIP : /path/to/Phoenix/NAIP_Raw/NAIP_raw/<stem>.tif
    NDVI : /path/to/Phoenix/NDVI/<stem>_ndvi.tif
    CHM  : /path/to/Phoenix/LiDAR/per_naip/<stem>_chm.tif
A quad is skipped (and logged) if the NDVI or CHM partner is missing.

Output:
    /path/to/Phoenix_LC/tile/config.json          normalization keys
                                                           COPIED from the
                                                           training tile config
    /path/to/Phoenix_LC/tile/<stem>/meta.json     quad size/crs/grid
    /path/to/Phoenix_LC/tile/<stem>/tile_rRRRR_cCCCC.npz

npz layout (same fields as the training tiles, minus the mask):
    image     float32 (256, 256, 6)  raw values, NOT normalized
              (normalization happens at predict time via config.json,
               exactly like training)
    transform float64 (6,)           affine of THIS tile window
    source    str                    "<stem>_r<row0>_c<col0>"
    valid_h   int                    real pixel rows (edge tiles are zero
    valid_w   int                    padded up to 256; crop back with these)

Notes:
    * Tiles stay on the quad's native grid and CRS (EPSG:26912). No
      reprojection, no mosaicking here.
    * naip_scale / chm_clip_m are NOT re-derived. They are copied from the
      training tile config so prediction preprocessing is byte identical
      to training. The script dies if no training config is found.
    * Resume-safe: a quad whose <stem>/meta.json already exists is skipped,
      so the job can just be resubmitted after a timeout.

Run (CPU only):
    python prepare_tiles_prediction.py
"""

import glob
import json
import os
from datetime import datetime

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

# =============================================================================
# CONFIG
# =============================================================================
NAIP_DIR = "/path/to/Phoenix/NAIP_Raw/NAIP_raw"
NDVI_DIR = "/path/to/Phoenix/NDVI"
CHM_DIR = "/path/to/Phoenix/LiDAR/per_naip"

OUT_ROOT = "/path/to/Phoenix_LC/tile"

# Training tile config to copy normalization constants from. Auto-search picks
# the newest config.json under TRAIN_CFG_SEARCH; set TRAIN_CFG_PATH to
# override with an explicit file.
TRAIN_CFG_SEARCH = "/path/to/Phoenix/scripts/Tile/256"
TRAIN_CFG_PATH = None

PATCH = 256

# Process only the first N quads (sorted by name) for a test run.
# Set to None for the full set. Ignored when running as a SLURM array task.
LIMIT = None

REQUIRED_CFG_KEYS = ("naip_band_count", "naip_scale", "chm_clip_m")
# Keys copied verbatim from the training config into the prediction config.
COPY_KEYS = ("num_classes", "class_names", "codes", "naip_band_count",
             "base_in_channels", "naip_scale", "ndvi_auto_rescale",
             "chm_clip_m")


# =============================================================================
# HELPERS
# =============================================================================
def resolve_train_cfg():
    """Find the training tile config.json (newest first)."""
    if TRAIN_CFG_PATH:
        if not os.path.exists(TRAIN_CFG_PATH):
            raise SystemExit(f"[FATAL] TRAIN_CFG_PATH missing: {TRAIN_CFG_PATH}")
        return TRAIN_CFG_PATH
    candidates = sorted(
        glob.glob(os.path.join(TRAIN_CFG_SEARCH, "*", "config.json")),
        reverse=True)  # tiles_<stamp>_n<count> sorts by stamp, newest first
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            continue
        if all(k in cfg for k in REQUIRED_CFG_KEYS):
            return path
    raise SystemExit(
        f"[FATAL] no usable training config.json under {TRAIN_CFG_SEARCH}\n"
        f"        needs keys {REQUIRED_CFG_KEYS}\n"
        f"        set TRAIN_CFG_PATH at the top of this file to override.")


def read_on_grid(path, ref_crs, transform, width, height):
    """Read a single-band raster onto the given grid. Plain read if it is
    already on that exact grid, WarpedVRT (bilinear) otherwise.
    Nodata -> 0.0 (nodata NDVI ~ bare/none, nodata CHM ~ 0 m height)."""
    with rasterio.open(path) as src:
        same = (src.transform == transform and src.width == width
                and src.height == height)
        if same:
            arr = src.read(1).astype(np.float32)
            nd = src.nodata
        else:
            print(f"[WARN] {os.path.basename(path)} not on the NAIP grid, "
                  f"resampling via WarpedVRT")
            with WarpedVRT(src, src_crs=(src.crs or ref_crs), crs=ref_crs,
                           transform=transform, width=width, height=height,
                           resampling=Resampling.bilinear) as vrt:
                arr = vrt.read(1).astype(np.float32)
                nd = vrt.nodata
    if nd is not None:
        arr = np.where(arr == nd, 0.0, arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # Guard against un-flagged -9999 style nodata
    arr[arr < -1000] = 0.0
    return arr


def quad_stems():
    """NAIP stems (sorted) with their NDVI/CHM partners resolved.
    Returns (usable, missing) lists."""
    usable, missing = [], []
    for p in sorted(glob.glob(os.path.join(NAIP_DIR, "m_*.tif"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        ndvi = os.path.join(NDVI_DIR, f"{stem}_ndvi.tif")
        chm = os.path.join(CHM_DIR, f"{stem}_chm.tif")
        if os.path.exists(ndvi) and os.path.exists(chm):
            usable.append((stem, p, ndvi, chm))
        else:
            missing.append((stem,
                            "ndvi" if not os.path.exists(ndvi) else "",
                            "chm" if not os.path.exists(chm) else ""))
    return usable, missing


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    train_cfg_path = resolve_train_cfg()
    with open(train_cfg_path, encoding="utf-8") as f:
        train_cfg = json.load(f)
    nb = int(train_cfg["naip_band_count"])
    print(f"[INFO] training config: {train_cfg_path}")
    print(f"[INFO] naip_band_count={nb}  naip_scale={train_cfg['naip_scale']}  "
          f"chm_clip_m={train_cfg['chm_clip_m']}")

    usable, missing = quad_stems()
    for stem, nd_miss, ch_miss in missing:
        print(f"[SKIP] {stem}: missing "
              f"{' '.join(x for x in (nd_miss, ch_miss) if x)}")
    print(f"[INFO] quads usable={len(usable)}  missing-partner={len(missing)}")
    if not usable:
        raise SystemExit("[FATAL] no usable quads found.")

    # SLURM array mode: each task handles exactly one quad, picked by index
    # into the SORTED usable list (stable across tasks because quad_stems()
    # sorts by filename).
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        idx = int(task_id)
        if idx >= len(usable):
            print(f"[INFO] array task {idx}: only {len(usable)} usable quads, "
                  f"nothing to do")
            return
        usable = [usable[idx]]
        print(f"[INFO] array task {idx} -> {usable[0][0]}")
    elif LIMIT is not None:
        usable = usable[:LIMIT]
        print(f"[INFO] LIMIT={LIMIT} -> processing {len(usable)} quad(s) only")

    # Prediction tile config: normalization keys copied from training so
    # predict-time preprocessing matches training exactly.
    out_cfg = {k: train_cfg[k] for k in COPY_KEYS if k in train_cfg}
    out_cfg.update({
        "purpose": "prediction tiles (no labels)",
        "tile_size": [PATCH, PATCH],
        "patch_size": PATCH,
        "train_config_source": train_cfg_path,
        "naip_dir": NAIP_DIR,
        "ndvi_dir": NDVI_DIR,
        "chm_dir": CHM_DIR,
        "prepared_at": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    })
    # Atomic write: parallel array tasks all write the same content, so the
    # temp-then-replace just prevents a reader ever seeing a half-written file.
    cfg_out = os.path.join(OUT_ROOT, "config.json")
    cfg_tmp = f"{cfg_out}.tmp{os.getpid()}"
    with open(cfg_tmp, "w") as f:
        json.dump(out_cfg, f, indent=2)
    os.replace(cfg_tmp, cfg_out)

    done, skipped_existing = 0, 0
    for qi, (stem, naip_path, ndvi_path, chm_path) in enumerate(usable, 1):
        quad_dir = os.path.join(OUT_ROOT, stem)
        if os.path.exists(os.path.join(quad_dir, "meta.json")):
            skipped_existing += 1
            print(f"[SKIP] {stem}: already done (meta.json exists)")
            continue

        tmp_dir = quad_dir + "_preparing"
        os.makedirs(tmp_dir, exist_ok=True)

        with rasterio.open(naip_path) as naip:
            if naip.count != nb:
                raise SystemExit(f"[FATAL] {stem}: NAIP has {naip.count} bands "
                                 f"but training config says {nb}")
            H, W = naip.height, naip.width
            transform, crs = naip.transform, naip.crs
            na = naip.read().astype(np.float32)          # (nb, H, W)

        nd = read_on_grid(ndvi_path, crs, transform, W, H)
        ch = read_on_grid(chm_path, crs, transform, W, H)

        # (H, W, nb + 2) raw stack, same band order as training tiles
        img = np.concatenate([np.transpose(na, (1, 2, 0)),
                              nd[..., None], ch[..., None]], axis=2)
        del na, nd, ch

        n_rows = int(np.ceil(H / PATCH))
        n_cols = int(np.ceil(W / PATCH))
        n_tiles = 0
        for r0 in range(0, H, PATCH):
            for c0 in range(0, W, PATCH):
                h0 = min(PATCH, H - r0)
                w0 = min(PATCH, W - c0)
                sub = img[r0:r0 + h0, c0:c0 + w0, :]
                if h0 < PATCH or w0 < PATCH:
                    # Edge tile: zero pad up to 256, record the valid size.
                    full = np.zeros((PATCH, PATCH, img.shape[2]), np.float32)
                    full[:h0, :w0, :] = sub
                    sub = full
                ptrans = rasterio.windows.transform(
                    Window(c0, r0, PATCH, PATCH), transform)
                out = os.path.join(tmp_dir, f"tile_r{r0:05d}_c{c0:05d}.npz")
                np.savez_compressed(
                    out,
                    image=sub.astype(np.float32),
                    transform=np.array([ptrans.a, ptrans.b, ptrans.c,
                                        ptrans.d, ptrans.e, ptrans.f],
                                       dtype=np.float64),
                    source=f"{stem}_r{r0}_c{c0}",
                    valid_h=h0,
                    valid_w=w0,
                )
                n_tiles += 1
        del img

        meta = {
            "stem": stem,
            "naip_path": naip_path,
            "ndvi_path": ndvi_path,
            "chm_path": chm_path,
            "width": W,
            "height": H,
            "crs": str(crs),
            "transform": [transform.a, transform.b, transform.c,
                          transform.d, transform.e, transform.f],
            "patch_size": PATCH,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_tiles": n_tiles,
        }
        with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Atomic finish: rename only after every tile + meta.json is written,
        # so a killed job never leaves a quad that LOOKS finished.
        os.rename(tmp_dir, quad_dir)
        done += 1
        print(f"[DONE] ({qi}/{len(usable)}) {stem}: {W}x{H} -> {n_tiles} tiles "
              f"({n_rows}x{n_cols} grid)")

    print("\n" + "=" * 60)
    print(f"[DONE] quads written={done}  already-done={skipped_existing}  "
          f"missing-partner={len(missing)}")
    print(f"[DONE] tiles -> {OUT_ROOT}")
    if LIMIT is not None:
        print(f"[NEXT] check the output, then set LIMIT = None and resubmit "
              f"(finished quads are skipped automatically)")


if __name__ == "__main__":
    main()
