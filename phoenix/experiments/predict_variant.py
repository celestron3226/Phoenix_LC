"""
predict_variant.py
------------------
City-wide prediction for the hybrid and Plan-B methods, clipped to the Phoenix
Council_District boundary. Same quad loop / boundary handling / atomic writes as
predict_phoenix.py (Plan A); the only differences are the input channels and the
post-processing, both read from the run's run_config.json.

  hybrid : DL predicts 5 classes {Veg, Soil, Building, Asphalt, Water}; the CHM
           height rule re-splits Veg -> final CODES 1..7
           (1 Tree 2 Shrub 3 Grass 4 Soil 5 Building 6 Asphalt 7 Water).
  planb  : DL predicts 4 classes -> CODES 1..4
           (1 Tree 2 Shrub 3 Grass 4 NonVeg). Input is NAIP(+coord) only unless
           the run was trained with --leak-features.

Outputs in <out-dir>:
  landcover_<method>_<stem>.tif  uint8 codes (0 = outside boundary), LZW
  veg_<method>_<stem>.tif        1 = veg (Tree/Shrub/Grass), 0 = other, 255 = outside

Usage:
  python predict_variant.py --run-dir <RUN_DIR>            # all quads
  python predict_variant.py --run-dir <RUN_DIR> --index 7  # one quad (SLURM array)
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box
import torch
import segmentation_models_pytorch as smp

# phoenix_common.py lives in ../training in this repo layout; make it importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from phoenix_common import (coord_channels, normalize_stack, real_ndvi_chm,
                            hybrid_pred5_to_codes)

# =============================================================================
# CONFIG   ---  >>> EDIT THESE PATHS FOR YOUR ENVIRONMENT (see README) <<<
# =============================================================================
NAIP_DIR = "/path/to/Phoenix/NAIP_Raw/NAIP_raw"
NDVI_DIR = "/path/to/Phoenix/NDVI"
CHM_DIR = "/path/to/Phoenix/LiDAR/per_naip"
BOUNDARY_SHP = "/path/to/Phoenix/boundary/Council_District.shp"
OUT_DIR = "/path/to/Phoenix/Result/prediction"

CHIP = 512
PAD = 32
BATCH = 8
FORCE_CRS = CRS.from_epsg(3857)
VEG_CODES = {1, 2, 3}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Phoenix hybrid/Plan-B prediction")
parser.add_argument("--run-dir", type=str, required=True)
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--index", type=int, default=None)
parser.add_argument("--out-dir", type=str, default=OUT_DIR)
args = parser.parse_args()


def pick_best_model(run_dir):
    cands = sorted(glob.glob(os.path.join(run_dir, "best_unet_*.pth")))
    if not cands:
        raise FileNotFoundError(f"No best_unet_*.pth in {run_dir}")
    return max(cands, key=lambda p: p.split("miou")[-1])


def read_padded(ds, band, col_off, row_off, width, height):
    W, H = ds.width, ds.height
    c0, r0 = max(col_off, 0), max(row_off, 0)
    c1, r1 = min(col_off + width, W), min(row_off + height, H)
    out = (np.zeros((ds.count, height, width), np.float32) if band is None
           else np.zeros((height, width), np.float32))
    if c1 <= c0 or r1 <= r0:
        return out
    win = Window(c0, r0, c1 - c0, r1 - r0)
    arr = (ds.read(window=win) if band is None else ds.read(band, window=win)).astype(np.float32)
    ro, co = r0 - row_off, c0 - col_off
    if band is None:
        out[:, ro:ro + arr.shape[-2], co:co + arr.shape[-1]] = arr
    else:
        out[ro:ro + arr.shape[-2], co:co + arr.shape[-1]] = arr
    return out


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, "run_config.json")) as f:
        RC = json.load(f)
    with open(os.path.join(RC["tile_root"], "config.json")) as f:
        CFG = json.load(f)

    method = RC["method"]
    coord_mode = RC["coord"]
    use_ndvi_chm = bool(RC["use_ndvi_chm_input"])
    bounds = CFG["coord_bounds"]
    nb = int(CFG["naip_band_count"])

    model_path = args.model or pick_best_model(args.run_dir)
    print(f"[INFO] method={method} model={model_path}")
    print(f"[INFO] in_channels={RC['in_channels']} dl_classes={RC['dl_classes']} "
          f"coord={coord_mode} ndvi_chm_input={use_ndvi_chm}")

    model = smp.Unet(encoder_name=RC["arch"], encoder_weights=None,
                     in_channels=RC["in_channels"], classes=RC["dl_classes"]).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    bnd = gpd.read_file(BOUNDARY_SHP)
    bnd = bnd[bnd.geometry.notnull()]
    if bnd.crs is None:
        bnd = bnd.set_crs(FORCE_CRS)
    elif bnd.crs.to_epsg() != 3857:
        bnd = bnd.to_crs(FORCE_CRS)
    boundary = bnd.union_all() if hasattr(bnd, "union_all") else bnd.unary_union

    quads = sorted(glob.glob(os.path.join(NAIP_DIR, "m_*.tif")))
    if not quads:
        raise SystemExit(f"No NAIP quads in {NAIP_DIR}")
    idx = args.index
    if idx is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    todo = [quads[idx]] if idx is not None else quads
    if idx is not None:
        print(f"[INFO] array task -> quad {idx}/{len(quads)}: {os.path.basename(quads[idx])}")

    for naip_path in todo:
        predict_quad(naip_path, model, RC, CFG, method, coord_mode,
                     use_ndvi_chm, bounds, nb, boundary)


def predict_quad(naip_path, model, RC, CFG, method, coord_mode,
                 use_ndvi_chm, bounds, nb, boundary):
    stem = os.path.splitext(os.path.basename(naip_path))[0]
    ndvi_path = os.path.join(NDVI_DIR, f"{stem}_ndvi.tif")
    chm_path = os.path.join(CHM_DIR, f"{stem}_chm.tif")
    for p, tag in ((ndvi_path, "NDVI"), (chm_path, "CHM")):
        if not os.path.exists(p):
            print(f"[WARN] {stem}: missing {tag} -- skipping quad")
            return

    out_lc = os.path.join(args.out_dir, f"landcover_{method}_{stem}.tif")
    out_veg = os.path.join(args.out_dir, f"veg_{method}_{stem}.tif")
    if os.path.exists(out_lc):
        print(f"[SKIP] {stem}: output exists")
        return

    with rasterio.open(naip_path) as naip:
        H, W = naip.height, naip.width
        transform = naip.transform
        quad_poly = box(*rasterio.windows.bounds(Window(0, 0, W, H), transform))
        if not quad_poly.intersects(boundary):
            print(f"[SKIP] {stem}: outside Phoenix boundary")
            return
        print(f"[INFO] {stem}: {W}x{H}")
        inside = ~geometry_mask([boundary], out_shape=(H, W), transform=transform, invert=False)

        meta = {"driver": "GTiff", "height": H, "width": W, "count": 1,
                "dtype": "uint8", "crs": FORCE_CRS, "transform": transform,
                "nodata": 0, "compress": "LZW", "tiled": True, "BIGTIFF": "IF_SAFER"}
        meta_veg = dict(meta, nodata=255)

        with rasterio.open(ndvi_path) as ndvi_src, rasterio.open(chm_path) as chm_src, \
             WarpedVRT(ndvi_src, src_crs=FORCE_CRS, crs=FORCE_CRS, transform=transform,
                       width=W, height=H, resampling=Resampling.bilinear) as ndvi, \
             WarpedVRT(chm_src, src_crs=FORCE_CRS, crs=FORCE_CRS, transform=transform,
                       width=W, height=H, resampling=Resampling.bilinear) as chm, \
             rasterio.open(out_lc + ".part.tif", "w", **meta) as dst_lc, \
             rasterio.open(out_veg + ".part.tif", "w", **meta_veg) as dst_veg:

            ndvi_nd, chm_nd = ndvi_src.nodata, chm_src.nodata
            batch_imgs, batch_chm, batch_infos = [], [], []

            def flush():
                if not batch_imgs:
                    return
                x = torch.from_numpy(np.stack(batch_imgs)).to(DEVICE)
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    pred = torch.argmax(model(x), dim=1).cpu().numpy().astype(np.uint8)
                for i, (l, t, w0, h0) in enumerate(batch_infos):
                    core = pred[i, PAD:PAD + h0, PAD:PAD + w0]
                    if method == "hybrid":
                        chm_core = batch_chm[i][PAD:PAD + h0, PAD:PAD + w0]
                        lc = hybrid_pred5_to_codes(core, chm_core, CFG)      # codes 1..7
                    else:
                        lc = (core + 1).astype(np.uint8)                     # codes 1..4
                    m = inside[t:t + h0, l:l + w0]
                    lc = lc.copy(); lc[~m] = 0
                    veg = np.where(m, np.isin(lc, list(VEG_CODES)).astype(np.uint8), 255)
                    dst_lc.write(lc, 1, window=Window(l, t, w0, h0))
                    dst_veg.write(veg.astype(np.uint8), 1, window=Window(l, t, w0, h0))
                batch_imgs.clear(); batch_chm.clear(); batch_infos.clear()

            n_win = 0
            for top in range(0, H, CHIP):
                for left in range(0, W, CHIP):
                    hh, ww = min(CHIP, H - top), min(CHIP, W - left)
                    if not inside[top:top + hh, left:left + ww].any():
                        continue
                    c0, r0 = left - PAD, top - PAD
                    wpx, hpx = CHIP + 2 * PAD, CHIP + 2 * PAD

                    na = read_padded(naip, None, c0, r0, wpx, hpx)
                    nd = read_padded(ndvi, 1, c0, r0, wpx, hpx)
                    ch = read_padded(chm, 1, c0, r0, wpx, hpx)
                    if ndvi_nd is not None:
                        nd = np.where(nd == ndvi_nd, 0.0, nd)
                    if chm_nd is not None:
                        ch = np.where(ch == chm_nd, 0.0, ch)
                    nd[nd < -1000] = 0.0
                    ch[ch < -1000] = 0.0

                    raw = np.concatenate([np.transpose(na, (1, 2, 0)),
                                          nd[..., None], ch[..., None]], axis=2)
                    _, chm_real = real_ndvi_chm(raw, CFG)     # metres, for the rule
                    norm = normalize_stack(raw, CFG)
                    img = norm if use_ndvi_chm else norm[..., :nb]

                    if coord_mode != "none":
                        win_tr = rasterio.windows.transform(Window(c0, r0, wpx, hpx), transform)
                        cc = coord_channels((win_tr.a, win_tr.b, win_tr.c,
                                             win_tr.d, win_tr.e, win_tr.f),
                                            hpx, wpx, bounds, coord_mode)
                        img = np.concatenate([img, cc], axis=2)

                    batch_imgs.append(np.transpose(img, (2, 0, 1)).astype(np.float32))
                    batch_chm.append(chm_real)
                    batch_infos.append((left, top, ww, hh))
                    n_win += 1
                    if len(batch_imgs) == BATCH:
                        flush()
            flush()

    os.replace(out_lc + ".part.tif", out_lc)
    os.replace(out_veg + ".part.tif", out_veg)
    print(f"[DONE] {stem}: {n_win} windows -> {out_lc}")


if __name__ == "__main__":
    main()
