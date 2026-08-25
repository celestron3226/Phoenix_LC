# predict_local.py
# Week12: single-quad local prediction (encoder_compare chesapeake_resnet18 coord-xy model)
# phoenix_common merged in; same style as predict_model_shadow.py but with the
# three things this model additionally needs: coord-xy channels, WarpedVRT
# alignment of NDVI/CHM onto the NAIP grid, and nodata(-9999) handling.
import json
import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
import torch
import segmentation_models_pytorch as smp

# =============================================================================
# PATHS / SETTINGS
# =============================================================================
WEEK12 = r"C:\Users\Insoo\Desktop\2026 Summer\Dr. Tong\Week 12"
NAIP_PATH = os.path.join(WEEK12, "m_3311240_se_12_030_20230917.tif")
NDVI_PATH = os.path.join(WEEK12, "m_3311240_se_12_030_20230917_ndvi.tif")
CHM_PATH = os.path.join(WEEK12, "m_3311240_se_12_030_20230917_chm.tif")
OUT_DIR = os.path.join(WEEK12, "Predict")

MODEL_PATH = (r"C:\Users\Insoo\Desktop\2026 Summer\Dr. Tong\Week 11"
              r"\Encoder_experiment_PHX\Result\encoder_compare"
              r"\chesapeake_resnet18_bs4_coord-xy_seed2\2026-07-23_152437"
              r"\model_best.pth")
CFG_PATH = (r"C:\Users\Insoo\Desktop\2026 Summer\Dr. Tong\Week 11"
            r"\Encoder_experiment_PHX\config.json")

ENCODER_NAME = "resnet18"
COORD_MODE = "xy"          # -> +2 channels; IN_CHANNELS = 4 + 2 + 2 = 8
NUM_CLASSES = 7            # codes 1..7: Tree Shrub Grass Soil Building Asphalt Water
CHIP = 512                 # sliding window (U-Net is fully conv; 512 is fine)
PAD = 32                   # overlap padding cropped away -> no seam artifacts
BATCH = 4
VEG_CODES = (1, 2, 3)      # Tree, Shrub, Grass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# from phoenix_common.py (byte-identical to training preprocessing)
# ============================================================================
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


def coord_channels(transform_abcdef, height, width, bounds):
    """xy mode: per-pixel EPSG:3857 coords normalized to [0,1] over the
    training coord_bounds (must be the SAME bounds as training)."""
    a, b, c, d, e, f = [float(v) for v in transform_abcdef]
    xs = c + (np.arange(width, dtype=np.float64) + 0.5) * a
    ys = f + (np.arange(height, dtype=np.float64) + 0.5) * e
    nx = (xs - bounds["xmin"]) / max(bounds["xmax"] - bounds["xmin"], 1e-9)
    ny = (ys - bounds["ymin"]) / max(bounds["ymax"] - bounds["ymin"], 1e-9)
    gx = np.broadcast_to(nx[None, :], (height, width))
    gy = np.broadcast_to(ny[:, None], (height, width))
    return np.stack([gx, gy], axis=-1).astype(np.float32)


def read_padded(ds, band, col_off, row_off, width, height):
    """Windowed read with manual zero padding (boundless reads on WarpedVRT
    are unreliable, so pad ourselves)."""
    W, H = ds.width, ds.height
    c0, r0 = max(col_off, 0), max(row_off, 0)
    c1, r1 = min(col_off + width, W), min(row_off + height, H)
    out = (np.zeros((ds.count, height, width), np.float32) if band is None
           else np.zeros((height, width), np.float32))
    if c1 <= c0 or r1 <= r0:
        return out
    win = Window(c0, r0, c1 - c0, r1 - r0)
    arr = (ds.read(window=win) if band is None
           else ds.read(band, window=win)).astype(np.float32)
    ro, co = r0 - row_off, c0 - col_off
    if band is None:
        out[:, ro:ro + arr.shape[-2], co:co + arr.shape[-1]] = arr
    else:
        out[ro:ro + arr.shape[-2], co:co + arr.shape[-1]] = arr
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CFG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    bounds = cfg["coord_bounds"]
    nb = int(cfg["naip_band_count"])
    in_channels = nb + 2 + (2 if COORD_MODE == "xy" else 0)

    print(f"[INFO] device={DEVICE}  in_channels={in_channels}  "
          f"classes={NUM_CLASSES}  coord={COORD_MODE}")
    model = smp.Unet(encoder_name=ENCODER_NAME, encoder_weights=None,
                     in_channels=in_channels, classes=NUM_CLASSES).to(DEVICE)
    ck = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(ck, dict) and "model" in ck:      # wrapped checkpoint
        ck = ck["model"]
    model.load_state_dict(ck)
    model.eval()

    stem = os.path.splitext(os.path.basename(NAIP_PATH))[0]
    out_lc = os.path.join(OUT_DIR, f"landcover_{stem}.tif")
    out_veg = os.path.join(OUT_DIR, f"veg_{stem}.tif")

    with rasterio.open(NAIP_PATH) as naip, \
         rasterio.open(NDVI_PATH) as ndvi_src, \
         rasterio.open(CHM_PATH) as chm_src:

        H, W = naip.height, naip.width
        transform = naip.transform
        print(f"[INFO] {stem}: {W}x{H}")

        meta = {"driver": "GTiff", "height": H, "width": W, "count": 1,
                "dtype": "uint8", "crs": naip.crs, "transform": transform,
                "nodata": 0, "compress": "LZW", "tiled": True,
                "BIGTIFF": "IF_SAFER"}

        # NDVI/CHM locked onto the NAIP grid, exactly like the training tiles
        with WarpedVRT(ndvi_src, src_crs=(ndvi_src.crs or naip.crs), crs=naip.crs,
                       transform=transform, width=W, height=H,
                       resampling=Resampling.bilinear) as ndvi, \
             WarpedVRT(chm_src, src_crs=(chm_src.crs or naip.crs), crs=naip.crs,
                       transform=transform, width=W, height=H,
                       resampling=Resampling.bilinear) as chm, \
             rasterio.open(out_lc, "w", **meta) as dst_lc, \
             rasterio.open(out_veg, "w", **meta) as dst_veg:

            ndvi_nd, chm_nd = ndvi_src.nodata, chm_src.nodata
            batch_imgs, batch_infos = [], []
            n_total = int(np.ceil(H / CHIP)) * int(np.ceil(W / CHIP))
            n_win = 0

            def flush():
                if not batch_imgs:
                    return
                x = torch.from_numpy(np.stack(batch_imgs)).to(DEVICE)
                with torch.no_grad(), torch.amp.autocast(
                        "cuda", enabled=(DEVICE.type == "cuda")):
                    pred = torch.argmax(model(x), dim=1).cpu().numpy().astype(np.uint8)
                for i, (l, t, w0, h0) in enumerate(batch_infos):
                    lc = (pred[i, PAD:PAD + h0, PAD:PAD + w0] + 1).astype(np.uint8)
                    dst_lc.write(lc, 1, window=Window(l, t, w0, h0))
                    dst_veg.write(np.isin(lc, VEG_CODES).astype(np.uint8), 1,
                                  window=Window(l, t, w0, h0))
                batch_imgs.clear()
                batch_infos.clear()

            for top in range(0, H, CHIP):
                for left in range(0, W, CHIP):
                    hh, ww = min(CHIP, H - top), min(CHIP, W - left)
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

                    img = np.concatenate([np.transpose(na, (1, 2, 0)),
                                          nd[..., None], ch[..., None]], axis=2)
                    img = normalize_stack(img, cfg)
                    if COORD_MODE == "xy":
                        win_tr = rasterio.windows.transform(
                            Window(c0, r0, wpx, hpx), transform)
                        cc = coord_channels((win_tr.a, win_tr.b, win_tr.c,
                                             win_tr.d, win_tr.e, win_tr.f),
                                            hpx, wpx, bounds)
                        img = np.concatenate([img, cc], axis=2)

                    batch_imgs.append(np.transpose(img, (2, 0, 1)).astype(np.float32))
                    batch_infos.append((left, top, ww, hh))
                    n_win += 1
                    if len(batch_imgs) == BATCH:
                        flush()
                    if n_win % 50 == 0:
                        print(f"[.. ] {n_win}/{n_total} windows")
            flush()

    print(f"[DONE] {n_win} windows")
    print(f"[DONE] landcover -> {out_lc}   (1 Tree 2 Shrub 3 Grass 4 Soil "
          f"5 Building 6 Asphalt 7 Water)")
    print(f"[DONE] veg       -> {out_veg}   (1 = Tree/Shrub/Grass)")


if __name__ == "__main__":
    main()
