#!/usr/bin/env python
"""
LAZ_to_CHM.py  --  Sol HPC version, per-NAIP-tile output
========================================================
Full LAZ -> CHM pipeline. Output is one CHM .tif per NAIP tile, aligned
exactly to NAIP's CRS / grid / extent, so NAIP + NDVI + CHM form 1:1:1
paired tiles that load directly into TorchGeo.

Pipeline:
  1-2. LAZ -> DTM (Classification[2:2], min) + DSM (all returns, max), 0.5 m
       Per-tile TIN nodata fill, float32 enforced.
  3-4. Build DTM/DSM VRT mosaics with -tap pixel grid alignment.
  5.   Per NAIP tile: reproject DSM/DTM VRTs onto NAIP grid (0.3 m),
       compute CHM = max(DSM - DTM, 0), save as CHM_<naip_stem>.tif.

Resume-friendly at every stage: existing outputs are skipped.
"""

import os
import sys
import glob
import json
import subprocess
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# ---------------------------------------------------------------------------#
# CONFIG                                                                     #
# ---------------------------------------------------------------------------#
CONFIG = {
    "laz_folder":    "/path/to/Phoenix/LiDAR/raw_data",
    "naip_folder":   "/path/to/Phoenix/NAIP_Raw/NAIP_raw",  # NAIP GeoTIFFs (56 tiles)
    "output_folder": "/path/to/Phoenix/LiDAR",
    "native_res":    0.5,         # PDAL output (matches LAZ point density)
    "target_res":    0.3,         # final CHM resolution (matches NAIP)
    "epsg":          6341,        # UTM 12N ellipsoidal (USGS LAZ default)
    "nodata":        -9999.0,
    "max_workers":   16,          # Steps 1-2: 16GB/worker (256G ÷ 16) - TIN-safe with subsample
    "naip_workers":  8,           # Step 5: ~15 GB/worker for full-NAIP-tile float32 ops
}
# ---------------------------------------------------------------------------#


# ===========================================================================
# Steps 1-2: LAZ -> DTM/DSM tiles
# ===========================================================================

def tin_fill_nodata(tif_path, nodata, max_tin_points=500_000):
    """Fill nodata via TIN interpolation (scipy LinearNDInterpolator).

    Memory safety: if the number of valid points exceeds max_tin_points, we
    randomly subsample for triangulation. This keeps memory bounded without
    meaningfully degrading interpolation quality (TIN on 500k well-distributed
    points is more than enough for 0.5 m DTM/DSM accuracy).
    """
    from scipy.interpolate import LinearNDInterpolator

    with rasterio.open(tif_path) as src:
        data    = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    valid_mask  = data != nodata
    nodata_mask = ~valid_mask

    if not nodata_mask.any():
        if profile.get("dtype") != "float32":
            profile.update({"dtype": "float32"})
            with rasterio.open(tif_path, "w", **profile) as dst:
                dst.write(data, 1)
        return

    # Skip fill on essentially-empty tiles - nothing meaningful to draw from.
    if valid_mask.sum() < 100:
        profile.update({"dtype": "float32"})
        with rasterio.open(tif_path, "w", **profile) as dst:
            dst.write(data, 1)
        return

    vr, vc = np.where(valid_mask)
    vv     = data[valid_mask]

    # Subsample valid points if too many (memory safety).
    n_valid = len(vv)
    if n_valid > max_tin_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_valid, size=max_tin_points, replace=False)
        vr, vc, vv = vr[idx], vc[idx], vv[idx]

    nr, nc = np.where(nodata_mask)

    points = np.column_stack([vc.astype(np.float64), vr.astype(np.float64)])
    interp = LinearNDInterpolator(points, vv.astype(np.float64))
    filled = interp(nc.astype(np.float64), nr.astype(np.float64))

    # Edges outside the triangulation convex hull come back as NaN - leave as nodata.
    filled[np.isnan(filled)] = nodata
    data[nr, nc] = filled.astype(np.float32)

    profile.update({"dtype": "float32"})
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(data, 1)


def laz_to_raster(laz_path, out_tif, resolution, surface_type, epsg, nodata):
    """One LAZ -> single-band GeoTIFF via PDAL."""
    if os.path.exists(out_tif) and os.path.getsize(out_tif) > 0:
        return out_tif

    stages = [{"type":         "readers.las",
               "filename":     laz_path,
               "override_srs": f"EPSG:{epsg}"}]   # set SRS at reader for full-pipeline consistency
    if surface_type == "dtm":
        stages.append({"type": "filters.range", "limits": "Classification[2:2]"})
        output_type = "min"
    elif surface_type == "dsm":
        # Exclude noise classes (7=Low Noise, 18=High Noise) before taking max.
        # Critical for Geiger-mode LiDAR - random single-photon false detections
        # can sit hundreds of meters above ground and contaminate the DSM max.
        stages.append({"type":   "filters.range",
                       "limits": "Classification![7:7], Classification![18:18]"})
        output_type = "max"
    else:
        raise ValueError(surface_type)

    stages.append({
        "type":         "writers.gdal",
        "filename":     out_tif,
        "gdaldriver":   "GTiff",
        "output_type":  output_type,
        "resolution":   resolution,
        "nodata":       nodata,
        "gdalopts":     "COMPRESS=DEFLATE,TILED=YES",
    })

    try:
        result = subprocess.run(
            ["pdal", "pipeline", "--stdin"],   # PDAL's real stdin flag
            input=json.dumps({"pipeline": stages}),
            capture_output=True, text=True, timeout=1800,   # 30 min - safe for dense Geiger tiles
        )
        if result.returncode != 0:
            # Capture PDAL stderr so we know WHY it failed.
            err = (result.stderr or "").strip()[:400]
            print(f"[PDAL FAIL] {os.path.basename(laz_path)} ({surface_type}): {err}",
                  flush=True)
            if os.path.exists(out_tif):
                os.remove(out_tif)
            return None
        return out_tif
    except Exception as e:
        print(f"[PDAL EXC ] {os.path.basename(laz_path)} ({surface_type}): {e}",
              flush=True)
        if os.path.exists(out_tif):
            os.remove(out_tif)
        return None


def process_single_tile(args):
    """LAZ -> DTM + DSM, then TIN-fill both. TIN failures are non-fatal."""
    laz_path, dtm_dir, dsm_dir, resolution, epsg, nodata = args
    base = Path(laz_path).stem
    dtm_tif = os.path.join(dtm_dir, f"{base}_dtm.tif")
    dsm_tif = os.path.join(dsm_dir, f"{base}_dsm.tif")

    dtm_r = laz_to_raster(laz_path, dtm_tif, resolution, "dtm", epsg, nodata)
    dsm_r = laz_to_raster(laz_path, dsm_tif, resolution, "dsm", epsg, nodata)

    # TIN fill: if scipy blows up on a sparse tile, keep PDAL output as-is
    # rather than killing the worker (which would break the entire pool).
    if dtm_r:
        try:
            tin_fill_nodata(dtm_tif, nodata)
        except Exception as e:
            print(f"[TIN SKIP] {base} (dtm): {e}", flush=True)
    if dsm_r:
        try:
            tin_fill_nodata(dsm_tif, nodata)
        except Exception as e:
            print(f"[TIN SKIP] {base} (dsm): {e}", flush=True)

    return (dtm_r, dsm_r)


def run_steps_1_2(cfg):
    print("\n" + "=" * 70, flush=True)
    print("[Steps 1-2] LAZ -> DTM/DSM tiles", flush=True)
    print("=" * 70, flush=True)

    dtm_dir = os.path.join(cfg["output_folder"], "dtm_tiles")
    dsm_dir = os.path.join(cfg["output_folder"], "dsm_tiles")
    os.makedirs(dtm_dir, exist_ok=True)
    os.makedirs(dsm_dir, exist_ok=True)

    laz_files = sorted(glob.glob(os.path.join(cfg["laz_folder"], "*.laz")))
    if not laz_files:
        print(f"  ERROR: no .laz in {cfg['laz_folder']}", flush=True)
        return False

    todo, done = [], 0
    for f in laz_files:
        base = Path(f).stem
        dt = os.path.join(dtm_dir, f"{base}_dtm.tif")
        ds = os.path.join(dsm_dir, f"{base}_dsm.tif")
        if (os.path.exists(dt) and os.path.getsize(dt) > 0 and
                os.path.exists(ds) and os.path.getsize(ds) > 0):
            done += 1
        else:
            todo.append(f)

    print(f"  Total LAZ    : {len(laz_files)}", flush=True)
    print(f"  Already done : {done}", flush=True)
    print(f"  To process   : {len(todo)}", flush=True)
    if not todo:
        return True

    task_args = [
        (f, dtm_dir, dsm_dir, cfg["native_res"], cfg["epsg"], cfg["nodata"])
        for f in todo
    ]
    success = failed = 0
    print(f"  Workers      : {cfg['max_workers']}", flush=True)

    with ProcessPoolExecutor(max_workers=cfg["max_workers"]) as ex:
        futures = {ex.submit(process_single_tile, a): a[0] for a in task_args}
        with tqdm(total=len(futures), desc="  Tiles", smoothing=0.05) as pbar:
            for fut in as_completed(futures):
                try:
                    dt, ds = fut.result()
                    if dt and ds:
                        success += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
                pbar.update(1)

    print(f"  This run     : {success} ok, {failed} failed", flush=True)
    return (done + success) > 0


# ===========================================================================
# Steps 3-4: VRT mosaics
# ===========================================================================

def is_tif_valid(tif_path):
    try:
        with rasterio.open(tif_path) as src:
            src.read(1, window=Window(0, 0, min(1, src.width), min(1, src.height)))
        return True
    except Exception:
        return False


def build_vrt(tile_folder, vrt_path, resolution):
    """gdalbuildvrt with -tap for pixel grid alignment."""
    all_tifs = sorted(glob.glob(os.path.join(tile_folder, "*.tif")))
    if not all_tifs:
        return None

    valid = []
    for t in tqdm(all_tifs, desc="  Validating"):
        if is_tif_valid(t):
            valid.append(t)
    print(f"  Valid: {len(valid)}/{len(all_tifs)}", flush=True)
    if not valid:
        return None

    flist = vrt_path + ".filelist.txt"
    with open(flist, "w") as f:
        f.write("\n".join(valid) + "\n")

    r = subprocess.run(
        ["gdalbuildvrt",
         "-tr", str(resolution), str(resolution),
         "-tap",
         "-input_file_list", flist,
         vrt_path],
        capture_output=True, text=True,
    )
    os.remove(flist)
    if r.returncode != 0:
        print(f"  ERROR gdalbuildvrt: {r.stderr[:300]}", flush=True)
        return None
    print(f"  VRT: {vrt_path}", flush=True)
    return vrt_path


# ===========================================================================
# Step 5: per-NAIP-tile CHM, aligned to NAIP grid
# ===========================================================================

def chm_for_naip_tile(args):
    """Reproject DSM/DTM VRTs onto one NAIP tile's grid, compute CHM."""
    naip_path, dsm_vrt, dtm_vrt, out_dir, nodata = args
    stem = Path(naip_path).stem
    out_path = os.path.join(out_dir, f"{stem}_chm.tif")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return ("skip", stem)

    try:
        # Take target grid from NAIP tile.
        with rasterio.open(naip_path) as naip:
            dst_crs       = naip.crs
            dst_transform = naip.transform
            dst_h         = naip.height
            dst_w         = naip.width

        dsm = np.full((dst_h, dst_w), nodata, dtype=np.float32)
        dtm = np.full((dst_h, dst_w), nodata, dtype=np.float32)

        # Reproject DSM mosaic onto NAIP grid.
        with rasterio.open(dsm_vrt) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dsm,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=nodata,
                dst_nodata=nodata,
            )

        # Reproject DTM mosaic onto NAIP grid.
        with rasterio.open(dtm_vrt) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dtm,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=nodata,
                dst_nodata=nodata,
            )

        # CHM = max(0, DSM - DTM), nodata where either side is missing.
        valid = (dsm != nodata) & (dtm != nodata)
        chm = np.full_like(dsm, nodata, dtype=np.float32)
        chm[valid] = np.maximum(dsm[valid] - dtm[valid], 0.0)

        profile = {
            "driver":     "GTiff",
            "dtype":      "float32",
            "count":      1,
            "height":     dst_h,
            "width":      dst_w,
            "crs":        dst_crs,
            "transform":  dst_transform,
            "nodata":     nodata,
            "compress":   "deflate",
            "predictor":  3,           # float predictor
            "tiled":      True,
            "blockxsize": 512,
            "blockysize": 512,
            "BIGTIFF":    "IF_SAFER",
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(chm, 1)

        return ("ok", stem)
    except Exception as e:
        if os.path.exists(out_path):
            os.remove(out_path)
        return ("fail", f"{stem}: {e}")


def run_step_5_per_naip(cfg, dsm_vrt, dtm_vrt):
    print("\n" + "=" * 70, flush=True)
    print("[Step 5] Per-NAIP-tile CHM (DSM-DTM resampled to NAIP grid)", flush=True)
    print("=" * 70, flush=True)

    out_dir = os.path.join(cfg["output_folder"], "per_naip")
    os.makedirs(out_dir, exist_ok=True)

    naip_files = sorted(
        glob.glob(os.path.join(cfg["naip_folder"], "**", "*.tif"), recursive=True) +
        glob.glob(os.path.join(cfg["naip_folder"], "**", "*.TIF"), recursive=True)
    )
    # Dedup case-insensitive
    seen, unique = set(), []
    for f in naip_files:
        k = os.path.normpath(f).lower()
        if k not in seen:
            seen.add(k)
            unique.append(f)
    naip_files = unique

    if not naip_files:
        print(f"  ERROR: no NAIP tiles in {cfg['naip_folder']}", flush=True)
        return False

    print(f"  NAIP tiles : {len(naip_files)}", flush=True)
    print(f"  Output dir : {out_dir}", flush=True)
    print(f"  Workers    : {cfg['naip_workers']}", flush=True)

    task_args = [
        (f, dsm_vrt, dtm_vrt, out_dir, cfg["nodata"])
        for f in naip_files
    ]

    ok = skip = fail = 0
    failures = []

    with ProcessPoolExecutor(max_workers=cfg["naip_workers"]) as ex:
        futures = {ex.submit(chm_for_naip_tile, a): a[0] for a in task_args}
        with tqdm(total=len(futures), desc="  CHM tiles", smoothing=0.1) as pbar:
            for fut in as_completed(futures):
                status, info = fut.result()
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    fail += 1
                    failures.append(info)
                pbar.update(1)

    print(f"  Done: {ok} new, {skip} skipped, {fail} failed", flush=True)
    if failures:
        for m in failures[:10]:
            print(f"    [FAIL] {m}", flush=True)
    return True


# ===========================================================================
# main
# ===========================================================================

def main():
    print("=" * 70, flush=True)
    print("LAZ -> CHM pipeline (Sol, per-NAIP-tile output)", flush=True)
    print("=" * 70, flush=True)

    cfg = CONFIG
    final_dir = os.path.join(cfg["output_folder"], "final")
    os.makedirs(final_dir, exist_ok=True)

    # Steps 1-2: LAZ -> DTM/DSM tiles
    if not run_steps_1_2(cfg):
        print("ABORT: no LAZ tiles processed", flush=True)
        sys.exit(1)

    dtm_dir = os.path.join(cfg["output_folder"], "dtm_tiles")
    dsm_dir = os.path.join(cfg["output_folder"], "dsm_tiles")

    # Step 3: DTM VRT
    print("\n[Step 3] DTM VRT", flush=True)
    dtm_vrt = os.path.join(final_dir, "DTM_phoenix_0.5m.vrt")
    if not build_vrt(dtm_dir, dtm_vrt, cfg["native_res"]):
        print("ABORT: DTM VRT failed", flush=True)
        sys.exit(1)

    # Step 4: DSM VRT
    print("\n[Step 4] DSM VRT", flush=True)
    dsm_vrt = os.path.join(final_dir, "DSM_phoenix_0.5m.vrt")
    if not build_vrt(dsm_dir, dsm_vrt, cfg["native_res"]):
        print("ABORT: DSM VRT failed", flush=True)
        sys.exit(1)

    # Step 5: per-NAIP-tile CHM
    run_step_5_per_naip(cfg, dsm_vrt, dtm_vrt)

    print("\n" + "=" * 70, flush=True)
    print("ALL DONE", flush=True)
    print(f"  Per-NAIP CHM dir : {os.path.join(cfg['output_folder'], 'per_naip')}",
          flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
