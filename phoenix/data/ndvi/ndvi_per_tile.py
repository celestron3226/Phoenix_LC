"""
NDVI Per-Tile Pipeline (Sol HPC, v5 -- GeoTIFF input)
=====================================================
Computes NDVI for each selected NAIP tile independently.

v5 change vs v4: input is now GeoTIFF (already converted from JP2).
GeoTIFF supports cheap windowed reads, so memory is no longer bounded by
holding entire tile in RAM. Workers can be raised safely.

NAIP band order: 1=Red, 2=Green, 3=Blue, 4=NIR (RGBN).
"""

import os
import glob
import numpy as np
import rasterio
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ============================================================================
# CONFIG
# ============================================================================
RAW_FOLDER  = "/path/to/Phoenix/NAIP_Raw/NAIP_raw"   # GeoTIFFs now
NDVI_FOLDER = "/path/to/Phoenix/NDVI"
MAX_WORKERS = 4        # each worker holds ~10 GB (Red+NIR float32 full-tile + NDVI)
NODATA      = -9999.0
# If None, process every .tif found. Otherwise restrict to these stems.
SELECTED_TILES = None
# ============================================================================


def compute_ndvi_for_tile(tif_path):
    """Compute NDVI for a single NAIP GeoTIFF tile, write to matching .tif."""
    basename = Path(tif_path).stem
    out_path = os.path.join(NDVI_FOLDER, f"{basename}_ndvi.tif")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return ("skip", basename)

    try:
        with rasterio.open(tif_path) as src:
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                count=1,
                dtype="float32",
                nodata=NODATA,
                compress="lzw",
                predictor=3,        # float predictor
                tiled=True,
                blockxsize=1024,
                blockysize=1024,
                bigtiff="YES",
            )

            # Read Red (band 1) and NIR (band 4)
            arr = src.read([1, 4]).astype("float32")
            red = arr[0]
            nir = arr[1]

            # Mask black/no-data pixels (all-zero pixels at tile borders)
            invalid = (red == 0) & (nir == 0)
            denom = nir + red
            ndvi = np.where(
                (denom > 0) & ~invalid,
                (nir - red) / denom,
                NODATA,
            ).astype("float32")

            with rasterio.open(out_path, "w", NUM_THREADS="ALL_CPUS",
                               **profile) as dst:
                dst.write(ndvi, 1)
        return ("ok", basename)
    except Exception as e:
        if os.path.exists(out_path):
            os.remove(out_path)
        return ("fail", f"{basename}: {e}")


def main():
    os.makedirs(NDVI_FOLDER, exist_ok=True)

    all_tifs = sorted(
        glob.glob(os.path.join(RAW_FOLDER, "**", "*.tif"), recursive=True) +
        glob.glob(os.path.join(RAW_FOLDER, "**", "*.TIF"), recursive=True)
    )
    seen, unique = set(), []
    for f in all_tifs:
        k = os.path.normpath(f).lower()
        if k not in seen:
            seen.add(k)
            unique.append(f)
    all_tifs = unique

    if SELECTED_TILES is None:
        tif_files = all_tifs
        print(f"Processing ALL {len(tif_files)} tiles", flush=True)
    else:
        tif_files = [f for f in all_tifs if Path(f).stem in SELECTED_TILES]
        print(f"Found {len(all_tifs)} total, processing {len(tif_files)} selected", flush=True)
        missing = SELECTED_TILES - {Path(f).stem for f in tif_files}
        if missing:
            print(f"\n[WARN] {len(missing)} selected tiles NOT FOUND:")
            for m in sorted(missing):
                print(f"  - {m}")

    if not tif_files:
        print("[ERROR] No tiles to process.", flush=True)
        return

    print(f"Output : {NDVI_FOLDER}", flush=True)
    print(f"Workers: {MAX_WORKERS}\n", flush=True)

    ok = skip = fail = 0
    fail_msgs = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(compute_ndvi_for_tile, f): f for f in tif_files}
        with tqdm(total=len(futures), desc="NDVI tiles") as pbar:
            for fut in as_completed(futures):
                status, info = fut.result()
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    fail += 1
                    fail_msgs.append(info)
                pbar.update(1)

    print(f"\nDone: {ok} new, {skip} skipped, {fail} failed", flush=True)
    if fail_msgs:
        print("\nFailures:")
        for m in fail_msgs[:20]:
            print(f"  {m}")
        if len(fail_msgs) > 20:
            print(f"  ... and {len(fail_msgs) - 20} more")


if __name__ == "__main__":
    main()
