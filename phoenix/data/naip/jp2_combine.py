"""
NAIP JP2 -> tiled GeoTIFF conversion (Sol HPC)
==============================================
One-time conversion so downstream steps (NDVI, U-Net patch sampling, CHM
crop alignment) can do cheap windowed reads instead of full-codestream
JP2 decodes per access.

Quality: no additional loss. We store the same pixel values that the JP2
decoder produces; we just stop re-decoding them on every access.

Set SELECTED_TILES = None to convert all JP2 in RAW_FOLDER.
"""

import os
import glob
import rasterio
from rasterio.shutil import copy as rio_copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ============================================================================
# CONFIG
# ============================================================================
RAW_FOLDER = "/path/to/Phoenix/NAIP_Raw/NAIP compressed_raw"
OUT_FOLDER = "/path/to/Phoenix/NAIP_Raw/NAIP_combined"
# Memory-bound: each worker fully decodes one JP2 tile.
# Same reasoning as NDVI script v4.
MAX_WORKERS = 4

# Same 56 tiles as ndvi_per_tile.py. Keep in sync.
SELECTED_TILES = {
    "m_3311248_nw_12_030_20230917",
    "m_3311248_ne_12_030_20230917",
    "m_3311247_se_12_030_20230917",
    "m_3311247_nw_12_030_20230917",
    "m_3311247_ne_12_030_20230917",
    "m_3311240_sw_12_030_20230917",
    "m_3311240_se_12_030_20230917",
    "m_3311240_nw_12_030_20230917",
    "m_3311240_ne_12_030_20230917",
    "m_3311239_sw_12_030_20230917",
    "m_3311239_se_12_030_20230917",
    "m_3311239_nw_12_030_20230917",
    "m_3311239_ne_12_030_20230917",
    "m_3311238_se_12_030_20230917",
    "m_3311238_nw_12_030_20230917",
    "m_3311238_ne_12_030_20230917",
    "m_3311232_sw_12_030_20230917",
    "m_3311232_se_12_030_20230917",
    "m_3311232_nw_12_030_20230917",
    "m_3311232_ne_12_030_20230917",
    "m_3311231_sw_12_030_20230917",
    "m_3311231_se_12_030_20230917",
    "m_3311231_ne_12_030_20230917",
    "m_3311230_sw_12_030_20230917",
    "m_3311230_se_12_030_20230917",
    "m_3311224_sw_12_030_20230616",
    "m_3311224_se_12_030_20230616",
    "m_3311224_nw_12_030_20230616",
    "m_3311224_ne_12_030_20230616",
    "m_3311223_se_12_030_20230616",
    "m_3311223_nw_12_030_20230616",
    "m_3311223_ne_12_030_20230616",
    "m_3311216_sw_12_030_20230616",
    "m_3311216_se_12_030_20230616",
    "m_3311216_nw_12_030_20230616",
    "m_3311215_sw_12_030_20230616",
    "m_3311215_se_12_030_20230616",
    "m_3311215_nw_12_030_20230616",
    "m_3311215_ne_12_030_20230616",
    "m_3311214_ne_12_030_20230616",
    "m_3311207_sw_12_030_20230616",
    "m_3311207_se_12_030_20230616",
    "m_3311141_sw_12_030_20230917",
    "m_3311141_nw_12_030_20230917",
    "m_3311133_sw_12_030_20230915",
    "m_3311133_nw_12_030_20230915",
    "m_3311125_sw_12_030_20230915",
    "m_3311125_nw_12_030_20230915",
    "m_3311125_ne_12_030_20230915",
    "m_3311117_sw_12_030_20230616",
    "m_3311117_se_12_030_20230615",   # 0615 not 0616
    "m_3311117_nw_12_030_20230616",
    "m_3311117_ne_12_030_20230615",   # 0615 not 0616
    "m_3311109_sw_12_030_20230616",
    "m_3311248_sw_12_030_20230917",
    "m_3311248_se_12_030_20230917",
}

# GeoTIFF creation options. predictor=2 because NAIP is uint8 (integer).
# Tiled blocks at 512 align with typical TorchGeo patch sizes (256/512).
GTIFF_OPTS = dict(
    driver='GTiff',
    COMPRESS='LZW',
    PREDICTOR=2,
    TILED='YES',
    BLOCKXSIZE=512,
    BLOCKYSIZE=512,
    BIGTIFF='IF_SAFER',
    NUM_THREADS='ALL_CPUS',
)
# ============================================================================


def convert_one(jp2_path):
    basename = Path(jp2_path).stem
    out_path = os.path.join(OUT_FOLDER, f"{basename}.tif")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return ("skip", basename)

    try:
        # rasterio.shutil.copy preserves CRS, transform, all bands, all
        # metadata. The JP2 is decoded once during this copy.
        rio_copy(jp2_path, out_path, **GTIFF_OPTS)
        return ("ok", basename)
    except Exception as e:
        if os.path.exists(out_path):
            os.remove(out_path)
        return ("fail", f"{basename}: {e}")


def main():
    os.makedirs(OUT_FOLDER, exist_ok=True)

    all_jp2 = sorted(
        glob.glob(os.path.join(RAW_FOLDER, "**", "*.jp2"), recursive=True) +
        glob.glob(os.path.join(RAW_FOLDER, "**", "*.JP2"), recursive=True)
    )
    seen = set()
    unique = []
    for f in all_jp2:
        key = os.path.normpath(f).lower()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    all_jp2 = unique

    if SELECTED_TILES is None:
        jp2_files = all_jp2
        print(f"Converting ALL {len(jp2_files)} JP2 tiles", flush=True)
    else:
        jp2_files = [f for f in all_jp2 if Path(f).stem in SELECTED_TILES]
        print(f"Found {len(all_jp2)} JP2 files total, converting {len(jp2_files)} selected (target: {len(SELECTED_TILES)})", flush=True)

        found_stems = {Path(f).stem for f in jp2_files}
        missing = SELECTED_TILES - found_stems
        if missing:
            print(f"\n[WARN] {len(missing)} selected tiles NOT FOUND in {RAW_FOLDER}:", flush=True)
            for m in sorted(missing):
                print(f"  - {m}", flush=True)

    if not jp2_files:
        print("[ERROR] No tiles to convert.", flush=True)
        return

    print(f"Output: {OUT_FOLDER}", flush=True)
    print(f"Workers: {MAX_WORKERS}\n", flush=True)

    ok = skip = fail = 0
    fail_msgs = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(convert_one, f): f for f in jp2_files}
        with tqdm(total=len(futures), desc="JP2->GTiff") as pbar:
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

    print(f"\nDone: {ok} new, {skip} skipped (already done), {fail} failed", flush=True)
    if fail_msgs:
        print("\nFailures:")
        for m in fail_msgs[:20]:
            print(f"  {m}")
        if len(fail_msgs) > 20:
            print(f"  ... and {len(fail_msgs) - 20} more")


if __name__ == "__main__":
    main()
