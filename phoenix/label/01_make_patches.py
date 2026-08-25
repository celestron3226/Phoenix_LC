#!/usr/bin/env python
"""
01_make_patches.py
==================
PASS 1 of the Phoenix labeling pipeline (CPU, 'geopandas' env).

Sample N random PATCH x PATCH NAIP patches whose CENTER lies inside the Phoenix
boundary, and for EACH patch write, in one shot:

    patch_XXXX/naip.tif    NAIP window (4-band uint8)
    patch_XXXX/chm.tif     CHM clipped to the SAME grid (1-band float32)
    patch_XXXX/ndvi.tif    NDVI clipped to the SAME grid (1-band float32)
    patch_XXXX/label.shp   seed polygons, int field 'class':
                               Building -> 5   (MS footprints)
                               Road     -> 6   (ADOT centerlines, buffered)
                               Wetland  -> 7   (as Water for now)

WHY clip chm/ndvi HERE (not in Pass 2/3): we already hold this patch's grid, so
one clip now means Pass 2 (GPU veg seeds) and Pass 3 (bundling) only ever read
small patch-local tiles -- no giant-source windowed reads, no duplicate NDVI
clip, no repeated CRS workarounds. Clip once, reuse everywhere.

CRS: raw NAIP tiles on Sol can carry a broken LOCAL_CS/EngineeringCRS tag. We
stamp ONE chosen CRS onto naip/chm/ndvi/label so every product co-registers from
birth (all share the same NAIP affine numbers -> stamping is not reprojection).
Set SRC_CRS_OVERRIDE to match what the tiles truly are (see config note).

Tree/Shrub/Grass/Soil are added later (Pass 2 auto-seeds veg; Soil by hand).
All comments in English (script convention).
"""

import os
if "PROJ_DATA" not in os.environ:
    import pyproj
    _pd = pyproj.datadir.get_data_dir()
    os.environ["PROJ_DATA"] = _pd
    os.environ["PROJ_LIB"] = _pd

import csv
import glob
import random
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import rasterio
from rasterio.windows import Window, bounds as window_bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import shapely
from shapely.geometry import box
from shapely.ops import unary_union
from pyproj import CRS

# ---------------------------------------------------------------------------#
# CONFIG  ---  >>> CHECK THESE <<<                                            #
# ---------------------------------------------------------------------------#
NAIP_DIR = "/path/to/Phoenix/NAIP_Raw/NAIP_raw"
BOUNDARY = "/path/to/Phoenix/boundary/Council_District.shp"
BUILDING = "/path/to/Phoenix/Vector/Building/phoenix_buildings_ms_2026-02-03.gpkg"
ROAD     = "/path/to/Phoenix/Vector/Roadnetwork/PHX road network.gpkg"
WETLAND  = "/path/to/Phoenix/Vector/Wetland/wetland.gpkg"

CHM_SRC  = "/path/to/Phoenix/LiDAR/per_naip"   # dir(per-tile *_chm.tif) or single .tif
NDVI_SRC = "/path/to/Phoenix/NDVI"             # dir(per-tile *_ndvi.tif) or single .tif

OUT_DIR  = "/path/to/Phoenix/labeling_new/patches"

PATCH       = 256        # pixels (0.3 m -> 76.8 m). Bump to 512 if that's your target.
STRIDE      = 256        # non-overlapping candidate grid
N_SAMPLES   = 3000
ROAD_BUFFER = 3.0        # meters each side of centerline (~6 m corridor)
NODATA_MAX  = 0.20       # reject patch if > 20% all-band-zero (black) pixels
SEED        = 42

# CRS STAMP. The NAIP tiles read as LOCAL_CS on Sol, so the old pipeline forced
# EPSG:3857 as an identity stamp. Keep that unless your converted tiles now carry
# a correct tag -- then set None to propagate each tile's own CRS faithfully.
#   "EPSG:3857" -> stamp 3857 on every output (matches old make_veg_seeds)
#   None        -> trust and propagate the tile's embedded CRS
SRC_CRS_OVERRIDE = "EPSG:3857"

CLS_BUILDING, CLS_ROAD, CLS_WATER = 5, 6, 7
# ---------------------------------------------------------------------------#

warnings.filterwarnings("ignore")
random.seed(SEED); np.random.seed(SEED)
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
_OVR = CRS.from_user_input(SRC_CRS_OVERRIDE) if SRC_CRS_OVERRIDE else None


def log(*a): print(*a, flush=True)


def layer_crs(path):
    return CRS.from_user_input(pyogrio.read_info(path)["crs"])


def ensure_source(src):
    """Single .tif -> as-is. Directory of tiles -> build+cache one VRT."""
    if os.path.isfile(src):
        return src
    tifs = sorted(glob.glob(os.path.join(src, "*.tif")) + glob.glob(os.path.join(src, "*.tiff")))
    if not tifs:
        raise SystemExit(f"No .tif in {src}")
    vrt = os.path.join(OUT_DIR, "_" + os.path.basename(src.rstrip("/")) + "_src.vrt")
    if not os.path.exists(vrt):
        log(f"Building source VRT for {src} ({len(tifs)} tiles)")
        subprocess.run(["gdalbuildvrt", vrt, *tifs], check=True, stdout=subprocess.DEVNULL)
    return vrt


def read_on_grid(path, transform, w, h, crs):
    """Clip a raster onto the patch grid. src & dst CRS forced to `crs` (identity
    warp) -- all products share the NAIP affine numbers, so this is a windowed
    resample, not a reprojection, and it sidesteps the LOCAL_CS WarpedVRT crash."""
    with rasterio.open(path) as r:
        with WarpedVRT(r, src_crs=crs, crs=crs, transform=transform,
                       width=w, height=h, resampling=Resampling.bilinear) as vrt:
            arr = vrt.read(1).astype("float32")
            nd = vrt.nodata
    if nd is not None:
        arr = np.where(arr == nd, np.nan, arr)
    return np.nan_to_num(arr, nan=-9999.0)


def write_float_tif(arr, transform, crs, path):
    prof = {"driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
            "count": 1, "dtype": "float32", "crs": crs, "transform": transform,
            "nodata": -9999.0, "compress": "lzw"}
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)


def read_clip(path, dst_crs, patch_geom, klass, buffer_m=0.0, pad=2.0):
    """Spatially read features near the patch (in the layer's own CRS), reproject
    to dst_crs, clip to the patch, buffer lines, tag class. Returns gdf or None."""
    read_box = patch_geom.buffer(buffer_m + pad)
    bbox_src = gpd.GeoSeries([read_box], crs=dst_crs).to_crs(layer_crs(path)).total_bounds
    g = gpd.read_file(path, bbox=tuple(bbox_src))
    if g.empty:
        return None
    g = g.to_crs(dst_crs)
    g["geometry"] = g.geometry.buffer(buffer_m) if buffer_m > 0 else g.geometry.buffer(0)
    g = gpd.clip(g, patch_geom)
    g = g[~g.geometry.is_empty & g.geometry.notna()]
    if g.empty:
        return None
    g = g.explode(index_parts=False)
    g = g[g.geometry.geom_type == "Polygon"]
    if g.empty:
        return None
    return gpd.GeoDataFrame({"class": [klass] * len(g)}, geometry=g.geometry.values, crs=dst_crs)


def write_label(path, gdf, dst_crs):
    if gdf is None or len(gdf) == 0:
        gdf = gpd.GeoDataFrame({"class": pd.Series([], dtype="int32")},
                               geometry=gpd.GeoSeries([], crs=dst_crs))
    gdf["class"] = gdf["class"].astype("int32")
    pyogrio.write_dataframe(gdf, str(path), driver="ESRI Shapefile", geometry_type="Polygon")


# ---------------------------------------------------------------------------#
# 0. sources + boundary                                                       #
# ---------------------------------------------------------------------------#
log("Preparing sources ...")
CHM_PATH  = ensure_source(CHM_SRC)
NDVI_PATH = ensure_source(NDVI_SRC)

bnd = gpd.read_file(BOUNDARY)
BND_CRS = bnd.crs
bnd_union = unary_union(bnd.geometry.values)
naip_files = sorted(str(p) for p in Path(NAIP_DIR).glob("*.tif"))
log(f"Boundary CRS {BND_CRS.to_string()} ({len(bnd)} feat) | {len(naip_files)} NAIP tiles")

# ---------------------------------------------------------------------------#
# 1. candidate windows whose CENTER is inside the boundary                    #
# ---------------------------------------------------------------------------#
log("Enumerating candidate windows ...")
bnd_cache, candidates = {}, []
for ti, tif in enumerate(naip_files):
    with rasterio.open(tif) as src:
        tcrs = _OVR if _OVR else CRS.from_wkt(src.crs.to_wkt())
        key = tcrs.to_string()
        if key not in bnd_cache:
            bnd_cache[key] = gpd.GeoSeries([bnd_union], crs=BND_CRS).to_crs(tcrs).iloc[0]
        bnd_t = bnd_cache[key]
        W, H, a = src.width, src.height, src.transform
        cols0 = np.arange(0, W - PATCH + 1, STRIDE)
        rows0 = np.arange(0, H - PATCH + 1, STRIDE)
        if cols0.size == 0 or rows0.size == 0:
            continue
        ccol, crow = np.meshgrid(cols0 + PATCH / 2.0, rows0 + PATCH / 2.0)
        cx = a.c + a.a * ccol + a.b * crow
        cy = a.f + a.d * ccol + a.e * crow
        inside = shapely.contains(bnd_t, shapely.points(cx.ravel(), cy.ravel()))
        gc, gr = np.meshgrid(cols0, rows0)
        candidates.extend((ti, int(r), int(c))
                          for r, c in zip(gr.ravel()[inside], gc.ravel()[inside]))

log(f"Candidates inside boundary: {len(candidates)}")
if len(candidates) < N_SAMPLES:
    log(f"WARNING: only {len(candidates)} candidates (< {N_SAMPLES})")
random.shuffle(candidates)

# ---------------------------------------------------------------------------#
# 2. build patches (image + chm + ndvi + seed label)                          #
# ---------------------------------------------------------------------------#
src_cache = {}
def get_src(ti):
    if ti not in src_cache:
        src_cache[ti] = rasterio.open(naip_files[ti])
    return src_cache[ti]

manifest = []
made = skipped_nodata = failed = 0
for (ti, row, col) in candidates:
    if made >= N_SAMPLES:
        break
    try:
        src = get_src(ti)
        win = Window(col, row, PATCH, PATCH)
        arr = src.read(window=win)
        if np.all(arr == 0, axis=0).mean() > NODATA_MAX:
            skipped_nodata += 1
            continue

        transform = src.window_transform(win)
        out_crs = _OVR if _OVR else CRS.from_wkt(src.crs.to_wkt())   # stamp CRS
        minx, miny, maxx, maxy = window_bounds(win, src.transform)
        pbox = box(minx, miny, maxx, maxy)

        pid = made + 1
        folder = Path(OUT_DIR) / f"patch_{pid:04d}"
        folder.mkdir(parents=True, exist_ok=True)

        # --- image (stamped with out_crs, not the possibly-broken src tag) ---
        profile = {"driver": "GTiff", "dtype": src.dtypes[0], "count": src.count,
                   "height": PATCH, "width": PATCH, "crs": out_crs.to_wkt(),
                   "transform": transform, "compress": "lzw", "predictor": 2, "tiled": False}
        if src.nodata is not None:
            profile["nodata"] = src.nodata
        with rasterio.open(folder / "naip.tif", "w", **profile) as dst:
            dst.write(arr)

        # --- chm + ndvi clipped onto the SAME grid (clip-once) ---
        write_float_tif(read_on_grid(CHM_PATH,  transform, PATCH, PATCH, out_crs),
                        transform, out_crs.to_wkt(), folder / "chm.tif")
        write_float_tif(read_on_grid(NDVI_PATH, transform, PATCH, PATCH, out_crs),
                        transform, out_crs.to_wkt(), folder / "ndvi.tif")

        # --- seed label (building/road/water) ---
        parts = []
        for path, klass, buf in [(BUILDING, CLS_BUILDING, 0.0),
                                 (ROAD, CLS_ROAD, ROAD_BUFFER),
                                 (WETLAND, CLS_WATER, 0.0)]:
            part = read_clip(path, out_crs, pbox, klass, buffer_m=buf)
            if part is not None:
                parts.append(part)
        label = (gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry",
                                  crs=out_crs) if parts else None)
        write_label(folder / "label.shp", label, out_crs)

        manifest.append({"patch_id": f"patch_{pid:04d}", "tile": Path(naip_files[ti]).name,
                         "row": row, "col": col,
                         "label_features": 0 if label is None else len(label)})
        made += 1
        if made % 50 == 0:
            log(f"  built {made}/{N_SAMPLES} (nodata-skipped {skipped_nodata}, failed {failed})")
    except Exception as e:
        failed += 1
        log(f"  ! tile{ti} r{row} c{col} failed: {e}")
        continue

# ---------------------------------------------------------------------------#
# 3. manifest + cleanup                                                       #
# ---------------------------------------------------------------------------#
man_path = Path(OUT_DIR) / "manifest.csv"
with open(man_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["patch_id", "tile", "row", "col", "label_features"])
    w.writeheader(); w.writerows(manifest)
for s in src_cache.values():
    s.close()

log("-" * 60)
log(f"DONE built={made} nodata_skipped={skipped_nodata} failed={failed}")
log(f"Output: {OUT_DIR}  |  stamp CRS: {SRC_CRS_OVERRIDE or 'per-tile'}")
