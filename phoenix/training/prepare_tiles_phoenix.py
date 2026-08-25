"""
prepare_tiles_phoenix.py  (v3 -- hand-modified labels, multi patch size)
------------------------------------------------------------------------
Convert the HAND-MODIFIED Phoenix label groups into training .npz tiles for
train_phoenix.py.

Changes vs v2:
  * Writes TWO tile sets in one run, one per patch size in PATCH_SIZES:
        256 : the full source raster tile, saved as before
        128 : each 256 tile cut into a 2x2 grid of non-overlapping 128 crops
              (4x the tile count)
    Output roots (one timestamped folder per size, so runs can coexist):
        <TILE_OUT_BASE>/256/tiles_<stamp>_n<NNNN>/
        <TILE_OUT_BASE>/128/tiles_<stamp>_n<NNNN>/
    -> pass the wanted folder to train_phoenix.py via --tile-root
  * The train/val split is decided ONCE per source tile and inherited by all
    of its 128 crops, so crops of the same tile never leak across splits.
  * Each 128 crop is re-checked against MIN_LABELED_FRAC (a quadrant can be
    empty even when the full tile is not) and gets its own geotransform.

Changes vs v1 (kept from v2):
  * Labels come from LABEL_ROOT (ArcGIS-edited shapefiles):
        <LABEL_ROOT>/label_0001-0050/label_0001-0050.shp  (+ .dbf .prj .shx ...)
    Only groups that exist under LABEL_ROOT are processed. New label_XXXX-YYYY
    folders added later are picked up automatically on the next run.
  * NAIP / NDVI / CHM rasters still come from the ORIGINAL bundles:
        <BUNDLE_ROOT>/group_XXXX-YYYY/
  * Tree-over-Building rule: where a Tree polygon overlaps a Building polygon
    (canopy hanging over a roof), the pixel is labeled Tree, regardless of
    polygon draw order. All other overlaps keep the original behavior
    (later polygon in the shapefile wins).

Class codes (Dataset-A philosophy: shadows were labeled as the underlying
surface during annotation, so there is NO shadow class):
  1 Tree  2 Shrub  3 Grass  4 Soil  5 Building  6 Asphalt  7 Water
Internally stored as 0..6; unlabeled pixels = 255 (IGNORE_INDEX).

Run (CPU only):
  python prepare_tiles_phoenix.py
"""

import glob
import json
import os
import re
from datetime import datetime

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
import geopandas as gpd
from shapely.geometry import box
from shapely.strtree import STRtree

# =============================================================================
# CONFIG   ---  >>> EDIT THESE PATHS FOR YOUR ENVIRONMENT (see README) <<<
# =============================================================================
BUNDLE_ROOT = "/path/to/Phoenix/labeling_new/label_bundles"        # naip/ndvi/chm ONLY
LABEL_ROOT = "/path/to/Phoenix/labeling_hand/Label_modified"       # hand-edited labels
BOUNDARY_SHP = "/path/to/Phoenix/boundary/Council_District.shp"
TILE_OUT_BASE = "/path/to/Phoenix/scripts/Tile"                    # <base>/<patch_size>/tiles_<stamp>_n<count>/

# Patch sizes to export in one run. 256 is the native tile size (saved whole);
# smaller sizes are cut as non-overlapping grids from each source tile.
PATCH_SIZES = [256, 128]

NUM_CLASSES = 7
CLASS_NAMES = ["Tree", "Shrub", "Grass", "Soil", "Building", "Asphalt", "Water"]
VALID_CODES = set(range(1, NUM_CLASSES + 1))          # shapefile codes 1..7
IGNORE_INDEX = 255
TREE_CODE = 1                                          # shapefile code
BUILDING_CODE = 5                                      # shapefile code

VAL_RATIO = 0.20
SEED = 42
MIN_LABELED_FRAC = 0.01      # tiles are hand-labeled -> almost always >> this

CHM_CLIP_M = 30.0

# All Phoenix rasters/labels live on a clean EPSG:3857 grid, but the embedded
# GeoTIFF keys sometimes read back as LOCAL_CS and break CRS comparisons.
# Force a clean definition everywhere (same trick as make_veg_seeds.py).
FORCE_CRS = CRS.from_epsg(3857)

# Candidate names for the class attribute in the label shapefile (auto-detect).
CLASS_FIELD_CANDIDATES = ["class", "Class", "CLASS", "ClassValue", "classvalue",
                          "class_id", "gridcode", "Gridcode", "GRIDCODE",
                          "code", "Code", "CODE"]
# columns that are never the class field (lowercase compare)
NON_CLASS_COLS = {"fid", "objectid", "id", "shape_leng", "shape_length",
                  "shape_area", "geometry"}


# =============================================================================
# HELPERS
# =============================================================================
def detect_class_field(gdf, shp_path):
    for f in CLASS_FIELD_CANDIDATES:
        if f in gdf.columns:
            return f
    # Fallback: a single numeric column whose values are all within 1..7
    # (ArcGIS exports sometimes rename the field).
    numeric_hits = []
    for col in gdf.columns:
        if col.lower() in NON_CLASS_COLS or col == gdf.geometry.name:
            continue
        vals = gdf[col].dropna()
        if len(vals) == 0:
            continue
        try:
            v = vals.astype(float)
        except (ValueError, TypeError):
            continue
        if np.all(v == v.astype(int)) and v.min() >= 1 and v.max() <= NUM_CLASSES:
            numeric_hits.append(col)
    if len(numeric_hits) == 1:
        print(f"[WARN] class field auto-detected by value range: "
              f"'{numeric_hits[0]}' in {os.path.basename(shp_path)}")
        return numeric_hits[0]
    raise ValueError(f"No class field found in {shp_path}. "
                     f"Columns: {gdf.columns.tolist()} "
                     f"(expected one of {CLASS_FIELD_CANDIDATES}; "
                     f"range-based candidates: {numeric_hits})")


def read_on_naip_grid(path, transform, width, height):
    """Read a single-band raster onto the given grid. If it is already on that
    exact grid this is a plain read; otherwise WarpedVRT resamples bilinearly.
    Nodata -> 0.0 (nodata NDVI ~ bare/none, nodata CHM ~ 0 m height)."""
    with rasterio.open(path) as src:
        same = (src.transform == transform and src.width == width and src.height == height)
        if same:
            arr = src.read(1).astype(np.float32)
            nd = src.nodata
        else:
            with WarpedVRT(src, src_crs=FORCE_CRS, crs=FORCE_CRS,
                           transform=transform, width=width, height=height,
                           resampling=Resampling.bilinear) as vrt:
                arr = vrt.read(1).astype(np.float32)
                nd = vrt.nodata
    if nd is not None:
        arr = np.where(arr == nd, 0.0, arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # Guard against un-flagged -9999 nodata
    arr[arr < -1000] = 0.0
    return arr


def tile_ids_in_group(gdir):
    """Tile ids from chm_XXXX.tif filenames (chm always exists per tile)."""
    ids = []
    for p in sorted(glob.glob(os.path.join(gdir, "chm_*.tif"))):
        m = re.search(r"chm_(\d+)\.tif$", os.path.basename(p))
        if m:
            ids.append(m.group(1))
    return ids


def find_band_file(gdir, prefix, tid):
    """naip_0001.tif (exact) or any naip*0001*.tif fallback."""
    exact = os.path.join(gdir, f"{prefix}_{tid}.tif")
    if os.path.exists(exact):
        return exact
    c = sorted(glob.glob(os.path.join(gdir, f"{prefix}*{tid}*.tif")))
    return c[0] if c else None


def find_label_shp(ldir):
    """label_0001-0050/label_0001-0050.shp (exact) or the only *.shp inside.
    ArcGIS .lock / .sr.lock files are ignored (they do not match *.shp)."""
    lname = os.path.basename(ldir)
    exact = os.path.join(ldir, f"{lname}.shp")
    if os.path.exists(exact):
        return exact
    cands = sorted(glob.glob(os.path.join(ldir, "*.shp")))
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        print(f"[WARN] {lname}: multiple .shp found, using {os.path.basename(cands[0])} "
              f"of {[os.path.basename(c) for c in cands]}")
        return cands[0]
    return None


def to_metric_crs(gdf, what):
    """Ensure the GeoDataFrame is on EPSG:3857 (ArcGIS may have re-saved the
    labels with an ESRI-flavored Web Mercator definition or another CRS)."""
    if gdf.crs is None:
        return gdf.set_crs(FORCE_CRS)
    try:
        epsg = gdf.crs.to_epsg()
    except Exception:
        epsg = None
    if epsg == 3857:
        return gdf
    print(f"[INFO] {what}: reprojecting labels {gdf.crs} -> EPSG:3857")
    return gdf.to_crs(FORCE_CRS)


# =============================================================================
# MAIN
# =============================================================================
def main():
    rng = np.random.default_rng(SEED)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # One temp output folder per patch size; renamed with the final tile count
    # at the end (atomic, so half-written runs are never mistaken for done).
    out_tmp = {}
    for ps in PATCH_SIZES:
        out_tmp[ps] = os.path.join(TILE_OUT_BASE, str(ps), f"tiles_{stamp}_preparing")
        for split in ("train", "val"):
            os.makedirs(os.path.join(out_tmp[ps], "tiles", split), exist_ok=True)

    # ---- Coordinate normalization bounds: Phoenix boundary in EPSG:3857 ----
    bnd = gpd.read_file(BOUNDARY_SHP)
    bnd = bnd[bnd.geometry.notnull()]
    if bnd.crs is None:
        bnd = bnd.set_crs(FORCE_CRS)
    elif bnd.crs.to_epsg() != 3857:
        bnd = bnd.to_crs(FORCE_CRS)
    xmin, ymin, xmax, ymax = [float(v) for v in bnd.total_bounds]
    print(f"[INFO] Phoenix boundary bounds (3857): "
          f"x [{xmin:.0f}, {xmax:.0f}]  y [{ymin:.0f}, {ymax:.0f}]")

    # ---- discover hand-modified label groups ----
    label_dirs = sorted(d for d in glob.glob(os.path.join(LABEL_ROOT, "label_*"))
                        if os.path.isdir(d))
    if not label_dirs:
        raise SystemExit(f"No label_* folders in {LABEL_ROOT}")
    print(f"[INFO] {len(label_dirs)} hand-modified label folders found "
          f"(only these groups will be tiled)")

    # Per-size bookkeeping (counts, class balance, low-label skips).
    counters = {ps: {"train": 0, "val": 0} for ps in PATCH_SIZES}
    class_px = {ps: np.zeros(NUM_CLASSES, dtype=np.int64) for ps in PATCH_SIZES}
    low_label_skips = {ps: 0 for ps in PATCH_SIZES}
    # Tile-level skips (apply to the source tile before any cropping).
    skipped = {"missing_band": 0, "low_label": 0, "no_polys": 0}
    unknown_codes_seen = {}
    groups_used = []
    tree_over_building_px = 0
    null_code_polys = 0
    naip_scale = None
    naip_bands = None
    src_tile_size = None

    for ldir in label_dirs:
        lname = os.path.basename(ldir)                       # label_0001-0050
        rng_id = lname[len("label_"):] if lname.startswith("label_") else lname
        gdir = os.path.join(BUNDLE_ROOT, f"group_{rng_id}")  # rasters live here
        if not os.path.isdir(gdir):
            print(f"[WARN] {lname}: no matching raster bundle "
                  f"group_{rng_id} in {BUNDLE_ROOT} -- skipping")
            continue

        # ---- hand-edited labels for this 50-tile group ----
        label_shp = find_label_shp(ldir)
        if label_shp is None:
            print(f"[WARN] {lname}: no .shp inside -- skipping group")
            continue
        gdf = gpd.read_file(label_shp)
        gdf = gdf[gdf.geometry.notnull()].copy()
        gdf = to_metric_crs(gdf, lname)
        gdf["geometry"] = gdf.geometry.buffer(0)
        gdf = gdf[gdf.is_valid & ~gdf.geometry.is_empty].copy()
        field = detect_class_field(gdf, label_shp)

        n_null = int(gdf[field].isnull().sum())
        if n_null:
            null_code_polys += n_null
            print(f"[WARN] {lname}: {n_null} polygons with NULL '{field}' dropped")
            gdf = gdf[gdf[field].notnull()].copy()
        gdf["code"] = gdf[field].astype(float).astype(int)

        unknown = sorted(set(gdf["code"].unique()) - VALID_CODES)
        if unknown:
            for u in unknown:
                u = int(u)
                unknown_codes_seen[u] = unknown_codes_seen.get(u, 0) + int((gdf["code"] == u).sum())
            gdf = gdf[gdf["code"].isin(VALID_CODES)].copy()

        if len(gdf) == 0:
            print(f"[WARN] {lname}: 0 valid polygons -- skipping group")
            continue

        geoms = gdf.geometry.values
        codes = gdf["code"].values.astype(np.uint8)
        tree = STRtree(geoms)

        n_group = 0
        for tid in tile_ids_in_group(gdir):
            naip_p = find_band_file(gdir, "naip", tid)
            ndvi_p = find_band_file(gdir, "ndvi", tid)
            chm_p = find_band_file(gdir, "chm", tid)
            if not (naip_p and ndvi_p and chm_p):
                skipped["missing_band"] += 1
                continue

            with rasterio.open(naip_p) as src:
                naip = src.read().astype(np.float32)          # (4,H,W)
                transform = src.transform
                H, W = src.height, src.width
                if naip_scale is None:
                    dt = np.dtype(src.dtypes[0])
                    naip_scale = float(np.iinfo(dt).max) if np.issubdtype(dt, np.integer) else 1.0
                    naip_bands = src.count
                    src_tile_size = (H, W)

            ndvi = read_on_naip_grid(ndvi_p, transform, W, H)
            chm = read_on_naip_grid(chm_p, transform, W, H)

            # ---- rasterize the labels that intersect this tile ----
            tile_poly = box(*rasterio.windows.bounds(
                rasterio.windows.Window(0, 0, W, H), transform))
            hits = tree.query(tile_poly)
            if len(hits) and not isinstance(hits[0], (int, np.integer)):
                # old shapely returns geometries; map back to indices
                gid = {id(g): i for i, g in enumerate(geoms)}
                hits = np.array([gid[id(g)] for g in hits], dtype=int)
            else:
                hits = np.asarray(hits, dtype=int)
            if hits.size == 0:
                skipped["no_polys"] += 1
                continue

            shapes = [(geoms[i], int(codes[i]) - 1) for i in hits]   # internal 0..6
            mask = rasterize(shapes, out_shape=(H, W), transform=transform,
                             fill=IGNORE_INDEX, dtype=np.uint8)

            # ---- Tree-over-Building rule ----
            # Canopy hanging over a roof must be Tree even if the building
            # polygon was drawn later. Other class overlaps keep draw order.
            tree_hits = [i for i in hits if codes[i] == TREE_CODE]
            if tree_hits and np.any(mask == BUILDING_CODE - 1):
                tmask = rasterize([(geoms[i], 1) for i in tree_hits],
                                  out_shape=(H, W), transform=transform,
                                  fill=0, dtype=np.uint8)
                flip = (mask == BUILDING_CODE - 1) & (tmask == 1)
                if flip.any():
                    mask[flip] = TREE_CODE - 1
                    tree_over_building_px += int(flip.sum())

            labeled_frac = float(np.mean(mask != IGNORE_INDEX))
            if labeled_frac < MIN_LABELED_FRAC:
                skipped["low_label"] += 1
                continue

            img = np.concatenate([np.transpose(naip, (1, 2, 0)),
                                  ndvi[..., None], chm[..., None]], axis=2)

            # One split decision per SOURCE tile. Every patch cut from this
            # tile inherits it, so 128 crops of one tile can never end up on
            # both sides of the train/val split (no spatial leakage).
            split = "val" if rng.random() < VAL_RATIO else "train"

            # ---- save patches for every configured size ----
            for ps in PATCH_SIZES:
                if ps > H or ps > W:
                    print(f"[WARN] patch size {ps} > tile {H}x{W}, skipping size")
                    continue
                # Non-overlapping grid; for ps == tile size this is one patch.
                for r0 in range(0, H - ps + 1, ps):
                    for c0 in range(0, W - ps + 1, ps):
                        sub_mask = mask[r0:r0 + ps, c0:c0 + ps]
                        # A crop can be nearly unlabeled even when the full
                        # tile passed the check, so re-check per crop.
                        if float(np.mean(sub_mask != IGNORE_INDEX)) < MIN_LABELED_FRAC:
                            low_label_skips[ps] += 1
                            continue
                        sub_img = img[r0:r0 + ps, c0:c0 + ps, :]
                        ptrans = rasterio.windows.transform(
                            rasterio.windows.Window(c0, r0, ps, ps), transform)
                        for c in range(NUM_CLASSES):
                            class_px[ps][c] += int((sub_mask == c).sum())
                        source = (f"{lname}/{tid}" if (ps == H and ps == W)
                                  else f"{lname}/{tid}_p{ps}_r{r0}_c{c0}")
                        idx = counters[ps]["train"] + counters[ps]["val"]
                        out = os.path.join(out_tmp[ps], "tiles", split,
                                           f"tile_{idx:06d}.npz")
                        np.savez_compressed(
                            out,
                            image=sub_img.astype(np.float32),
                            mask=sub_mask.astype(np.uint8),
                            transform=np.array([ptrans.a, ptrans.b, ptrans.c,
                                                ptrans.d, ptrans.e, ptrans.f],
                                               dtype=np.float64),
                            source=source,
                        )
                        counters[ps][split] += 1
            n_group += 1

        groups_used.append(lname)
        print(f"[INFO] {lname}: {n_group} source tiles used "
              f"(field='{field}', {len(gdf)} polygons, rasters from group_{rng_id})")

    if counters[PATCH_SIZES[0]]["train"] + counters[PATCH_SIZES[0]]["val"] == 0:
        raise SystemExit("[FATAL] 0 tiles produced -- check the warnings above.")

    # ---- per-size config + atomic rename ----
    out_roots = {}
    for ps in PATCH_SIZES:
        total_tiles = counters[ps]["train"] + counters[ps]["val"]
        out_roots[ps] = os.path.join(TILE_OUT_BASE, str(ps),
                                     f"tiles_{stamp}_n{total_tiles:04d}")
        total_px = int(class_px[ps].sum())
        cfg = {
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "codes": {str(i + 1): n for i, n in enumerate(CLASS_NAMES)},
            "ignore_index": IGNORE_INDEX,
            "naip_band_count": int(naip_bands or 4),
            "base_in_channels": int((naip_bands or 4) + 2),
            "naip_scale": naip_scale or 255.0,
            "ndvi_auto_rescale": True,
            "chm_clip_m": CHM_CLIP_M,
            # thresholds shared by the hybrid height-rule and the Plan-B pseudo-labeler
            # (grass < th_grass_m <= shrub <= th_tree_m < tree; veg if NDVI >= ndvi_veg_min)
            "th_grass_m": 0.3,
            "th_tree_m": 1.5,
            "ndvi_veg_min": 0.2,
            "tile_size": [ps, ps],
            "patch_size": ps,
            "source_tile_size": list(src_tile_size) if src_tile_size else None,
            "crs_epsg": 3857,
            # coordinate normalization bounds (Phoenix boundary, EPSG:3857)
            "coord_bounds": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
            "seed": SEED,
            "val_ratio": VAL_RATIO,
            "train_tiles": counters[ps]["train"],
            "val_tiles": counters[ps]["val"],
            "class_pixel_counts": {n: int(class_px[ps][i]) for i, n in enumerate(CLASS_NAMES)},
            "class_pixel_frac": {n: (int(class_px[ps][i]) / total_px if total_px else 0.0)
                                 for i, n in enumerate(CLASS_NAMES)},
            "skipped": {**skipped, "crop_low_label": low_label_skips[ps]},
            "unknown_codes_dropped": unknown_codes_seen,
            "null_code_polygons_dropped": null_code_polys,
            "bundle_root": BUNDLE_ROOT,
            "label_root": LABEL_ROOT,
            "groups_used": groups_used,
            "overlap_rule": "tree_over_building",
            "tree_over_building_px": int(tree_over_building_px),
            "boundary_shp": BOUNDARY_SHP,
            "prepared_at": stamp,
        }
        with open(os.path.join(out_tmp[ps], "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        # atomic rename: <..._preparing> -> tiles_<stamp>_n<count>
        os.rename(out_tmp[ps], out_roots[ps])

    print("\n" + "=" * 60)
    print(f"[DONE] groups used ({len(groups_used)}): {groups_used}")
    print(f"[DONE] skipped source tiles: {skipped}")
    if unknown_codes_seen:
        print(f"[WARN] polygons with unexpected codes dropped: {unknown_codes_seen}")
    ref_px = int(class_px[PATCH_SIZES[0]].sum())
    print(f"[DONE] Building px relabeled as Tree (canopy over roof): "
          f"{tree_over_building_px} ({tree_over_building_px / max(ref_px, 1):.4%} of labeled px)")
    for ps in PATCH_SIZES:
        total_tiles = counters[ps]["train"] + counters[ps]["val"]
        total_px = int(class_px[ps].sum())
        print(f"\n[DONE] === patch size {ps} ===")
        print(f"[DONE] train={counters[ps]['train']}  val={counters[ps]['val']}  "
              f"total={total_tiles}  (crops dropped as low-label: {low_label_skips[ps]})")
        print(f"[DONE] class pixel fractions:")
        for i, n in enumerate(CLASS_NAMES):
            print(f"        {i + 1} {n:<9s} {class_px[ps][i] / max(total_px, 1):8.4f}")
        print(f"[DONE] tiles -> {out_roots[ps]}")
    print(f"\n[NEXT] train with e.g.:")
    print(f"  python train_phoenix.py --bs 16 --lr 1e-4 --seed 1 --coord xy \\")
    print(f"      --tile-root {out_roots[PATCH_SIZES[0]]}")


if __name__ == "__main__":
    main()
