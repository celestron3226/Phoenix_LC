#!/usr/bin/env python
"""
03_build_bundles.py
===================
PASS 3 (CPU, 'geopandas' env). Package every CHUNK tiles into ONE downloadable
.tar.gz for hand-labeling in ArcGIS.

Pass 1 already clipped chm/ndvi to each patch grid and locked the CRS to the
NAIP's own EPSG:3857, so this pass does NO clipping and NO CRS workarounds --
it copies patch-local tiles, MERGES the vectors into one editable layer, builds
relative-path VRTs, and tars.

Reads:
    PATCH_DIR/patch_XXXX/{naip.tif, chm.tif, ndvi.tif, label.shp}   (Pass 1)
    SEEDS_DIR/patch_XXXX/seed_*.shp                                 (Pass 2)

Each bundle group_SSSS-EEEE/:
    naip_group.vrt  chm_group.vrt  ndvi_group.vrt   (relative-path reference layers)
    {naip,chm,ndvi}_XXXX.tif                         (the tiles they reference)
    label_group.shp   <-- THE edit file. Every pre-made feature from all CHUNK
                          tiles, kept as INDIVIDUAL objects (no rectangle merge,
                          no empty-space fill) at real 3857 coords.

label_group.shp schema:  class(int)  src(str)  layer(str)  patch_id(int)  geometry
    class : training target y. 1 tree 2 shrub 3 grass 4 soil
                               5 building 6 asphalt 7 water 8 agri
    src   : 'auto' = pre-made for you (vector building/road/water + chm/SAM veg).
            'hand' = features YOU digitize in ArcGIS -> set this on your new rows.
    layer : origin stem (e.g. 'label', 'seed_tree_dfSAM') -> tells which pre-made
            source a feature came from (e.g. tree Method A vs B).
    patch_id : which tile the feature belongs to (tiles scattered at real coords).

ArcGIS workflow:
    load the 3 *_group.vrt as backdrop, open label_group.shp for editing.
    delete the 'auto' veg you don't trust, draw soil/grass/shrub as new features
    with src='hand'. Coordinates stay identical to the full NAIP, so finished
    labels drop straight onto any full-city NAIP for training/prediction.

All comments in English (script convention).
"""

import os
if "PROJ_DATA" not in os.environ:
    try:
        import pyproj
        _pd = pyproj.datadir.get_data_dir()
        os.environ["PROJ_DATA"] = _pd
        os.environ["PROJ_LIB"] = _pd
    except Exception:
        pass

import glob
import math
import shutil
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import geopandas as gpd
import rasterio

# ---------------------------------------------------------------------------#
# CONFIG  ---  >>> CHECK THESE <<<                                            #
# ---------------------------------------------------------------------------#
PATCH_DIR = "/path/to/Phoenix/labeling_new/patches"        # Pass 1 output
SEEDS_DIR = "/path/to/Phoenix/labeling_new/seeds"          # Pass 2 output
OUT_DIR   = "/path/to/Phoenix/labeling_new/label_bundles"

N_PATCHES   = 3000    # patch_0001 .. patch_N_PATCHES
CHUNK       = 50      # tiles per downloadable bundle -> ceil(N/CHUNK) bundles
MAKE_TAR    = True
KEEP_FOLDER = True
OVERWRITE   = False   # skip a chunk whose .tar.gz already exists (resumable)
# ---------------------------------------------------------------------------#


def find(pid, name):
    p = os.path.join(PATCH_DIR, f"patch_{pid:04d}", name)
    return p if os.path.exists(p) else None


def find_seed_shps(pid):
    return sorted(glob.glob(os.path.join(SEEDS_DIR, f"patch_{pid:04d}", "seed_*.shp")))


def build_vrt(group_dir, out_name, tif_glob):
    """gdalbuildvrt INSIDE group_dir so stored tile paths are relative filenames."""
    tifs = sorted(f.name for f in Path(group_dir).glob(tif_glob))
    if not tifs:
        return 0
    subprocess.run(["gdalbuildvrt", out_name, *tifs],
                   cwd=group_dir, check=True, stdout=subprocess.DEVNULL)
    return len(tifs)


def load_vec(path, bundle_crs, pid):
    """Read one pre-made vector, align to bundle_crs, tag class/src/layer/patch_id.
    Every feature here is pre-made -> src='auto'. Individual objects preserved."""
    try:
        g = gpd.read_file(path)
    except Exception as e:
        print(f"    ! read fail {os.path.basename(path)}: {e}")
        return None
    if len(g) == 0:
        return None
    # numbers already share the patch grid; override the tag only if it drifts
    # (e.g. a seed shp from an older forced-CRS run) -- never reproject.
    if bundle_crs is not None and g.crs != bundle_crs:
        g = g.set_crs(bundle_crs, allow_override=True)
    if "class" not in g.columns:
        g["class"] = 0
    g["class"] = pd.to_numeric(g["class"], errors="coerce").fillna(0).astype("int32")
    g["src"] = "auto"
    g["layer"] = os.path.basename(path)[:-4]
    g["patch_id"] = pid
    return g[["class", "src", "layer", "patch_id", "geometry"]]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n_chunks = math.ceil(N_PATCHES / CHUNK)
    print(f"{N_PATCHES} tiles / CHUNK {CHUNK} = {n_chunks} bundles -> {OUT_DIR}")

    bundle_crs = None
    totals = {"bundles": 0, "naip": 0, "chm": 0, "ndvi": 0, "objects": 0}

    for c in range(n_chunks):
        start = c * CHUNK + 1
        end   = min(start + CHUNK - 1, N_PATCHES)
        pids  = range(start, end + 1)
        name  = f"group_{start:04d}-{end:04d}"
        gdir  = os.path.join(OUT_DIR, name)
        tar_path = os.path.join(OUT_DIR, name + ".tar.gz")

        if MAKE_TAR and not OVERWRITE and os.path.exists(tar_path):
            print(f"[{name}] tar exists -- skip")
            continue

        os.makedirs(gdir, exist_ok=True)
        print("=" * 60); print(f"[{name}] tiles {start}..{end}")

        n_naip = n_chm = n_ndvi = 0
        vecs = []
        for pid in pids:
            naip = find(pid, "naip.tif")
            if naip is None:
                print(f"    no naip patch_{pid:04d} -- skip"); continue
            if bundle_crs is None:
                with rasterio.open(naip) as s:
                    bundle_crs = s.crs
                print(f"    bundle CRS = {bundle_crs.to_string() if bundle_crs else 'None'}")

            shutil.copy2(naip, os.path.join(gdir, f"naip_{pid:04d}.tif")); n_naip += 1
            chm = find(pid, "chm.tif")
            if chm:
                shutil.copy2(chm, os.path.join(gdir, f"chm_{pid:04d}.tif")); n_chm += 1
            ndvi = find(pid, "ndvi.tif")
            if ndvi:
                shutil.copy2(ndvi, os.path.join(gdir, f"ndvi_{pid:04d}.tif")); n_ndvi += 1

            lbl = find(pid, "label.shp")               # building/road/water
            if lbl:
                v = load_vec(lbl, bundle_crs, pid)
                if v is not None:
                    vecs.append(v)
            for shp in find_seed_shps(pid):            # tree/shrub/grass (chm/SAM)
                v = load_vec(shp, bundle_crs, pid)
                if v is not None:
                    vecs.append(v)

        if n_naip == 0:
            print("    empty chunk -- removing"); shutil.rmtree(gdir, ignore_errors=True); continue

        build_vrt(gdir, "naip_group.vrt", "naip_*.tif")
        build_vrt(gdir, "chm_group.vrt",  "chm_*.tif")
        build_vrt(gdir, "ndvi_group.vrt", "ndvi_*.tif")

        # THE edit file: all pre-made features merged, each object kept separate.
        out_shp = os.path.join(gdir, "label_group.shp")
        if vecs:
            merged = gpd.GeoDataFrame(pd.concat(vecs, ignore_index=True),
                                      geometry="geometry", crs=bundle_crs)
        else:
            merged = gpd.GeoDataFrame(
                {"class": pd.Series([], dtype="int32"), "src": pd.Series([], dtype=object),
                 "layer": pd.Series([], dtype=object), "patch_id": pd.Series([], dtype="int32")},
                geometry=gpd.GeoSeries([], crs=bundle_crs))
        merged.to_file(out_shp)
        n_obj = len(merged)
        print(f"    naip {n_naip} | chm {n_chm} | ndvi {n_ndvi} | label_group {n_obj} objects")

        if MAKE_TAR:
            with tarfile.open(tar_path, "w:gz") as t:
                t.add(gdir, arcname=name)
            print(f"    -> {os.path.basename(tar_path)}")
            if not KEEP_FOLDER:
                shutil.rmtree(gdir, ignore_errors=True)

        totals["bundles"] += 1; totals["naip"] += n_naip; totals["chm"] += n_chm
        totals["ndvi"] += n_ndvi; totals["objects"] += n_obj

    with open(os.path.join(OUT_DIR, "_run_info.txt"), "w") as f:
        f.write(f"03_build_bundles.py run {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"PATCH_DIR={PATCH_DIR}\nSEEDS_DIR={SEEDS_DIR}\n")
        f.write(f"N_PATCHES={N_PATCHES} CHUNK={CHUNK}\n")
        f.write(f"bundle_crs={bundle_crs.to_string() if bundle_crs else 'None'}\ntotals={totals}\n")
    print("=" * 60); print(f"done {totals}")


if __name__ == "__main__":
    main()
