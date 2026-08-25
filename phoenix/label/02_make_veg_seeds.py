"""
Per-patch vegetation SEED labels -- in-place / REAL coordinates (no mosaic).

For each patch in PATCH_IDS, using its matching NAIP + NDVI + CHM:

    veg   = NDVI >= NDVI_VEG_MIN
    grass = veg & (CHM <  0.3)          -> class 3
    shrub = veg & (0.3 <= CHM <= 1.5)   -> class 2
    tree  = veg & (CHM >  1.5)          -> class 1

grass / shrub  -> connected-component polygons (no SAM3; they are ground cover,
                  not discrete crowns -> box-prompting is meaningless).
tree           -> TWO methods, saved SEPARATELY so you can compare & delete one:
    A) rule       : tree mask -> CC -> boxes -> SAM3   -> seed_tree_ruleSAM_XXXX.shp
    B) deepforest : DeepForest -> score + CHM/NDVI filter -> SAM3 -> seed_tree_dfSAM_XXXX.shp

Every polygon carries:  id (per layer), class (1/2/3), chm_m (median height, m).

WHY IN-PLACE: outputs are born at each patch's real CRS/transform, perfectly
co-registered with that patch's naip -> directly trainable, NO split-back, and
absolute coords stay valid for SatCLIP / CoordConv / city-wide prediction.

.qml sidecars give per-category colours in QGIS (categorized on `class`; the two
tree methods get distinct outline colours for A/B comparison).

Run in the 'segment' env (GPU). SAM3 needs prior `hf auth login`.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # before torch

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from datetime import datetime
import glob
import math
import shutil
import subprocess
import numpy as np
import torch
import rasterio
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import xy
from rasterio.features import shapes
from shapely.geometry import box, shape
import geopandas as gpd
import pandas as pd
from scipy import ndimage
from deepforest import main
from samgeo import SamGeo3
from samgeo.common import raster_to_vector

# ============================================================
# Paths  ---  >>> CHECK THESE <<<
# (Sol defaults. NDVI/CHM may be a DIRECTORY of per-tile files -> auto-VRT,
#  or a single merged .tif -> used directly. Both work.)
# ============================================================
PATCH_DIR  = r"/path/to/Phoenix/labeling_new/patches"          # patch_XXXX/ folders
NDVI_SRC   = r"/path/to/Phoenix/NDVI"                      # dir(56) or single .tif
CHM_SRC    = r"/path/to/Phoenix/LiDAR/per_naip"            # dir(56) or single .tif
OUTPUT_DIR = r"/path/to/Phoenix/labeling_new/seeds"

# Patch selection ------------------------------------------------------------
# Full run = patch_0001 .. patch_1000. Launched as a SLURM job array, each task
# auto-takes its own slice (N_TASKS MUST match --array width in the .sh).
# No array env -> this process does ALL patches in one go.
N_PATCHES = 3000
N_TASKS   = 50          # must equal --array=0-49 in make_veg_seeds.sh
_aid = os.environ.get("SLURM_ARRAY_TASK_ID")
if _aid is not None:
    _aid = int(_aid)
    _per = math.ceil(N_PATCHES / N_TASKS)
    _start = _aid * _per + 1
    _end   = min(_start + _per - 1, N_PATCHES)
    PATCH_IDS = range(_start, _end + 1)
    print(f"[array task {_aid}/{N_TASKS}] patches {_start}..{_end}")
else:
    PATCH_IDS = range(1, N_PATCHES + 1)

# ============================================================
# Params
# ============================================================
NDVI_VEG_MIN = 0.20     # vegetation cutoff (lower -> more recall; see notes)
GRASS_MAX    = 0.30     # CHM <  0.30  -> grass
SHRUB_MIN    = 0.30     # 0.30 <= CHM <= 1.50 -> shrub
SHRUB_MAX    = 1.50
TREE_MIN     = 1.50     # CHM >  1.50  -> tree

MIN_PX_VEG  = 6        # grass/shrub: keep small shrubs (a 1 m shrub ~ 9 px). Tune.
MIN_PX_TREE = 30       # tree: bigger min before SAM box-prompt
OPEN_IT  = 1           # opening: tree mask only (despeckle / split touching blobs)
CLOSE_IT = 1           # closing: fill small holes

SCORE_THR    = 0.20     # DeepForest score filter (method B)
DF_PATCH     = 400      # DeepForest predict_tile patch_size (clamped to patch dims)
DF_OVERLAP   = 0.15
SAM_BANDS    = [1, 2, 3]

# NAIP/NDVI/CHM are all EPSG:3857 (confirmed). Their embedded GeoTIFF keys read as
# a LOCAL_CS / "engineering" CRS that breaks WarpedVRT, so force a clean definition
# everywhere -> source==target==clean 3857 -> identity warp, no proj transform.
FORCE_EPSG = 3857
FORCE_CRS  = CRS.from_epsg(FORCE_EPSG)

os.makedirs(OUTPUT_DIR, exist_ok=True)
TMP = os.path.join(OUTPUT_DIR, f"_tmp_{_aid}" if _aid is not None else "_tmp")
os.makedirs(TMP, exist_ok=True)
print(f"Output: {OUTPUT_DIR}")

# ============================================================
# QGIS style sidecars
# ============================================================
PALETTE = {  # class code -> (label, "R,G,B,A")  A<255 -> imagery shows through
    1: ("Tree",     "27,120,55,160"),
    2: ("Shrub",    "140,170,80,160"),
    3: ("Grass",    "120,200,90,160"),
    4: ("Soil",     "200,160,110,160"),
    5: ("Building", "220,70,50,160"),
    6: ("Asphalt",  "90,90,95,160"),
    7: ("Water",    "60,120,200,160"),
    8: ("Agri",     "180,90,180,160"),
}

def _fill_layer(rgba, outline="20,20,20,255", width="0.3"):
    return (f'<layer class="SimpleFill" enabled="1" locked="0" pass="0">'
            f'<Option type="Map">'
            f'<Option name="color" type="QString" value="{rgba}"/>'
            f'<Option name="outline_color" type="QString" value="{outline}"/>'
            f'<Option name="outline_width" type="QString" value="{width}"/>'
            f'<Option name="outline_style" type="QString" value="solid"/>'
            f'<Option name="style" type="QString" value="solid"/>'
            f'</Option></layer>')

def categorized_qml():
    cats, syms = [], []
    for i, (code, (name, rgba)) in enumerate(PALETTE.items()):
        cats.append(f'<category render="true" value="{code}" symbol="{i}" label="{name}"/>')
        syms.append(f'<symbol type="fill" name="{i}" alpha="1" clip_to_extent="1" force_rhr="0">'
                    f'{_fill_layer(rgba)}</symbol>')
    return ('<!DOCTYPE qgis>\n<qgis version="3.34.0" styleCategories="Symbology">\n'
            '<renderer-v2 type="categorizedSymbol" attr="class" forceraster="0" '
            'symbollevels="0" enableorderby="0">\n'
            f'<categories>{"".join(cats)}</categories>\n'
            f'<symbols>{"".join(syms)}</symbols>\n'
            '</renderer-v2>\n<layerGeometryType>2</layerGeometryType>\n</qgis>')

def single_qml(fill, outline, width="0.5"):
    return ('<!DOCTYPE qgis>\n<qgis version="3.34.0" styleCategories="Symbology">\n'
            '<renderer-v2 type="singleSymbol" forceraster="0" symbollevels="0" enableorderby="0">\n'
            f'<symbols><symbol type="fill" name="0" alpha="1" clip_to_extent="1" force_rhr="0">'
            f'{_fill_layer(fill, outline, width)}</symbol></symbols>\n'
            '</renderer-v2>\n<layerGeometryType>2</layerGeometryType>\n</qgis>')

QML_CATEG     = categorized_qml()
QML_TREE_RULE = single_qml("27,120,55,70", "120,255,80,255")    # lime outline
QML_TREE_DF   = single_qml("27,120,55,70", "255,60,200,255")    # magenta outline

def write_qml(shp_path, text):
    with open(shp_path[:-4] + ".qml", "w") as f:
        f.write(text)

# ============================================================
# Helpers
# ============================================================
def find_patch_naip(folder):
    """naip.tif or naip_XXXX.tif inside a patch folder."""
    c = glob.glob(os.path.join(folder, "naip*.tif")) + glob.glob(os.path.join(folder, "naip*.tiff"))
    return c[0] if c else None

def ensure_source(src):
    """Single .tif -> return as-is. Directory of tiles -> build a VRT once."""
    if os.path.isfile(src):
        return src
    if os.path.isdir(src):
        tifs = sorted(glob.glob(os.path.join(src, "*.tif")) + glob.glob(os.path.join(src, "*.tiff")))
        if not tifs:
            raise SystemExit(f"No .tif in {src}")
        vrt = os.path.join(OUTPUT_DIR, os.path.basename(src.rstrip("/")) + ".vrt")
        if not os.path.exists(vrt):
            print(f"Building VRT: {vrt}  ({len(tifs)} tiles)")
            subprocess.run(["gdalbuildvrt", vrt, *tifs], check=True,
                           stdout=subprocess.DEVNULL)
        return vrt
    raise SystemExit(f"Source not found: {src}")

def read_on_grid(path, transform, w, h):
    """Clip onto the patch grid. All inputs are EPSG:FORCE_EPSG, so we override
    BOTH source and target CRS to a clean definition -- avoids the LOCAL_CS /
    GeoTIFF-keys-vs-EPSG mismatch that otherwise crashes WarpedVRT. The warp is
    then an identity reproject + windowed resample to the patch grid."""
    with rasterio.open(path) as r:
        with WarpedVRT(r, src_crs=FORCE_CRS, crs=FORCE_CRS, transform=transform,
                       width=w, height=h, resampling=Resampling.bilinear) as vrt:
            arr = vrt.read(1).astype("float32")
            nd = vrt.nodata
    if nd is not None:
        arr = np.where(arr == nd, np.nan, arr)
    return np.nan_to_num(arr, nan=-9999.0)

def write_tif(arr, transform, crs, path):
    prof = {"driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
            "count": 1, "dtype": "float32", "crs": crs,
            "transform": transform, "nodata": -9999.0}
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)

def zonal_median(gdf, raster):
    """Median raster value per polygon. rasterstats if available, else rasterio."""
    with rasterio.open(raster) as r:
        rc, nd = r.crs, r.nodata
    g = gdf.to_crs(rc) if gdf.crs != rc else gdf
    try:
        from rasterstats import zonal_stats
        return [s["median"] for s in zonal_stats(g, raster, stats=["median"], nodata=nd)]
    except ImportError:
        meds = []
        with rasterio.open(raster) as r:
            for geom in g.geometry:
                try:
                    win = rasterio.windows.from_bounds(*geom.bounds, transform=r.transform)
                    a = r.read(1, window=win)
                    if nd is not None:
                        a = a[a != nd]
                    meds.append(float(np.median(a)) if a.size else None)
                except Exception:
                    meds.append(None)
        return meds

def clean(mask, open_it, close_it):
    if open_it:
        mask = ndimage.binary_opening(mask, iterations=open_it)
    if close_it:
        mask = ndimage.binary_closing(mask, iterations=close_it)
    return mask

def polygonize(mask, transform, crs):
    """Mask -> blob polygons (grass/shrub). Closing only (NO opening) so small
    shrubs survive; drop sub-MIN_PX_VEG speckle."""
    m = clean(mask, 0, CLOSE_IT)
    px_area = abs(transform.a * transform.e)
    geoms = [shape(g) for g, v in shapes(m.astype(np.uint8), mask=m, transform=transform)
             if v == 1 and shape(g).area >= MIN_PX_VEG * px_area]
    return gpd.GeoDataFrame(geometry=geoms, crs=crs)

def boxes_from_mask(mask, transform, crs):
    """Mask -> one bbox per connected component (tree, for SAM prompting)."""
    m = clean(mask, OPEN_IT, CLOSE_IT)
    labels, _ = ndimage.label(m, structure=np.ones((3, 3)))
    boxes = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        rows, cols = sl
        if int((labels[sl] == i).sum()) < MIN_PX_TREE:
            continue
        left, top = xy(transform, rows.start, cols.start, offset="ul")
        right, bottom = xy(transform, rows.stop, cols.stop, offset="ul")
        boxes.append(box(left, bottom, right, top))
    return gpd.GeoDataFrame({"id": range(len(boxes))}, geometry=boxes, crs=crs)

def sam_polys(sam, boxes_gdf, naip_crs, tag):
    """Run SAM3 box-prompts -> crown polygons (GeoDataFrame). Empty -> empty gdf."""
    if len(boxes_gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=naip_crs)
    if boxes_gdf.crs != naip_crs:
        boxes_gdf = boxes_gdf.to_crs(naip_crs)
    bf = os.path.join(TMP, f"boxes_{tag}.gpkg")
    mt = os.path.join(TMP, f"mask_{tag}.tif")
    pg = os.path.join(TMP, f"poly_{tag}.gpkg")
    boxes_gdf.assign(id=range(len(boxes_gdf)))[["id", "geometry"]].to_file(bf, driver="GPKG")
    sam.generate_masks_by_boxes_inst(bf)
    sam.save_masks(output=mt, unique=True)
    torch.cuda.empty_cache()
    raster_to_vector(source=mt, output=pg)
    return gpd.read_file(pg)

def attr_save(gdf, cls, out_shp, qml):
    """Add id/class, write shp + qml. Returns feature count.
    Per-PIXEL height comes from the raw CHM raster (chm_XXXX.tif), NOT a
    per-polygon median -- click any pixel with QGIS Identify to read metres."""
    if len(gdf) == 0:
        print(f"    (0 features) {os.path.basename(out_shp)}")
        return 0
    gdf = gdf.reset_index(drop=True)
    gdf["id"] = range(1, len(gdf) + 1)
    gdf["class"] = cls
    gdf = gdf[["id", "class", "geometry"]]
    gdf = gdf.set_crs(FORCE_CRS, allow_override=True)   # clean 3857 (coords already 3857 m)
    gdf.to_file(out_shp)
    write_qml(out_shp, qml)
    print(f"    {len(gdf):4d} -> {os.path.basename(out_shp)}")
    return len(gdf)

# ============================================================
# Models (load once)
# ============================================================
print("Loading DeepForest + SAM3 ...")
df_model = main.deepforest()
df_model.load_model(model_name="weecology/deepforest-tree", revision="main")
try:
    df_model.model = df_model.model.float()   # fp32 weights (SAM3 may flip global precision)
except Exception:
    pass
sam = SamGeo3(backend="meta", enable_inst_interactivity=True)

# NDVI/CHM are NOT clipped here anymore: Pass 1 (01_make_patches.py) already wrote
# chm.tif / ndvi.tif onto each patch grid. ensure_source()/read_on_grid() are left
# defined but unused (clip-once lives in Pass 1). This avoids the giant-source
# windowed reads (JP2 amplification) and a duplicate NDVI clip.

# ============================================================
# Per-patch loop
# ============================================================
summary = []
naip_list, chm_list = [], []   # for the real-coord combined VRT
for pid in PATCH_IDS:
    folder = os.path.join(PATCH_DIR, f"patch_{pid:04d}")
    naip = find_patch_naip(folder)
    if naip is None:
        print(f"[patch_{pid:04d}] no naip found in {folder} -- skip")
        continue

    print("=" * 60)
    print(f"[patch_{pid:04d}] {naip}")
    outp = os.path.join(OUTPUT_DIR, f"patch_{pid:04d}")
    os.makedirs(outp, exist_ok=True)
    done_marker = os.path.join(outp, ".done")
    if os.path.exists(done_marker):
        print(f"    already done -- skip")
        continue

    with rasterio.open(naip) as s:
        transform, H, W = s.transform, s.height, s.width
    crs = FORCE_CRS   # MUST equal Pass 1's SRC_CRS_OVERRIDE stamp (default EPSG:3857)

    # chm/ndvi were already clipped to THIS exact patch grid by Pass 1 -> just read
    # them (no source clip, no re-write). chm.tif stays the kept height raster.
    chm_tif  = os.path.join(folder, "chm.tif")    # Pass 1 output: raw height (m)
    ndvi_tif = os.path.join(folder, "ndvi.tif")   # Pass 1 output: NDVI
    with rasterio.open(chm_tif) as r:
        chm = r.read(1)
    with rasterio.open(ndvi_tif) as r:
        ndvi = r.read(1)
    naip_list.append(naip)
    chm_list.append(chm_tif)

    veg   = ndvi >= NDVI_VEG_MIN
    grass = veg & (chm < GRASS_MAX)
    shrub = veg & (chm >= SHRUB_MIN) & (chm <= SHRUB_MAX)
    tree  = veg & (chm > TREE_MIN)
    print(f"    veg px: {int(veg.sum())}  grass:{int(grass.sum())} "
          f"shrub:{int(shrub.sum())} tree:{int(tree.sum())}")

    n = {"patch": f"patch_{pid:04d}"}

    # --- grass / shrub : CC polygons (no SAM) ---
    n["grass"] = attr_save(polygonize(grass, transform, crs), 3,
                           os.path.join(outp, f"seed_grass_{pid:04d}.shp"), QML_CATEG)
    n["shrub"] = attr_save(polygonize(shrub, transform, crs), 2,
                           os.path.join(outp, f"seed_shrub_{pid:04d}.shp"), QML_CATEG)

    # --- tree : SAM3 image set once, two prompt sources ---
    sam.set_image(naip, bands=SAM_BANDS)

    # A) rule boxes -> SAM
    try:
        boxes_a = boxes_from_mask(tree, transform, crs)
        polys_a = sam_polys(sam, boxes_a, crs, f"{pid:04d}_rule")
        n["tree_rule"] = attr_save(polys_a, 1,
                                   os.path.join(outp, f"seed_tree_ruleSAM_{pid:04d}.shp"),
                                   QML_TREE_RULE)
    except Exception as e:
        print(f"    WARNING rule method failed: {e}")
        n["tree_rule"] = -1

    # B) DeepForest -> score + CHM/NDVI filter -> SAM
    try:
        ps = max(64, min(DF_PATCH, H, W))
        # Force fp32: SAM3 can leave global precision in bf16, which makes
        # DeepForest's tensor->numpy conversion crash ("unsupported BFloat16").
        _prev_dt = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            with torch.autocast(device_type="cuda", enabled=False):
                preds = df_model.predict_tile(path=naip, patch_size=ps,
                                              patch_overlap=DF_OVERLAP)
        finally:
            torch.set_default_dtype(_prev_dt)
        if preds is None or len(preds) == 0:
            polys_b = gpd.GeoDataFrame(geometry=[], crs=crs)
        else:
            preds = preds[preds["score"] > SCORE_THR].reset_index(drop=True)
            geoms = []
            for _, r in preds.iterrows():
                left, top = xy(transform, r["ymin"], r["xmin"])
                right, bottom = xy(transform, r["ymax"], r["xmax"])
                geoms.append(box(left, bottom, right, top))
            dfb = gpd.GeoDataFrame(geometry=geoms, crs=crs)
            if len(dfb):
                dfb["chm_med"]  = zonal_median(dfb, chm_tif)
                dfb["ndvi_med"] = zonal_median(dfb, ndvi_tif)
                keep = (pd.notna(dfb["chm_med"]) & pd.notna(dfb["ndvi_med"])
                        & (dfb["chm_med"] >= TREE_MIN) & (dfb["ndvi_med"] >= NDVI_VEG_MIN))
                dfb = dfb[keep].reset_index(drop=True)
            polys_b = sam_polys(sam, dfb, crs, f"{pid:04d}_df")
        n["tree_df"] = attr_save(polys_b, 1,
                                 os.path.join(outp, f"seed_tree_dfSAM_{pid:04d}.shp"),
                                 QML_TREE_DF)
    except Exception as e:
        print(f"    WARNING deepforest method failed: {e}")
        n["tree_df"] = -1

    # cleanup per-patch temp
    for f in glob.glob(os.path.join(TMP, f"*_{pid:04d}*")) + glob.glob(os.path.join(TMP, f"*{pid:04d}_*")):
        try:
            os.remove(f)
        except OSError:
            pass
    open(done_marker, "w").close()   # mark complete -> reruns skip this patch
    summary.append(n)

# ============================================================
# Real-coord combined view (NOT a fake grid): patches at their TRUE positions.
# Built only for a single (non-array) run, since each array task sees just its
# slice. After an array run, build the full VRTs once with:
#   gdalbuildvrt OUTPUT_DIR/all_chm.vrt  OUTPUT_DIR/patch_*/chm_*.tif
# ============================================================
if _aid is None and naip_list:
    nvrt = os.path.join(OUTPUT_DIR, "all_naip.vrt")
    cvrt = os.path.join(OUTPUT_DIR, "all_chm.vrt")
    subprocess.run(["gdalbuildvrt", nvrt, *naip_list], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["gdalbuildvrt", cvrt, *chm_list],  check=True, stdout=subprocess.DEVNULL)
    print(f"Combined (real coords): {nvrt} | {cvrt}")

# ============================================================
# Run info
# ============================================================
shutil.rmtree(TMP, ignore_errors=True)
_info = f"run_info_task{_aid}.txt" if _aid is not None else "run_info.txt"
with open(os.path.join(OUTPUT_DIR, _info), "w") as f:
    f.write(f"Run: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    f.write(f"NDVI_VEG_MIN={NDVI_VEG_MIN}  grass<{GRASS_MAX}  "
            f"shrub[{SHRUB_MIN},{SHRUB_MAX}]  tree>{TREE_MIN}\n")
    f.write(f"score_thr={SCORE_THR}  min_px_veg={MIN_PX_VEG}  min_px_tree={MIN_PX_TREE}\n\n")
    f.write("patch        grass shrub tree_rule tree_df\n")
    for n in summary:
        f.write(f"{n['patch']}  {n.get('grass',0):5d} {n.get('shrub',0):5d} "
                f"{n.get('tree_rule',0):9d} {n.get('tree_df',0):7d}\n")

print("=" * 60)
print(f"Done. Per-patch outputs -> {OUTPUT_DIR}/patch_XXXX/")
print("  chm_XXXX.tif (raw height, Identify->m) + seed_grass/shrub/tree_ruleSAM/tree_dfSAM (+.qml)")
print(f"Real-coord combined view -> {OUTPUT_DIR}/batch_naip.vrt , batch_chm.vrt")
print("=" * 60)
