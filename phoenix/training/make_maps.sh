#!/bin/bash
#SBATCH -J phxlc_maps
#SBATCH -p public
#SBATCH -q public
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 1-12:00:00
#SBATCH -o /path/to/Phoenix_LC/logs/%x_%j.out

# CPU-only. Run AFTER every predict_lc.sh array task has finished.
# Builds the two final map products from the per-quad prediction rasters:
#   1) FULL  : all 56 quads mosaicked as-is
#   2) CITY  : the mosaic clipped to the City of Phoenix boundary
# Then polygonizes both into shapefiles.
#
# The raw Council_District.shp has a ring self-intersection that gdalwarp
# rejects ("Cutline polygon is invalid"), so the boundary is repaired with
# ogr2ogr -makevalid into a GPKG first and THAT is used as the cutline.
#
# Resume-safe: steps whose output already exists are skipped, so just
# resubmit this script after a failure.
#
# Requires the gdal CLI tools in the env. If gdalbuildvrt is missing:
#   mamba install -n geopandas -c conda-forge gdal -y
#
# NOTE: ESRI Shapefile was tried first and hit its hard 4GB .shp file cap
# at ~30% of the city map, so vector outputs are GeoPackage (.gpkg). QGIS,
# ArcGIS Pro, and geopandas all read GPKG natively.

set -e

PRED_DIR=/path/to/Phoenix_LC/predict
QUAD_DIR=$PRED_DIR/lc_per_quad
RESULT_DIR=$PRED_DIR
BOUNDARY=/path/to/Phoenix/boundary/Council_District.shp
BOUNDARY_FIXED=$PRED_DIR/boundary_fixed.gpkg

mkdir -p /path/to/Phoenix_LC/logs "$RESULT_DIR"

module load mamba/latest
source activate geopandas
GDAL_BIN="$CONDA_PREFIX/bin"

cd "$PRED_DIR"

N_QUADS=$(ls "$QUAD_DIR"/landcover_*.tif 2>/dev/null | wc -l)
echo "[INFO] per-quad rasters found: $N_QUADS"
if [ "$N_QUADS" -eq 0 ]; then
    echo "[FATAL] no landcover_*.tif in $QUAD_DIR"; exit 1
fi
# Guard against building the mosaic while some quads are still missing.
LEFTOVER=$(ls "$QUAD_DIR"/*.part.tif 2>/dev/null | wc -l)
if [ "$LEFTOVER" -ne 0 ]; then
    echo "[FATAL] $LEFTOVER .part.tif files present, prediction still "
    echo "        running or crashed mid-quad. Finish prediction first."
    exit 1
fi

# ---- 0) repair the boundary geometry ----
# Ring self-intersections in the source shapefile make gdalwarp refuse it
# as a cutline; -makevalid fixes the geometry without changing the outline.
if [ ! -f "$BOUNDARY_FIXED" ]; then
    echo "[STEP] repair boundary -> $BOUNDARY_FIXED"
    "$GDAL_BIN/ogr2ogr" -f GPKG -makevalid "$BOUNDARY_FIXED" "$BOUNDARY"
else
    echo "[SKIP] boundary already repaired"
fi

# ---- 1) FULL mosaic ----
if [ ! -f phoenix_lc_full.tif ]; then
    echo "[STEP] full mosaic"
    "$GDAL_BIN/gdalbuildvrt" phoenix_lc_full.vrt "$QUAD_DIR"/landcover_*.tif
    "$GDAL_BIN/gdal_translate" -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES \
        phoenix_lc_full.vrt phoenix_lc_full.tif
else
    echo "[SKIP] phoenix_lc_full.tif exists"
fi

# ---- 2) CITY clip ----
if [ ! -f phoenix_lc_city.tif ]; then
    echo "[STEP] city boundary clip"
    "$GDAL_BIN/gdalwarp" -overwrite \
        -cutline "$BOUNDARY_FIXED" -crop_to_cutline -dstnodata 0 \
        -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES \
        phoenix_lc_full.tif phoenix_lc_city.tif
else
    echo "[SKIP] phoenix_lc_city.tif exists"
fi

# ---- 3) polygonize to shapefiles ----
# nodata=0 pixels are excluded automatically (default validity mask).
# Field 'lc' holds the class code 1..7.
# gdal_polygonize ships with the 'gdal' PYTHON package, not with libgdal,
# so it can be missing even when gdalwarp works. Resolve it robustly:
if [ -f "$GDAL_BIN/gdal_polygonize.py" ]; then
    POLY=("$GDAL_BIN/gdal_polygonize.py")
elif "$GDAL_BIN/python" -c "import osgeo_utils.gdal_polygonize" 2>/dev/null; then
    POLY=("$GDAL_BIN/python" -m osgeo_utils.gdal_polygonize)
else
    echo "[FATAL] gdal python utils not in this env. On a login node run:"
    echo "        mamba install -n geopandas -c conda-forge gdal -y"
    echo "        then resubmit this script (finished steps are skipped)."
    exit 1
fi

if [ ! -f "$RESULT_DIR/phoenix_lc.gpkg" ]; then
    echo "[STEP] polygonize city -> $RESULT_DIR/phoenix_lc.gpkg"
    "${POLY[@]}" phoenix_lc_city.tif \
        -f GPKG "$RESULT_DIR/phoenix_lc.gpkg" phoenix_lc lc
else
    echo "[SKIP] phoenix_lc.gpkg exists"
fi

if [ ! -f "$RESULT_DIR/phoenix_lc_full.gpkg" ]; then
    echo "[STEP] polygonize full -> $RESULT_DIR/phoenix_lc_full.gpkg"
    "${POLY[@]}" phoenix_lc_full.tif \
        -f GPKG "$RESULT_DIR/phoenix_lc_full.gpkg" phoenix_lc_full lc
else
    echo "[SKIP] phoenix_lc_full.gpkg exists"
fi

echo "[DONE] rasters : $PRED_DIR/phoenix_lc_full.tif, $PRED_DIR/phoenix_lc_city.tif"
echo "[DONE] vectors : $RESULT_DIR/phoenix_lc.gpkg (city), $RESULT_DIR/phoenix_lc_full.gpkg (full)"
