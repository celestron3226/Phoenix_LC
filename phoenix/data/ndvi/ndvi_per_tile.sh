#!/bin/bash
#SBATCH -J ndvi_per_tile
#SBATCH -p public
#SBATCH -q public
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 0-01:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH -o /path/to/Phoenix/NDVI/ndvi_per_tile_%j.out
#SBATCH -e /path/to/Phoenix/NDVI/ndvi_per_tile_%j.err

PY="$HOME/.conda/envs/geopandas/bin/python"

# PROJ data dir
PROJ_DIR="$("$PY" -c 'import pyproj; print(pyproj.datadir.get_data_dir())')"
export PROJ_DATA="$PROJ_DIR"
export PROJ_LIB="$PROJ_DIR"
export PROJ_NETWORK=OFF
echo "PROJ_DATA=$PROJ_DATA"

cd /path/to/Phoenix/NDVI
"$PY" -u ndvi_per_tile.py
