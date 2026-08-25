#!/bin/bash
#SBATCH -J build_bundles
#SBATCH -p public
#SBATCH -q public
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-03:00:00
#SBATCH -o build_bundles_%j.out
#SBATCH -e build_bundles_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

PY="$HOME/.conda/envs/geopandas/bin/python"
PROJ_DIR="$("$PY" -c 'import pyproj; print(pyproj.datadir.get_data_dir())')"
export PROJ_DATA="$PROJ_DIR"; export PROJ_LIB="$PROJ_DIR"; export PROJ_NETWORK=OFF
echo "PROJ_DATA=$PROJ_DATA"
"$PY" -u /path/to/Phoenix/labeling_new/03_build_bundles.py
