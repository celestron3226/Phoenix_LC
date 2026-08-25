#!/bin/bash
#SBATCH -J chm_pipeline
#SBATCH -p public
#SBATCH -q public
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH -t 0-12:00:00
#SBATCH -o chm_pipeline_%j.out
#SBATCH -e chm_pipeline_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# Resume-friendly: each LAZ -> DTM/DSM step skips files that already exist,
# so re-submitting after a timeout will pick up where it left off.

PY="$HOME/.conda/envs/geopandas/bin/python"

# PROJ data dir (same fix as the patch script)
PROJ_DIR="$("$PY" -c 'import pyproj; print(pyproj.datadir.get_data_dir())')"
export PROJ_DATA="$PROJ_DIR"
export PROJ_LIB="$PROJ_DIR"
export PROJ_NETWORK=OFF
echo "PROJ_DATA=$PROJ_DATA"

# Make PDAL CLI visible (installed into geopandas env)
export PATH="$HOME/.conda/envs/geopandas/bin:$PATH"
echo "pdal: $(which pdal)"
pdal --version | head -1

"$PY" -u /path/to/Phoenix/LiDAR/LAZ_to_CHM.py
