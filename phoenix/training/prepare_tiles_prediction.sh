#!/bin/bash
#SBATCH -J phxlc_tiles
#SBATCH -p public
#SBATCH -q public
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH -a 0-55%8
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 0-04:00:00
#SBATCH -o /path/to/Phoenix_LC/logs/%x_%A_%a.out

# CPU-only, SLURM ARRAY: one quad per task, 8 tasks at a time (%8).
# Task index picks a quad from the sorted usable list inside the .py.
#
# BEFORE SUBMIT: set the array range to the actual quad count minus 1:
#   ls /path/to/Phoenix/NAIP_Raw/NAIP_raw/m_*.tif | wc -l
# e.g. 56 quads -> -a 0-55%8. A too-large range is harmless (extra tasks
# just print "nothing to do" and exit).
#
# Resume-safe: finished quads (meta.json present) are skipped, so resubmit
# the same script after any timeout/failure and only unfinished quads run.
mkdir -p /path/to/Phoenix_LC/logs

module load mamba/latest
source activate geopandas
# env 'geopandas' (only rasterio/numpy are actually used here)

cd /path/to/Phoenix_LC/tile
# call the env python explicitly: 'module load mamba' puts mamba's own python
# ahead on PATH and it shadows the activated env, so plain 'python' misses the
# env packages.
"$CONDA_PREFIX/bin/python" prepare_tiles_prediction.py
