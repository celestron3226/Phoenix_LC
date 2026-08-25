#!/bin/bash
#SBATCH -J laz_download
#SBATCH -p public
#SBATCH -q public
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 0-08:00:00
#SBATCH -o laz_download_%j.out
#SBATCH -e laz_download_%j.err

# Resume-friendly: existing LAZ files are skipped, so this script can be
# re-run after a timeout or partial failure with no risk of duplicate work.

PY="$HOME/.conda/envs/geopandas/bin/python"
"$PY" -u /path/to/Phoenix/LiDAR/download_laz.py
