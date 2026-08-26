#!/bin/bash
#SBATCH -J phx_tiles
#SBATCH -p public
#SBATCH -q public
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 0-04:00:00
#SBATCH -o /path/to/Phoenix/Result/logs/%x_%j.out

# CPU-only. v3: uses ONLY the hand-modified label groups in
#   /path/to/Phoenix/labeling_hand/Label_modified/label_XXXX-YYYY/
# (rasters still come from labeling_new/label_bundles), applies the
# Tree-over-Building overlap rule, and writes TWO tile sets in one run:
#   /path/to/Phoenix/scripts/Tile/256/tiles_<date-time>_n<count>/  (full tiles)
#   /path/to/Phoenix/scripts/Tile/128/tiles_<date-time>_n<count>/  (2x2 crops, 4x count)
# Both folder paths are printed at the end of the log -- pass the wanted one
# to train_phoenix.py via --tile-root.
mkdir -p /path/to/Phoenix/Result/logs

module load mamba/latest
source activate geopandas
# env 'geopandas' (conda-forge geopandas/rasterio/shapely/pyproj/numpy) for prepare.
# create with:
#   mamba create -n geopandas -c conda-forge python=3.11 \
#       geopandas rasterio shapely pyproj numpy -y
# (train/predict use the pytorch_gpu env)

cd /path/to/Phoenix/scripts
# call the env python explicitly: 'module load mamba' puts mamba's own python
# ahead on PATH and it shadows the activated env, so plain 'python' misses the
# env packages.
"$CONDA_PREFIX/bin/python" prepare_tiles_training.py