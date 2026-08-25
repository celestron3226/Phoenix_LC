#!/bin/bash
#SBATCH -p public
#SBATCH -q public
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-04:00:00
#SBATCH -J veg_seeds
#SBATCH -o veg_seeds_%A_%a.out
#SBATCH -e veg_seeds_%A_%a.err
#SBATCH --array=0-49%10
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
# 50 tasks x 60 patches = 3000.  %10 = at most 10 GPUs at once.
# N_TASKS=50 in 02_make_veg_seeds.py MUST match --array=0-49.
# Resumable via .done markers. Array emails per task (~150) -> set FAIL only if noisy.

PYBIN=$HOME/.conda/envs/segment/bin/python

# proj.db lives inside the segment env (PROJ via pip/pyproj). Point PROJ at it.
PROJ_DB=$(find $HOME/.conda/envs/segment -name proj.db 2>/dev/null | head -1)
if [ -n "$PROJ_DB" ]; then
  export PROJ_DATA=$(dirname "$PROJ_DB")
  export PROJ_LIB=$PROJ_DATA
fi
# GTIFF_SRS_SOURCE was the LOCAL_CS workaround for reading raw NAIP. Pass 2 no
# longer clips raw sources (reads Pass 1's clean patch tiles), so it's optional
# now -- kept as a harmless safety net.
export GTIFF_SRS_SOURCE=EPSG
echo "PROJ_DATA=${PROJ_DATA:-<unset: using bundled>}"

$PYBIN -u 02_make_veg_seeds.py
