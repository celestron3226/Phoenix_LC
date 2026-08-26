#!/bin/bash
#SBATCH -J phxlc_pred
#SBATCH -p public
#SBATCH -q public
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH -a 0-55%4
#SBATCH -G a100:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 0-04:00:00
#SBATCH -o /path/to/Phoenix_LC/logs/%x_%A_%a.out

# GPU array: one NAIP quad per task, 4 GPUs at a time (%4).
# Reads the 256px .npz prediction tiles from /path/to/Phoenix_LC/tile
# and writes one landcover_<stem>.tif per quad to
# /path/to/Phoenix_LC/predict/lc_per_quad/.
#
# Set the array range to (number of finished quad folders - 1); a too-large
# range is harmless (extra tasks print "nothing to do" and exit).
#
# Resume-safe: quads whose output tif exists are skipped, so resubmit the
# same script after any timeout/failure.
#
# AFTER all tasks finish, run make_maps.sh for mosaic + city clip + shp.
mkdir -p /path/to/Phoenix_LC/logs

module load mamba/latest
source activate pytorch_gpu

cd /path/to/Phoenix_LC/predict
# absolute env python: 'source activate' is unreliable on compute nodes and
# mamba's own python shadows the env python on PATH.
$HOME/.conda/envs/pytorch_gpu/bin/python predict_phoenix_lc.py
