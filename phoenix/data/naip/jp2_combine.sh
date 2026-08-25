#!/bin/bash
#SBATCH -p public
#SBATCH -q public
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 0-012:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH -o "/path/to/Phoenix/NAIP_Raw/naip_jp2_to_gtiff_%j.out"
#SBATCH -e "/path/to/Phoenix/NAIP_Raw/naip_jp2_to_gtiff_%j.err"

cd /path/to/Phoenix/NAIP_Raw
$HOME/.conda/envs/geopandas/bin/python -u jp2_combine.py
