#!/bin/bash
# ONE SLURM ARRAY TASK = ONE RUN (arch x init x patch x seed), 78 runs total.
#
# Layout: everything is relative to the folder you run sbatch FROM
# (SLURM_SUBMIT_DIR), so the folder can have any name:
#   <here>/  this file + encoder_experiment_week13_sol.py + phoenix_common.py
#   <here>/slurm_out/  SLURM stdout/stderr per array task (must exist before sbatch)
#   <here>/result/     coord-none_128/<init>_<arch>_bs4_seed<N>/<timestamp>/ per run
#                      logs/<condition>_<init>_<arch>_seed<N>_<jobid>.log   per run
#                      results.csv, results_summary.csv
#
# Before the first submission (login node, your GPU conda env):
#   cd <folder with the two scripts + phoenix_common.py>
#   source activate CHANGE_ME_GPU_ENV
#   python encoder_experiment_week13_sol.py --check          # tiles / ckpts found?
#   python encoder_experiment_week13_sol.py --predownload    # ImageNet weights
#   python encoder_experiment_week13_sol.py --list-jobs      # index -> run mapping
#   mkdir -p slurm_out
#
# Submit everything (finished runs exit immediately, so resubmitting is safe):
#   sbatch --array=0-77%8 --time=03:00:00 encoder_experiment_week13_sol.sh
# Limit how many run at once (fair-share friendly):
#   sbatch --array=0-77%6 encoder_experiment_week13_sol.sh
# Faster data loading (changes the augmentation RNG stream vs local):
#   sbatch --export=ALL,EXTRA="--workers 8" encoder_experiment_week13_sol.sh
#   (needs --cpus-per-task=8 above)
# Only some indices (see --list-jobs):
#   sbatch --array=4,5,30,31 encoder_experiment_week13_sol.sh
# Rebuild the csv tables when done:
#   python encoder_experiment_week13_sol.py --aggregate
#
#SBATCH --job-name=w13
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --array=0-77
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm_out/%A_%a.out
#SBATCH --error=slurm_out/%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# the folder sbatch was run from (holds the .py, this .sh and phoenix_common.py)
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
RESULT_DIR=$SCRIPT_DIR/result
LOG_DIR=$RESULT_DIR/logs
N_JOBS=78                      # must equal the number of rows in --list-jobs

mkdir -p "$LOG_DIR"

module purge
module load mamba/latest
source activate CHANGE_ME_GPU_ENV     # EDIT: conda env with torch/torchvision/smp/albumentations

cd "$SCRIPT_DIR"

# task -> exactly one run; the python script slices the same job list
TASK=${SLURM_ARRAY_TASK_ID:-0}
TAG=$(python encoder_experiment_week13_sol.py --job-tag --task-id "$TASK" --n-tasks "$N_JOBS")
LOG=$LOG_DIR/${TAG}_${SLURM_JOB_ID}.log

echo "[SLURM] job ${SLURM_JOB_ID} task ${TASK} -> ${TAG} on $(hostname)"
echo "[SLURM] log -> ${LOG}"

{
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
    python encoder_experiment_week13_sol.py \
        --task-id "$TASK" --n-tasks "$N_JOBS" \
        --workers 0 \
        --train-root "$RESULT_DIR" \
        ${EXTRA:-}
} > "$LOG" 2>&1
