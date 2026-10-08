#!/bin/bash
# ============================================================
# SLURM — pass@k trajectory eval (Qwen2.5-7B, GSM8K+MATH500)
# Cluster: CSCS Clariden (GH200 120GB)
#
# 3 methods × 6 steps {40,80,120,160,200,240} × 2 tasks
# = 36 (method, step, task) jobs, round-robin across 4 GPUs.
# Per task: T=1.0, top_p=1.0; N=8 (GSM8K) / N=16 (MATH500).
#
# Output: eval_results_final/passk_trajectory_7B/
#
# Usage:
#   sbatch open_r1_imp/scripts/submit_passk_trajectory_7b.sh
#
#   # Subset:
#   sbatch --export=ALL,STEPS=80,160,240,METHODS=multi_reward_klcov05 \
#       open_r1_imp/scripts/submit_passk_trajectory_7b.sh
# ============================================================

#SBATCH --job-name=passk7b
#SBATCH --account=a0264
#SBATCH --qos=slowdown
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --partition=low
#SBATCH --output=logs/passk7b-%j.out
#SBATCH --error=logs/passk7b-%j.err
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -euo pipefail

PROJECT_DIR="${HOME}/projects/unsupervised-rl"
VENV_DIR="${PROJECT_DIR}/venv"

STEPS="${STEPS:-40,80,120,160,200,240}"
METHODS="${METHODS:-}"
TASKS="${TASKS:-gsm8k,math500}"
OVERWRITE="${OVERWRITE:-0}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-unsupervised-rl-eval}"
WANDB_GROUP="${WANDB_GROUP:-passk_trajectory_7b_v1}"

mkdir -p "${PROJECT_DIR}/logs"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -c "import math_verify, latex2sympy2_extended" 2>/dev/null || \
    pip install --quiet math-verify latex2sympy2-extended || true

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=18

export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_DISABLE_WEAVE=true
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_GROUP="${WANDB_GROUP}"

echo "============================================================"
echo "  pass@k trajectory — Qwen2.5-7B (4 GPUs parallel)"
echo "  Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "  Node:         $(hostname)"
echo "  Steps:        ${STEPS}"
echo "  Methods:      ${METHODS:-<all>}"
echo "  Tasks:        ${TASKS}"
echo "  Overwrite:    ${OVERWRITE}"
echo "  Wandb:        ${WANDB} (project=${WANDB_PROJECT}, group=${WANDB_GROUP})"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================================"

ARGS=(--scale 7b --num_gpus 4 --steps "${STEPS}" --tasks "${TASKS}")
[ -n "${METHODS}" ] && ARGS+=(--methods "${METHODS}")
[ "${OVERWRITE}" = "1" ] && ARGS+=(--overwrite)
[ "${WANDB}" = "1" ] && ARGS+=(--wandb --wandb_project "${WANDB_PROJECT}" --wandb_group "${WANDB_GROUP}")

python -u evaluation/passk_trajectory.py "${ARGS[@]}"

echo ""
echo "=== Aggregating + plotting ==="
python -u evaluation/summarize_passk_trajectory.py --scale 7b || true

echo "[$(date)] passk_trajectory 7B done."
