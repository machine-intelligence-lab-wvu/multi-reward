#!/bin/bash
# ============================================================
# SLURM — Trajectory eval for the GDPO-normalization ablation
#         (Qwen2.5-1.5B, multi-reward klcov 0.05 variants)
# Cluster: CSCS Clariden (GH200 GPUs)
#
# Evaluates every-20-step checkpoint of the requested method(s)
# on GSM8K, MATH500, and MMLU-Pro. Distributes (method, ckpt)
# jobs across 4 GPUs.
#
# Output:
#   eval_results_final/gdpo_ablation/{method}_ckpt{step:03d}/...json
#
# Default: combnorm only (the run that just finished). Add the
# baseline / nognorm runs when those checkpoints exist:
#   sbatch --export=ALL,METHODS=multi_reward_combnorm,multi_reward_gdpo \
#          open_r1_imp/scripts/submit_trajectory_gdpo_ablation.sh
# ============================================================

#SBATCH --job-name=traj-gdpo-abl
#SBATCH --account=a0264
#SBATCH --qos=slowdown
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --partition=low
#SBATCH --output=logs/traj-gdpo-abl-%j.out
#SBATCH --error=logs/traj-gdpo-abl-%j.err
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -euo pipefail

PROJECT_DIR="${HOME}/projects/unsupervised-rl"
VENV_DIR="${PROJECT_DIR}/venv"

# Configurable (override via --export)
METHODS="${METHODS:-multi_reward_combnorm}"
TASKS="${TASKS:-gsm8k,math500,mmlu_pro}"
MIN_STEP="${MIN_STEP:-20}"
MAX_STEP="${MAX_STEP:-280}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_results_final/gdpo_ablation}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OVERWRITE="${OVERWRITE:-0}"
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-unsupervised-rl-eval}"
WANDB_GROUP="${WANDB_GROUP:-trajectory_gdpo_ablation_1.5b_v1}"

mkdir -p "${PROJECT_DIR}/logs"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Ensure math_verify available (paper-exact MATH500 scoring)
python -c "import math_verify, latex2sympy2_extended" 2>/dev/null || \
    pip install --quiet math-verify latex2sympy2-extended || true

# vLLM env (matches submit_trajectory_3b.sh)
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
echo "  GDPO-ablation trajectory eval (Qwen2.5-1.5B, 4 GPUs)"
echo "  Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "  Node:         $(hostname)"
echo "  Methods:      ${METHODS}"
echo "  Tasks:        ${TASKS}"
echo "  Step range:   [${MIN_STEP}, ${MAX_STEP}]"
echo "  Output root:  ${OUTPUT_ROOT}"
echo "  Max samples:  ${MAX_SAMPLES:-<full>}"
echo "  Wandb:        ${WANDB} (project=${WANDB_PROJECT}, group=${WANDB_GROUP})"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================================"

ARGS=(
    --num_gpus 4
    --methods "${METHODS}"
    --tasks "${TASKS}"
    --min_step "${MIN_STEP}"
    --max_step "${MAX_STEP}"
    --output_root "${OUTPUT_ROOT}"
    --enforce_eager
)
if [ -n "${MAX_SAMPLES}" ]; then
    ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [ "${OVERWRITE}" = "1" ]; then
    ARGS+=(--overwrite)
fi
if [ "${WANDB}" = "1" ]; then
    ARGS+=(--wandb --wandb_project "${WANDB_PROJECT}" --wandb_group "${WANDB_GROUP}")
fi

python -u evaluation/trajectory_gdpo_ablation_1.5b.py "${ARGS[@]}"

echo ""
echo "[$(date)] Eval complete. Per-method JSON counts under ${OUTPUT_ROOT}:"
for M in $(echo "${METHODS}" | tr ',' ' '); do
    N=$(find "${OUTPUT_ROOT}" -path "*/${M}_ckpt*/*.json" 2>/dev/null | wc -l)
    echo "  ${M}: ${N} JSONs"
done
echo "[$(date)] Done."
