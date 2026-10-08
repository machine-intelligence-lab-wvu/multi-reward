#!/bin/bash
# ============================================================
# SLURM — Trajectory Evaluation for Qwen2.5-3B runs
# Cluster: CSCS Clariden (GH200 GPUs)
#
# Evaluates every 20-step checkpoint of 4 methods (GRPO-GT,
# Multi-reward no-KLCov, Multi-reward + KLCov β=0.05, INTUITOR)
# on 5 benchmarks (GSM8K, MATH500, MMLU-Pro, AIME24, AIME25).
# ~68 (method, ckpt) jobs × 5 tasks = 340 evaluations, distributed
# across 4 GPUs.
#
# Usage:
#   sbatch open_r1_imp/scripts/submit_trajectory_3b.sh
#
#   # Subset of methods (comma-separated):
#   sbatch --export=ALL,METHODS=grpo_gt,intuitor \
#          open_r1_imp/scripts/submit_trajectory_3b.sh
#
#   # Shorter step range:
#   sbatch --export=ALL,MAX_STEP=200 \
#          open_r1_imp/scripts/submit_trajectory_3b.sh
# ============================================================

#SBATCH --job-name=traj-3b
#SBATCH --account=a0264
#SBATCH --qos=slowdown
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --partition=low
#SBATCH --output=logs/traj-3b-%j.out
#SBATCH --error=logs/traj-3b-%j.err
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -euo pipefail

PROJECT_DIR="${HOME}/projects/unsupervised-rl"
VENV_DIR="${PROJECT_DIR}/venv"

# Configurable (override via --export)
METHODS="${METHODS:-grpo_gt,multi_reward,multi_reward_klcov05,intuitor}"
TASKS="${TASKS:-gsm8k,math500,mmlu_pro,aime2024,aime2025}"
MIN_STEP="${MIN_STEP:-20}"
MAX_STEP="${MAX_STEP:-340}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OVERWRITE="${OVERWRITE:-0}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-unsupervised-rl-eval}"
WANDB_GROUP="${WANDB_GROUP:-trajectory_3b_v1}"

mkdir -p "${PROJECT_DIR}/logs"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Ensure math_verify for paper-exact scoring
python -c "import math_verify, latex2sympy2_extended" 2>/dev/null || \
    pip install --quiet math-verify latex2sympy2-extended || true

# vLLM env
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
echo "  Trajectory evaluation (Qwen2.5-3B, 4 GPUs parallel)"
echo "  Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "  Node:         $(hostname)"
echo "  Methods:      ${METHODS}"
echo "  Tasks:        ${TASKS}"
echo "  Step range:   [${MIN_STEP}, ${MAX_STEP}]"
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
    --enforce_eager
)
if [ -n "${MAX_SAMPLES}" ]; then
    ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [ "${OVERWRITE}" = "1" ]; then
    ARGS+=(--overwrite)
fi

python -u evaluation/trajectory_3b.py "${ARGS[@]}"

echo ""
echo "[$(date)] Eval complete — per-task results:"
for T in gsm8k math500 mmlu_pro aime2024 aime2025; do
    D="evaluation_results/accuracy_trajectory_${T}_qwen3b"
    if [ -d "$D" ]; then
        N=$(find "$D" -name '*.json' -not -path '*/_logs*/*' | wc -l)
        echo "  ${D}  (${N} JSONs)"
    fi
done

# ════════════════════════════════════════════════════════════════
# POST-PROCESS: plots + wandb trajectory runs per dataset
# run names: trajectory_{task}_3b_{method}
# ════════════════════════════════════════════════════════════════
echo ""
echo "[$(date)] Generating plots + uploading wandb trajectory runs..."
for T in gsm8k math500 mmlu_pro aime2024 aime2025; do
    D="evaluation_results/accuracy_trajectory_${T}_qwen3b"
    [ -d "$D" ] || continue

    # Plot
    python -u evaluation/plot_trajectory.py \
        --output_dir "$D" \
        --figure_dir paper/figures \
        --task "$T" || echo "  plot_trajectory failed for $T"

    # Wandb upload (only if WANDB=1)
    if [ "${WANDB}" = "1" ]; then
        python -u evaluation/log_trajectory_to_wandb.py \
            --output_dir "$D" \
            --task "$T" \
            --wandb_project "${WANDB_PROJECT}" \
            --wandb_group "${WANDB_GROUP}" \
            --run_name_template "trajectory_{task}_3b_{method}" \
            --figure_png "paper/figures/${T}_pass1_trajectory.png" || \
            echo "  wandb upload failed for $T"
    fi
done

echo "[$(date)] All done."
