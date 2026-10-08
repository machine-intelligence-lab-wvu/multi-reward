#!/bin/bash
# ============================================================
# SLURM Evaluation — Best Checkpoints, 4 GPUs in parallel
# Cluster: CSCS Clariden (GH200 120GB GPUs)
#
# Runs 7 checkpoints × 5 tasks (GSM8K, MATH500, AIME24,
# AIME25, MMLU-Pro) = 35 jobs, distributed across 4 GPUs.
#
# Usage:
#   sbatch open_r1_imp/scripts/submit_eval_best_ckpts.sh
#
#   # Only a subset of runs:
#   sbatch open_r1_imp/scripts/submit_eval_best_ckpts.sh \
#     --export=ALL,ONLY=multi_reward_klcov05_ckpt300,intuitor_imp_ckpt160
#
#   # Quick smoke test (first 20 samples per task):
#   sbatch open_r1_imp/scripts/submit_eval_best_ckpts.sh \
#     --export=ALL,MAX_SAMPLES=20
# ============================================================

#SBATCH --job-name=eval-best
#SBATCH --account=a0264
#SBATCH --qos=slowdown
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --partition=low
#SBATCH --output=logs/eval-best-%j.out
#SBATCH --error=logs/eval-best-%j.err
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -euo pipefail

PROJECT_DIR="${HOME}/projects/unsupervised-rl"
VENV_DIR="${PROJECT_DIR}/venv"

# Configurable
OUTPUT_DIR="${OUTPUT_DIR:-evaluation_results/best_ckpts_v1}"
TASKS="${TASKS:-gsm8k,math500,aime2024,aime2025,mmlu_pro}"
ONLY="${ONLY:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OVERWRITE="${OVERWRITE:-0}"
WANDB="${WANDB:-1}"                                    # 1 = enable wandb
WANDB_PROJECT="${WANDB_PROJECT:-unsupervised-rl-eval}"
WANDB_GROUP="${WANDB_GROUP:-best_ckpts_v1}"
N_SAMPLES="${N_SAMPLES:-1}"                            # >1 enables pass@k / maj@k
TEMPERATURE="${TEMPERATURE:-0.0}"                      # 0.0 = greedy; use 0.6 for sampling
TOP_P="${TOP_P:-1.0}"
TRAJECTORY="${TRAJECTORY:-0}"                          # 1 = expand each method → every ckpt
MAX_STEP="${MAX_STEP:-340}"                            # cap checkpoints in trajectory mode
STEP_STRIDE="${STEP_STRIDE:-20}"                       # ckpt cadence in trajectory mode (e.g. 40 = every 40 steps)

mkdir -p "${PROJECT_DIR}/logs"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Ensure math_verify and latex2sympy2_extended are installed for paper-exact math scoring
python -c "import math_verify, latex2sympy2_extended" 2>/dev/null || \
    pip install --quiet math-verify latex2sympy2-extended || true

# Ensure transformers is in the venv (vllm 0.12 needs >=4.56) — without it,
# `import transformers` falls back to the older uenv copy which pins
# huggingface-hub<1.0 and clashes with the venv's hub 1.x.
python -c "import transformers, sys; from packaging import version; sys.exit(0 if version.parse(transformers.__version__) >= version.parse('4.56.0') and 'venv' in transformers.__file__ else 1)" 2>/dev/null || \
    pip install --quiet "transformers>=4.56.0,<5" || true

# vLLM / LoRA-safe env
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=18

export HF_TOKEN="${HF_TOKEN:-}"

# wandb env
export WANDB_DISABLE_WEAVE=true
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_GROUP="${WANDB_GROUP}"
echo "============================================================"
echo "  Best-checkpoints evaluation (4 GPUs parallel)"
echo "  Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "  Node:         $(hostname)"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  Tasks:        ${TASKS}"
echo "  Only runs:    ${ONLY:-<all>}"
echo "  Max samples:  ${MAX_SAMPLES:-<full>}"
echo "  n_samples:    ${N_SAMPLES}  temperature:${TEMPERATURE}  top_p:${TOP_P}"
echo "  Trajectory:   ${TRAJECTORY}  max_step:${MAX_STEP}  step_stride:${STEP_STRIDE}"
echo "  Wandb:        ${WANDB} (project=${WANDB_PROJECT}, group=${WANDB_GROUP})"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================================"

ARGS=(
    --num_gpus 4
    --output_dir "${OUTPUT_DIR}"
    --tasks "${TASKS}"
    --enforce_eager
)
if [ -n "${MAX_SAMPLES}" ]; then
    ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [ -n "${ONLY}" ]; then
    ARGS+=(--only "${ONLY}")
fi
if [ "${OVERWRITE}" = "1" ]; then
    ARGS+=(--overwrite)
fi
if [ "${WANDB}" = "1" ]; then
    ARGS+=(--wandb --wandb_project "${WANDB_PROJECT}" --wandb_group "${WANDB_GROUP}")
fi
if [ "${N_SAMPLES}" != "1" ]; then
    ARGS+=(--n_samples "${N_SAMPLES}")
fi
if [ "${TEMPERATURE}" != "0.0" ]; then
    ARGS+=(--temperature "${TEMPERATURE}")
fi
if [ "${TOP_P}" != "1.0" ]; then
    ARGS+=(--top_p "${TOP_P}")
fi
if [ "${TRAJECTORY}" = "1" ]; then
    ARGS+=(--trajectory --max_step "${MAX_STEP}" --step_stride "${STEP_STRIDE}")
fi

python -u evaluation/orchestrate.py "${ARGS[@]}"

echo ""
echo "=== Aggregating results ==="
python -u evaluation/summarize.py --output_dir "${OUTPUT_DIR}"

echo "[$(date)] evaluation done."
