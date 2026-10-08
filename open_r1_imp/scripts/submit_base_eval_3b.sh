#!/bin/bash
# ============================================================
# SLURM — Qwen2.5-3B-Base (untrained) evaluation
# Cluster: CSCS Clariden (GH200 GPUs)
#
# Evaluates the raw base model on 5 benchmarks in parallel:
#   GPU 0: gsm8k
#   GPU 1: math500
#   GPU 2: mmlu_pro          (largest — ~12k samples in 5-shot prompts)
#   GPU 3: aime2024,aime2025 (tiny — 30+30 problems)
#
# Results land in:
#   evaluation_results/base_qwen2.5_3b/qwen25_3b_base/qwen25_3b_base__{task}.json
#
# Usage:
#   sbatch open_r1_imp/scripts/submit_base_eval_3b.sh
# ============================================================

#SBATCH --job-name=base-eval-3b
#SBATCH --account=a0264
#SBATCH --qos=slowdown
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --partition=low
#SBATCH --output=logs/base-eval-3b-%j.out
#SBATCH --error=logs/base-eval-3b-%j.err
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -euo pipefail

PROJECT_DIR="${HOME}/projects/unsupervised-rl"
VENV_DIR="${PROJECT_DIR}/venv"

MODEL="${MODEL:-Qwen/Qwen2.5-3B}"
RUN_NAME="${RUN_NAME:-qwen25_3b_base}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation_results/base_qwen2.5_3b}"

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
export OMP_NUM_THREADS=9

echo "============================================================"
echo "  Qwen2.5-3B-Base evaluation (4 GPUs parallel)"
echo "  Job ID:  ${SLURM_JOB_ID:-N/A}"
echo "  Node:    $(hostname)"
echo "  Model:   ${MODEL}"
echo "  Out:     ${OUTPUT_DIR}/${RUN_NAME}/"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================================"

mkdir -p "${OUTPUT_DIR}/${RUN_NAME}"
LOGDIR="${OUTPUT_DIR}/_logs_base_eval"
mkdir -p "${LOGDIR}"

run_gpu() {
    local GPU="$1"; shift
    local TASKS="$1"; shift
    local LOGFILE="${LOGDIR}/gpu${GPU}.log"
    echo "[GPU ${GPU}] launching tasks=${TASKS}   -> ${LOGFILE}"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u evaluation/run_task.py \
        --checkpoint "${MODEL}" \
        --run_name   "${RUN_NAME}" \
        --tasks      "${TASKS}" \
        --output_dir "${OUTPUT_DIR}" \
        --enforce_eager \
        > "${LOGFILE}" 2>&1 &
}

run_gpu 0 "gsm8k"
run_gpu 1 "math500"
run_gpu 2 "mmlu_pro"
run_gpu 3 "aime2024,aime2025"

wait

echo ""
echo "[$(date)] All GPU workers finished. Per-task JSONs:"
for T in gsm8k math500 mmlu_pro aime2024 aime2025; do
    JSON="${OUTPUT_DIR}/${RUN_NAME}/${RUN_NAME}__${T}.json"
    if [ -f "${JSON}" ]; then
        ACC=$(python - <<PY
import json
d = json.load(open("${JSON}"))
print(f"{d.get('pass@1', d.get('accuracy', -1))*100:.2f}")
PY
)
        printf "  %-12s  pass@1 = %s%%   (%s)\n" "${T}" "${ACC}" "${JSON}"
    else
        printf "  %-12s  MISSING (%s)\n" "${T}" "${JSON}"
    fi
done

echo ""
echo "[$(date)] done."
