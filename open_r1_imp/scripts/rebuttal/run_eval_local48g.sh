#!/bin/bash
# ============================================================
# Evaluate rebuttal checkpoints locally on 4 x RTX PRO 5000 48GB.
#
# Local analogue of submit_eval_best_ckpts.sh / orchestrate.py: same atomic unit
# (evaluation/run_task.py), same hard-coded eval settings (greedy, enforce_eager,
# bf16, max_model_len 16384) so numbers are comparable to the cluster results.
# orchestrate.py's JOBS list is hard-coded to /capstor paths, so we drive
# run_task.py directly instead of editing it.
#
#   bash open_r1_imp/scripts/rebuttal/run_eval_local48g.sh            # headline phase
#   PHASE=trajectory bash .../run_eval_local48g.sh                    # all checkpoints
#
# run_task.py loads vLLM ONCE per model and sweeps all tasks, so the natural
# unit of parallelism is one model per GPU. Idempotent: an existing
# {OUT}/{run_name}/{run_name}__{task}.json is skipped unless OVERWRITE=1.
# ============================================================
set -uo pipefail

PROJECT_DIR="/home/WVU-AD/aa00307/projects/unsupervised-rl"
VENV_DIR="${PROJECT_DIR}/venv"
cd "${PROJECT_DIR}"

OUT="${OUT:-eval-rebuttal-results}"
TASKS="${TASKS:-gsm8k,math500,mmlu_pro}"
PHASE="${PHASE:-headline}"
NUM_GPUS="${NUM_GPUS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OVERWRITE="${OVERWRITE:-0}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export CUDA_HOME="${CUDA_HOME:-${PROJECT_DIR}/.local-cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Datasets are pre-staged; keep workers off the network so 4 of them cannot race.
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

mkdir -p "${OUT}" logs/eval

RUNS_DIR="${PROJECT_DIR}/outputs/rebuttal"

# ─── Build the job list: "run_name|checkpoint_path" ─────────
JOBS=()
if [ "${PHASE}" = "headline" ]; then
    # Base model + each method's final checkpoint. 4 models / 4 GPUs.
    JOBS+=("qwen3_17b_base|Qwen/Qwen3-1.7B-Base")
    for r in full intuitor grpo_gt; do
        JOBS+=("a2_${r}_ckpt200|${RUNS_DIR}/A2_qwen3_17b_${r}/checkpoint-200")
    done
else
    # Trajectory: every checkpoint of every method (for best-ckpt selection
    # and accuracy-vs-step plots). Pair with TASKS=gsm8k,math500 — mmlu_pro
    # over 10 checkpoints x 3 runs is 360k generations and rarely worth it.
    #
    # TRAJ_RUNS is a space-separated list of "<short_name>:<output_dir_name>"
    # under outputs/rebuttal. Defaults to the A-2 suite.
    # NB: short names must keep the a2_ prefix or existing A-2 result JSONs stop
    # matching and the whole sweep re-runs instead of being skipped.
    TRAJ_RUNS="${TRAJ_RUNS:-a2_full:A2_qwen3_17b_full a2_intuitor:A2_qwen3_17b_intuitor a2_grpo_gt:A2_qwen3_17b_grpo_gt}"
    for spec in ${TRAJ_RUNS}; do
        short="${spec%%:*}"; dirname="${spec#*:}"
        for step in 20 40 60 80 100 120 140 160 180 200; do
            d="${RUNS_DIR}/${dirname}/checkpoint-${step}"
            [ -d "$d" ] && JOBS+=("${short}_ckpt${step}|${d}")
        done
    done
fi

EXTRA=""
[ -n "${MAX_SAMPLES}" ] && EXTRA="${EXTRA} --max_samples ${MAX_SAMPLES}"
# SAMPLE_SEED makes MAX_SAMPLES a seeded RANDOM subset. Mandatory whenever
# subsampling mmlu_pro: its test rows are ordered by category, so the first 2000
# are 55% law / 39.5% business / 5.5% psychology and cover 3 of 14 categories.
# A fixed seed gives every model the identical subset.
[ -n "${SAMPLE_SEED:-}" ] && EXTRA="${EXTRA} --sample_seed ${SAMPLE_SEED}"
[ "${OVERWRITE}" = "1" ] && EXTRA="${EXTRA} --overwrite"

echo "============================================================"
echo "  Local eval — phase=${PHASE}"
echo "  Tasks:   ${TASKS}"
echo "  Output:  ${OUT}/<run_name>/<run_name>__<task>.json"
echo "  Jobs:    ${#JOBS[@]} across ${NUM_GPUS} GPUs"
echo "============================================================"
for j in "${JOBS[@]}"; do echo "   - ${j%%|*}  <-  ${j##*|}"; done
echo "============================================================"

# ─── One worker per GPU, striding through the job list ──────
worker() {
    local gpu=$1
    local i=$gpu
    while [ $i -lt ${#JOBS[@]} ]; do
        local job="${JOBS[$i]}"
        local name="${job%%|*}"
        local ckpt="${job##*|}"
        echo "[$(date +%H:%M:%S)] gpu${gpu} START ${name}"
        CUDA_VISIBLE_DEVICES=${gpu} python evaluation/run_task.py \
            --checkpoint "${ckpt}" \
            --run_name "${name}" \
            --tasks "${TASKS}" \
            --output_dir "${OUT}" \
            --enforce_eager \
            ${EXTRA} > "logs/eval/${name}.log" 2>&1
        local rc=$?
        if [ $rc -eq 0 ]; then
            echo "[$(date +%H:%M:%S)] gpu${gpu} DONE  ${name}"
        else
            echo "[$(date +%H:%M:%S)] gpu${gpu} FAIL  ${name} (exit ${rc}) — logs/eval/${name}.log"
            tail -15 "logs/eval/${name}.log" | sed 's/^/      /'
        fi
        i=$(( i + NUM_GPUS ))
    done
}

for g in $(seq 0 $(( NUM_GPUS - 1 ))); do worker "$g" & done
wait

echo "============================================================"
echo "[$(date)] All eval jobs finished. Results:"
find "${OUT}" -name "*.json" | sort | sed 's/^/   /'
echo "============================================================"
