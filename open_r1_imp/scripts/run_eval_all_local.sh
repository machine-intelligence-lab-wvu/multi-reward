#!/bin/bash
# ============================================================
# Evaluate all 3 LoRA models — final saved model
# Uses GPU 2 (A100 80GB)
# ============================================================

set -euo pipefail

PROJECT_DIR="/home/WVU-AD/aa00307/projects/unsupervised-rl"
source "${PROJECT_DIR}/venv/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=2

BENCHMARKS="gsm8k math500"
COMMON_ARGS="--benchmarks ${BENCHMARKS} --save_records"

echo "============================================================"
echo "  Evaluation sweep — 3 models (final saved model)"
echo "============================================================"

# ── 1. Multi-Reward (no KL-Cov) ─────────────────────────────
echo ""
echo "[$(date)] 1/3  multi_reward_lora_local"
python evaluation/eval.py \
    --model_path outputs/multi_reward_lora_local \
    ${COMMON_ARGS}

# ── 2. Multi-Reward (KL-Cov 0.05) ───────────────────────────
echo ""
echo "[$(date)] 2/3  multi_reward_lora_local_klcov005"
python evaluation/eval.py \
    --model_path outputs/multi_reward_lora_local_klcov005 \
    ${COMMON_ARGS}

# ── 3. INTUITOR ──────────────────────────────────────────────
echo ""
echo "[$(date)] 3/3  intuitor_lora_local"
python evaluation/eval.py \
    --model_path outputs/intuitor_lora_local \
    ${COMMON_ARGS}

echo ""
echo "============================================================"
echo "[$(date)] All evaluations complete. Results in eval_results/"
echo "============================================================"
