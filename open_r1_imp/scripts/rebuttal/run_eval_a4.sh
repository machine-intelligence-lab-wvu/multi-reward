#!/bin/bash
# Evaluate the A-4 ablation (GRPO-GT +/- KL-Cov) on gsm8k + math500.
# RUN_PLAN.md lists GSM8K/MATH500 for these tiers; mmlu_pro is "where relevant"
# and would add ~1h/checkpoint for a comparison it does not settle.
# 2 GPUs, matching the load level that has not crashed the box.
set -uo pipefail
cd /home/WVU-AD/aa00307/projects/unsupervised-rl
mkdir -p logs

# Qwen2.5-1.5B base as the untrained reference (A-4 uses 1.5B, not Qwen3-1.7B,
# so the A-2 base numbers do not apply).
NUM_GPUS=2 PHASE=trajectory TASKS=gsm8k,math500 \
TRAJ_RUNS="a4_klcov:A4_grpo_gt_klcov_qwen25_15b a4_noklcov:A4_grpo_gt_noklcov_qwen25_15b" \
    bash open_r1_imp/scripts/rebuttal/run_eval_local48g.sh
echo "[$(date)] A-4 trajectory done (exit $?)"

echo "[$(date)] base model reference (Qwen2.5-1.5B)"
source venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}" HF_HOME="${HOME}/.cache/huggingface"
export CUDA_HOME="$PWD/.local-cuda-12.8"; export PATH="$CUDA_HOME/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 python evaluation/run_task.py \
    --checkpoint Qwen/Qwen2.5-1.5B --run_name qwen25_15b_base \
    --tasks gsm8k,math500 --output_dir eval-rebuttal-results --enforce_eager \
    > logs/eval/qwen25_15b_base.log 2>&1
echo "[$(date)] base done (exit $?)"
echo "[$(date)] A-4 EVAL COMPLETE"
