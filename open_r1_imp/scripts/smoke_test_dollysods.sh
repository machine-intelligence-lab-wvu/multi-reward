#!/bin/bash
# ============================================================
# Smoke Test - DollySods, 2x 24GB GPUs (interactive salloc)
#
# Purpose: validate the full multi-reward + KL-Cov pipeline end-to-end
# on a small node BEFORE queueing the 7B run on dscog035. Uses Qwen2.5-1.5B
# because 7B does not fit on 24GB cards (vLLM alone needs ~14GB params + KV).
#
# What this test exercises (same code path as the 7B run):
#   - vLLM serve + trl GRPO server bridge
#   - MultiRewardTrainer (cluster_voting + boxed_format reward funcs)
#   - KL-Cov regularizer (beta_cov=0.0005)
#   - DeepSpeed ZeRO-3 (single train GPU, sharding trivial but plumbing tested)
#   - dataset loading, tokenizer, save/checkpoint code path
#
# Layout (2 GPUs total):
#   GPU 0 : vLLM serve  (Qwen2.5-1.5B, ~3GB params + KV)
#   GPU 1 : training   (Zero3, per_device=4, num_generations=4, max_steps=3)
#
# Requires: an interactive salloc on a 2-GPU 24GB node (dscog005-030).
# See README at the bottom of this file for the salloc command + workflow.
# ============================================================
set -euo pipefail

PROJECT_DIR="${HOME}/unsupervised-rl"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-urlenv}"

# --- Conda + CUDA toolchain --------------------------------
module load cuda/12.3 || true
export CUDA_HOME="${CUDA_HOME:-/shared/software/nvidia/cuda/12.3}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
  || source "${HOME}/conda/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# --- Smoke-test env (no wandb, no internet, throwaway output) ----
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export VLLM_USE_V1=0
export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

# Use cached HF model (must be pre-downloaded on dsis001 login node).
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# --- Config: 1.5B + multi-reward + KL-Cov 0.0005 ----------
CONFIG="open_r1_imp/recipes/final/config_multi_reward_klcov_0.0005_qwen25_1.5b.yaml"
ACCEL_CONFIG="open_r1_imp/accelerate_configs/zero3_smoke.yaml"

# --- Smoke overrides (tiny + fast) -------------------------
# Effective batch = per_device(4) x grad_accum(1) x 1 train GPU = 4 (= num_generations)
EXTRA_ARGS="--use_peft false"
EXTRA_ARGS="${EXTRA_ARGS} --max_steps 3"
EXTRA_ARGS="${EXTRA_ARGS} --save_steps 999"
EXTRA_ARGS="${EXTRA_ARGS} --logging_steps 1"
EXTRA_ARGS="${EXTRA_ARGS} --per_device_train_batch_size 2"
EXTRA_ARGS="${EXTRA_ARGS} --gradient_accumulation_steps 1"
EXTRA_ARGS="${EXTRA_ARGS} --num_generations 2"
EXTRA_ARGS="${EXTRA_ARGS} --max_completion_length 1024"
EXTRA_ARGS="${EXTRA_ARGS} --output_dir /tmp/${USER}/smoke_mrew_klcov0005"
EXTRA_ARGS="${EXTRA_ARGS} --overwrite_output_dir true"
EXTRA_ARGS="${EXTRA_ARGS} --attn_implementation sdpa"  # smoke uses SDPA; flash-attn not installed

VLLM_GPU=0
TRAIN_GPUS=1,2,3
NUM_TRAIN_GPUS=3

echo "============================================================"
echo "  SMOKE TEST - DollySods 2x 24GB"
echo "  Node:      $(hostname)"
echo "  Config:    ${CONFIG}"
echo "  vLLM GPU:  ${VLLM_GPU} | Training GPUs: ${TRAIN_GPUS}"
echo "  Overrides: ${EXTRA_ARGS}"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "============================================================"

# --- Start vLLM on GPU 0 -----------------------------------
VLLM_MODEL=$(grep "model_name_or_path" "${CONFIG}" | head -1 | awk '{print $2}')
echo "[$(date)] Starting vLLM on GPU ${VLLM_GPU} for ${VLLM_MODEL}..."
CUDA_VISIBLE_DEVICES=${VLLM_GPU} trl vllm-serve \
    --model "${VLLM_MODEL}" \
    --dtype bfloat16 \
    --gpu_memory_utilization 0.7 \
    --max_model_len 2048 &
VLLM_PID=$!

VLLM_READY=false
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "[$(date)] vLLM ready!"
        VLLM_READY=true
        break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM died during startup."
        exit 1
    fi
    sleep 5
done
if [ "${VLLM_READY}" = false ]; then
    echo "ERROR: vLLM timeout (300s)."
    kill ${VLLM_PID} 2>/dev/null || true
    exit 1
fi

# --- Run training (3 steps) --------------------------------
echo "[$(date)] Launching trainer on GPU ${TRAIN_GPUS}..."
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_TRAIN_GPUS}" \
    open_r1_imp/multi_reward_train.py \
    --config "${CONFIG}" \
    ${EXTRA_ARGS}

TRAIN_EXIT=$?

kill ${VLLM_PID} 2>/dev/null || true
wait ${VLLM_PID} 2>/dev/null || true

if [ ${TRAIN_EXIT} -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "  SMOKE TEST PASSED"
    echo "  -> Pipeline is healthy. Safe to submit the 7B sbatch:"
    echo "     sbatch open_r1_imp/scripts/submit_dollysods_7b.sh"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "  SMOKE TEST FAILED (exit ${TRAIN_EXIT})"
    echo "  -> Investigate before submitting the 7B run."
    echo "============================================================"
fi

exit ${TRAIN_EXIT}
