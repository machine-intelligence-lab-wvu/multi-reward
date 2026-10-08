#!/bin/bash
# ============================================================
# Smoke Test - DollySods, SINGLE 24GB GPU (interactive salloc)
#
# Fallback variant of smoke_test_dollysods.sh for when the partition
# only grants 1 GPU (e.g., inter_a30). Differences from the multi-GPU
# version:
#   - No vLLM server (the trainer does in-process generation via HF)
#   - Single GPU does both generation and training
#   - DeepSpeed ZeRO-3 with CPU-offloaded optimizer state (zero3_smoke.yaml)
#   - Tiny effective batch (per_device=2, num_generations=2, max_completion=512)
#
# Validates the same code path as the 7B run except for the vLLM bridge.
# Once you have access to a multi-GPU partition, use the regular
# smoke_test_dollysods.sh instead — that exercises vLLM + DeepSpeed.
# ============================================================
set -euo pipefail

PROJECT_DIR="${HOME}/unsupervised-rl"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-urlenv}"

module load cuda/12.3 || true
export CUDA_HOME="${CUDA_HOME:-/shared/software/nvidia/cuda/12.3}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
  || source "${HOME}/conda/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

CONFIG="open_r1_imp/recipes/final/config_multi_reward_klcov_0.0005_qwen25_1.5b.yaml"
ACCEL_CONFIG="open_r1_imp/accelerate_configs/zero3_smoke.yaml"

# Always start clean — a previous run's partial checkpoint dir would otherwise
# trigger HF Trainer's auto-resume and crash on missing trainer_state.json.
rm -rf "/tmp/${USER}/smoke_mrew_klcov0005_1gpu"

# Effective batch = per_device(2) x grad_accum(1) x 1 GPU = 2
# max_completion_length=512 lowers activation memory since vLLM is gone
EXTRA_ARGS="--use_peft false"
EXTRA_ARGS="${EXTRA_ARGS} --use_vllm false"
EXTRA_ARGS="${EXTRA_ARGS} --max_steps 3"
EXTRA_ARGS="${EXTRA_ARGS} --save_strategy no"
EXTRA_ARGS="${EXTRA_ARGS} --logging_steps 1"
EXTRA_ARGS="${EXTRA_ARGS} --per_device_train_batch_size 2"
EXTRA_ARGS="${EXTRA_ARGS} --gradient_accumulation_steps 1"
EXTRA_ARGS="${EXTRA_ARGS} --num_generations 2"
EXTRA_ARGS="${EXTRA_ARGS} --max_completion_length 512"
EXTRA_ARGS="${EXTRA_ARGS} --max_prompt_length 256"
EXTRA_ARGS="${EXTRA_ARGS} --output_dir /tmp/${USER}/smoke_mrew_klcov0005_1gpu"
EXTRA_ARGS="${EXTRA_ARGS} --overwrite_output_dir true"
EXTRA_ARGS="${EXTRA_ARGS} --attn_implementation sdpa"

echo "============================================================"
echo "  SMOKE TEST - DollySods 1 GPU (no vLLM, CPU-offloaded AdamW)"
echo "  Node:      $(hostname)"
echo "  GPU:       ${CUDA_VISIBLE_DEVICES:-0}"
echo "  Config:    ${CONFIG}"
echo "  Overrides: ${EXTRA_ARGS}"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "============================================================"

accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes 1 \
    open_r1_imp/multi_reward_train.py \
    --config "${CONFIG}" \
    ${EXTRA_ARGS}

TRAIN_EXIT=$?

if [ ${TRAIN_EXIT} -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "  SMOKE TEST PASSED (1-GPU mode)"
    echo "  Note: this skipped the vLLM bridge. Re-run smoke_test_dollysods.sh"
    echo "  when you have a multi-GPU partition to exercise that path too,"
    echo "  before submitting the 7B sbatch."
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "  SMOKE TEST FAILED (exit ${TRAIN_EXIT})"
    echo "============================================================"
fi

exit ${TRAIN_EXIT}
