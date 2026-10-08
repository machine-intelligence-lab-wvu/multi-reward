#!/bin/bash
# ============================================================
# Multi-Reward GRPO — Open-R1 Integration
# Distributed training with vLLM server + DeepSpeed
# ============================================================
#
# Usage:
#   # LoRA training (3 training GPUs + 1 vLLM):
#   bash open_r1_imp/scripts/run_multi_reward.sh --mode lora --num_train_gpus 3
#
#   # Full fine-tuning:
#   bash open_r1_imp/scripts/run_multi_reward.sh --mode full --num_train_gpus 3
#
#   # Thinking mode (Qwen3):
#   bash open_r1_imp/scripts/run_multi_reward.sh --mode lora --thinking --num_train_gpus 3
#
#   # Single GPU (no vLLM server, for testing):
#   bash open_r1_imp/scripts/run_multi_reward.sh --mode lora --num_train_gpus 1 --no_vllm
#
#   # Override model or config:
#   bash open_r1_imp/scripts/run_multi_reward.sh --mode lora --model Qwen/Qwen2.5-3B
#
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── Parse arguments ──────────────────────────────────
MODE="lora"           # lora | full
NUM_TRAIN_GPUS=3      # GPUs for training (vLLM gets GPU 0)
THINKING=false
NO_VLLM=false
MODEL=""
EXTRA_ARGS=""
CONFIG_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --num_train_gpus) NUM_TRAIN_GPUS="$2"; shift 2 ;;
        --thinking) THINKING=true; shift ;;
        --no_vllm) NO_VLLM=true; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --config) CONFIG_OVERRIDE="$2"; shift 2 ;;
        *) EXTRA_ARGS="${EXTRA_ARGS} $1"; shift ;;
    esac
done

# ── Select config and accelerate based on mode ──────
if [ -n "$CONFIG_OVERRIDE" ]; then
    CONFIG="$CONFIG_OVERRIDE"
elif [ "$THINKING" = true ]; then
    CONFIG="open_r1_imp/recipes/config_math_thinking.yaml"
elif [ "$MODE" = "full" ]; then
    CONFIG="open_r1_imp/recipes/config_math_full.yaml"
else
    CONFIG="open_r1_imp/recipes/config_math_lora.yaml"
fi

if [ "$MODE" = "full" ]; then
    ACCEL_CONFIG="open_r1_imp/accelerate_configs/zero3.yaml"
    # Force full fine-tuning regardless of what the config YAML says
    EXTRA_ARGS="${EXTRA_ARGS} --use_peft false"
else
    ACCEL_CONFIG="open_r1_imp/accelerate_configs/zero2.yaml"
fi

# ── Build GPU device lists ──────────────────────────
# GPU 0 = vLLM server, GPUs 1..N = training
VLLM_GPU=0
TRAIN_GPUS=""
for i in $(seq 1 $NUM_TRAIN_GPUS); do
    if [ -n "$TRAIN_GPUS" ]; then
        TRAIN_GPUS="${TRAIN_GPUS},$i"
    else
        TRAIN_GPUS="$i"
    fi
done

# Model override
if [ -n "$MODEL" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --model_name_or_path ${MODEL}"
fi

# Thinking mode override
if [ "$THINKING" = true ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --thinking true"
fi

echo "============================================================"
echo "  Multi-Reward GRPO — Open-R1"
echo "  Mode: ${MODE} | Thinking: ${THINKING}"
echo "  Config: ${CONFIG}"
echo "  Accelerate: ${ACCEL_CONFIG}"
echo "  vLLM GPU: ${VLLM_GPU} | Training GPUs: ${TRAIN_GPUS} (${NUM_TRAIN_GPUS})"
echo "============================================================"

# ── Start vLLM server ──────────────────────────────
if [ "$NO_VLLM" = false ]; then
    # Extract model name from config
    VLLM_MODEL=$(grep "model_name_or_path" "$CONFIG" | head -1 | awk '{print $2}')
    if [ -n "$MODEL" ]; then
        VLLM_MODEL="$MODEL"
    fi

    echo "Starting vLLM server on GPU ${VLLM_GPU} for ${VLLM_MODEL}..."
    CUDA_VISIBLE_DEVICES=${VLLM_GPU} trl vllm-serve \
        --model "${VLLM_MODEL}" \
        --dtype bfloat16 \
        --gpu_memory_utilization 0.8 &
    VLLM_PID=$!

    # Wait for vLLM to be ready
    echo "Waiting for vLLM server to start..."
    for i in $(seq 1 60); do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "vLLM server ready!"
            break
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "ERROR: vLLM server died!"
            exit 1
        fi
        sleep 5
    done
fi

# ── Launch training ────────────────────────────────
echo "Launching training on GPUs: ${TRAIN_GPUS}..."
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_TRAIN_GPUS}" \
    open_r1_imp/multi_reward_train.py \
    --config "${CONFIG}" \
    ${EXTRA_ARGS}

TRAIN_EXIT=$?

# ── Cleanup ────────────────────────────────────────
if [ "$NO_VLLM" = false ] && [ -n "${VLLM_PID:-}" ]; then
    echo "Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
fi

exit $TRAIN_EXIT
