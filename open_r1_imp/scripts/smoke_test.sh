#!/bin/bash
# ============================================================
# Smoke Test — Run in an interactive salloc session
#
# Quick test of the full training pipeline (vLLM + accelerate)
# with minimal steps to catch errors fast.
#
# Usage:
#   # First get an interactive session:
#   salloc --account=a0264 --time=00:30:00 --nodes=1 --gpus-per-node=4 \
#          --cpus-per-task=36 --partition=normal \
#          --uenv=pytorch/v2.9.1:v2 --view=default
#
#   # Then run:
#   bash smoke_test.sh
#   bash smoke_test.sh --thinking        # test thinking mode
#   bash smoke_test.sh --mode lora       # test LoRA mode
#   bash smoke_test.sh --num_train_gpus 1 --no_vllm  # single GPU, no vLLM
# ============================================================
set -euo pipefail

PROJECT_DIR="${HOME}/projects/unsupervised-rl"
VENV_DIR="${PROJECT_DIR}/venv"

# ─── Defaults ────────────────────────────────────────────────
MODE="full"
NUM_TRAIN_GPUS=3
THINKING=false
NO_VLLM=false
MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --num_train_gpus) NUM_TRAIN_GPUS="$2"; shift 2 ;;
        --thinking) THINKING=true; shift ;;
        --no_vllm) NO_VLLM=true; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ─── Activate env ────────────────────────────────────────────
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export VLLM_USE_V1=0
export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ─── Select config ───────────────────────────────────────────
if [ "$THINKING" = true ]; then
    CONFIG="open_r1_imp/recipes/config_math_full_clariden.yaml"
elif [ "$MODE" = "full" ]; then
    CONFIG="open_r1_imp/recipes/config_math_full_clariden.yaml"
else
    CONFIG="open_r1_imp/recipes/config_math_lora.yaml"
fi

if [ "$MODE" = "full" ]; then
    ACCEL_CONFIG="open_r1_imp/accelerate_configs/zero3_smoke.yaml"
    EXTRA_ARGS="--use_peft false"
else
    ACCEL_CONFIG="open_r1_imp/accelerate_configs/zero2.yaml"
    EXTRA_ARGS=""
fi

# ─── Smoke test overrides (small & fast) ─────────────────────
EXTRA_ARGS="${EXTRA_ARGS} --max_steps 3 --save_steps 999 --logging_steps 1"
EXTRA_ARGS="${EXTRA_ARGS} --per_device_train_batch_size 7 --gradient_accumulation_steps 1"
EXTRA_ARGS="${EXTRA_ARGS} --num_generations 7"
EXTRA_ARGS="${EXTRA_ARGS} --output_dir /tmp/smoke_test_grpo"
EXTRA_ARGS="${EXTRA_ARGS} --overwrite_output_dir true"

if [ -n "$MODEL" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --model_name_or_path ${MODEL}"
fi
if [ "$THINKING" = true ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --thinking true"
fi

# ─── Build GPU lists ─────────────────────────────────────────
VLLM_GPU=0
TRAIN_GPUS=""
for i in $(seq 1 $NUM_TRAIN_GPUS); do
    [ -n "$TRAIN_GPUS" ] && TRAIN_GPUS="${TRAIN_GPUS},$i" || TRAIN_GPUS="$i"
done

echo "============================================================"
echo "  SMOKE TEST"
echo "  Mode: ${MODE} | Thinking: ${THINKING}"
echo "  Config: ${CONFIG}"
echo "  vLLM GPU: ${VLLM_GPU} | Training GPUs: ${TRAIN_GPUS}"
echo "  Extra: ${EXTRA_ARGS}"
echo "============================================================"

# ─── Start vLLM ──────────────────────────────────────────────
if [ "$NO_VLLM" = false ]; then
    VLLM_MODEL=$(grep "model_name_or_path" "$CONFIG" | head -1 | awk '{print $2}')
    [ -n "$MODEL" ] && VLLM_MODEL="$MODEL"

    echo "[$(date)] Starting vLLM for ${VLLM_MODEL}..."
    CUDA_VISIBLE_DEVICES=${VLLM_GPU} trl vllm-serve \
        --model "${VLLM_MODEL}" \
        --dtype bfloat16 \
        --gpu_memory_utilization 0.8 &
    VLLM_PID=$!

    VLLM_READY=false
    for i in $(seq 1 60); do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "[$(date)] vLLM ready!"
            VLLM_READY=true
            break
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "ERROR: vLLM died!"
            exit 1
        fi
        sleep 5
    done
    if [ "$VLLM_READY" = false ]; then
        echo "ERROR: vLLM timeout"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
fi

# ─── Run training (3 steps) ──────────────────────────────────
echo "[$(date)] Launching training..."
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_TRAIN_GPUS}" \
    open_r1_imp/multi_reward_train.py \
    --config "${CONFIG}" \
    ${EXTRA_ARGS}

TRAIN_EXIT=$?

# ─── Cleanup ─────────────────────────────────────────────────
if [ "$NO_VLLM" = false ] && [ -n "${VLLM_PID:-}" ]; then
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
fi

if [ $TRAIN_EXIT -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "  SMOKE TEST PASSED"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "  SMOKE TEST FAILED (exit code: ${TRAIN_EXIT})"
    echo "============================================================"
fi

exit $TRAIN_EXIT
