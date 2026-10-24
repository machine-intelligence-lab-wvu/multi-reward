#!/usr/bin/env bash
# Train one recipe: starts a vLLM generation server on the first GPU and runs
# training (DeepSpeed ZeRO-3) on the remaining GPUs.
#
#   bash scripts/train.sh recipes/qwen2.5-1.5b/multi_reward_klcov.yaml
#   bash scripts/train.sh RECIPE --max_steps 20 --output_dir outputs/test   # extra args override the recipe
#
# Environment variables:
#   NUM_GPUS        GPUs to use (default: all visible). One serves vLLM, the rest train.
#   KEEP_BATCH      1 (default): rescale gradient accumulation so the effective batch
#                   equals the paper's (per_device x grad_accum x 3 trainer GPUs).
#   VLLM_PORT       vLLM server port (default 8000).
#   VLLM_GPU_UTIL   vLLM gpu_memory_utilization (default 0.85).
#   ACCEL_CONFIG    accelerate config (default accelerate_configs/zero3.yaml).
set -euo pipefail
cd "$(dirname "$0")/.."

RECIPE="${1:?usage: bash scripts/train.sh RECIPE.yaml [extra --key value ...]}"; shift
[ -f "$RECIPE" ] || { echo "Recipe not found: $RECIPE"; exit 1; }
PAPER_TRAIN_GPUS=3
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"
KEEP_BATCH="${KEEP_BATCH:-1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
ACCEL_CONFIG="${ACCEL_CONFIG:-accelerate_configs/zero3.yaml}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

read -r MODEL USE_VLLM PER_DEVICE GA OUTPUT_DIR < <("${PYTHON:-python}" - "$RECIPE" <<'EOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(c["model_name_or_path"], str(c.get("use_vllm", False)).lower(), c["per_device_train_batch_size"],
      c["gradient_accumulation_steps"], c["output_dir"])
EOF
)

if [ "$USE_VLLM" = "true" ]; then
    [ "$NUM_GPUS" -ge 2 ] || { echo "use_vllm needs >= 2 GPUs (1 for vLLM). Found $NUM_GPUS."; exit 1; }
    NUM_TRAIN=$((NUM_GPUS - 1)); TRAIN_GPUS=$(seq -s, 1 $((NUM_GPUS - 1)))
else
    NUM_TRAIN=$NUM_GPUS; TRAIN_GPUS=$(seq -s, 0 $((NUM_GPUS - 1)))
fi

EXTRA=()
if [ "$KEEP_BATCH" = "1" ] && [ "$NUM_TRAIN" -ne "$PAPER_TRAIN_GPUS" ]; then
    TOTAL=$((PER_DEVICE * GA * PAPER_TRAIN_GPUS))
    if [ $((TOTAL % (PER_DEVICE * NUM_TRAIN))) -ne 0 ]; then
        echo "Cannot keep the effective batch ($TOTAL completions) with $NUM_TRAIN trainer GPUs and"
        echo "per_device_train_batch_size=$PER_DEVICE. Pass --per_device_train_batch_size, or KEEP_BATCH=0."; exit 1
    fi
    GA=$((TOTAL / (PER_DEVICE * NUM_TRAIN)))
    EXTRA+=(--gradient_accumulation_steps "$GA")
    echo "Keeping effective batch $TOTAL: gradient_accumulation_steps -> $GA on $NUM_TRAIN trainer GPUs"
fi
# A --gradient_accumulation_steps passed by the user wins (and must reach accelerate too).
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    if [ "${ARGS[$i]}" = "--gradient_accumulation_steps" ]; then GA="${ARGS[$((i + 1))]}"; fi
    if [ "${ARGS[$i]}" = "--output_dir" ]; then OUTPUT_DIR="${ARGS[$((i + 1))]}"; fi
done

mkdir -p "$OUTPUT_DIR"
VLLM_PID=""
cleanup() { [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

if [ "$USE_VLLM" = "true" ]; then
    echo "Starting vLLM for $MODEL on GPU 0 (log: $OUTPUT_DIR/vllm.log)"
    CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model "$MODEL" --dtype bfloat16 --port "$VLLM_PORT" \
        --gpu_memory_utilization "$VLLM_GPU_UTIL" --max_model_len 4096 > "$OUTPUT_DIR/vllm.log" 2>&1 &
    VLLM_PID=$!
    for _ in $(seq 1 120); do
        curl -s "http://localhost:$VLLM_PORT/health/" >/dev/null 2>&1 && break
        kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM died:"; tail -40 "$OUTPUT_DIR/vllm.log"; exit 1; }
        sleep 5
    done
    curl -s "http://localhost:$VLLM_PORT/health/" >/dev/null || { echo "vLLM not ready after 600 s"; exit 1; }
    EXTRA+=(--vllm_server_port "$VLLM_PORT")
fi

echo "Training $RECIPE on GPUs $TRAIN_GPUS (grad_accum=$GA)"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" accelerate launch --config_file "$ACCEL_CONFIG" \
    --num_processes "$NUM_TRAIN" --gradient_accumulation_steps "$GA" \
    -m open_r1_imp.train --config "$RECIPE" ${EXTRA[@]+"${EXTRA[@]}"} "$@"
