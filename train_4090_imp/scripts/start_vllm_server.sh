#!/bin/bash
# ============================================================
# vLLM Server for Dual-GPU Training
# Runs on GPU 0, serves generation requests to training on GPU 1
# ============================================================
# Usage:
#   bash scripts/start_vllm_server.sh                                    # defaults
#   bash scripts/start_vllm_server.sh Qwen/Qwen2.5-3B                   # different model
#   bash scripts/start_vllm_server.sh Qwen/Qwen2.5-1.5B 8001            # different port
#   bash scripts/start_vllm_server.sh Qwen/Qwen2.5-1.5B 8000 128 4096   # custom lora rank + context
#
# Args: MODEL PORT MAX_LORA_RANK MAX_MODEL_LEN
# ============================================================
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B}"
PORT="${2:-8000}"
MAX_LORA_RANK="${3:-128}"
MAX_MODEL_LEN="${4:-3584}"

echo "Starting vLLM server"
echo "  Model:         ${MODEL}"
echo "  Port:          ${PORT}"
echo "  Max LoRA rank: ${MAX_LORA_RANK}"
echo "  Max model len: ${MAX_MODEL_LEN}"
echo "  LoRA:          enabled (dynamic loading)"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.85 \
    --enable-lora \
    --max-lora-rank "${MAX_LORA_RANK}" \
    --trust-remote-code \
    --enforce-eager \
    --seed 42
