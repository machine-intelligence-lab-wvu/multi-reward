#!/usr/bin/env bash
# End-to-end check of the training code on CPU (no GPU, no vLLM): runs two optimizer
# steps of each method with a tiny random model, using the real paper recipes with
# only the hardware/size settings overridden. Takes a few minutes.
#
#   bash scripts/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
OUT="${SMOKE_DIR:-${TMPDIR:-/tmp}/multi_reward_smoke}"
PY="${PYTHON:-python}"
mkdir -p "$OUT"

RECIPES=(
    recipes/qwen2.5-1.5b/multi_reward_klcov.yaml
    recipes/qwen2.5-1.5b/multi_reward.yaml
    recipes/qwen2.5-1.5b/gdpo_ablation_combine_then_norm.yaml
    recipes/qwen2.5-1.5b/intuitor.yaml
    recipes/qwen2.5-1.5b/intuitor_klcov.yaml
    recipes/qwen2.5-1.5b/grpo_gt.yaml
    recipes/qwen3-1.7b/multi_reward_klcov.yaml
)
for recipe in "${RECIPES[@]}"; do
    name="$(basename "$(dirname "$recipe")")_$(basename "$recipe" .yaml)"
    echo "== $recipe"
    "$PY" -m open_r1_imp.train --config "$recipe" \
        --model_name_or_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 \
        --dtype float32 --attn_implementation eager --bf16 false --use_cpu true \
        --use_vllm false --gradient_checkpointing false \
        --per_device_train_batch_size 4 --num_generations 4 --gradient_accumulation_steps 2 \
        --max_completion_length 16 --max_steps 2 --save_strategy no \
        --output_dir "$OUT/$name" > "$OUT/$name.log" 2>&1 \
        || { echo "FAILED: $recipe (log: $OUT/$name.log)"; tail -20 "$OUT/$name.log"; exit 1; }
done
echo "All smoke runs passed. Logs: $OUT/*.log"
