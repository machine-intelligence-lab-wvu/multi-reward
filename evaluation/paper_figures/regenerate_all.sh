#!/usr/bin/env bash
# Regenerate every data figure in the ICLR paper with the CIRCA nomenclature.
#
# Inputs  : eval_results_final/ (symlink -> /ocean) and eval-rebuttal-results/
# Outputs : multi-reward-iclr2027/figures/circa/<subfolder>/
#
# The plot scripts under evaluation/paper_figures/ are repo-local copies of the
# generators that used to live beside the results on /ocean; only the display
# labels were changed (Multi-Reward -> CIRCA, Intuitor -> INTUITOR,
# cluster-only -> "Cluster voting only (EMPO)"). Colors, data wiring and layout
# are untouched, so each figure is identical to the published one but relabelled.
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$PWD"
FIG="$ROOT/multi-reward-iclr2027/figures/circa"
PF="$ROOT/evaluation/paper_figures"
PY="${PY:-python}"

mkdir -p "$FIG"/{main,ablation,7b,qwen3,grpo_klcov}

echo "== Fig 1: 1.5B + 3B combined trajectory panel =="
$PY "$PF/plot_trajectory_panel.py" --combined \
    --base    "$ROOT/eval_results_final/trajectory_postfix_1.5b" \
    --base-3b "$ROOT/eval_results_final/trajectory_postfix_3b" \
    --model-label Qwen2.5-1.5B --model-label-3b Qwen2.5-3B \
    --exclude intuitor_klcov,intuitor_klcov_3b \
    --out "$FIG/main/trajectory_panel_combined.pdf"

echo "== per-scale panels (appendix) =="
$PY "$PF/plot_trajectory_panel.py" --panel \
    --base "$ROOT/eval_results_final/trajectory_postfix_1.5b" \
    --model-label Qwen2.5-1.5B --exclude intuitor_klcov \
    --out "$FIG/main/trajectory_panel_1.5b.pdf"
$PY "$PF/plot_trajectory_panel.py" --panel \
    --base "$ROOT/eval_results_final/trajectory_postfix_3b" \
    --model-label Qwen2.5-3B --exclude intuitor_klcov_3b \
    --out "$FIG/main/trajectory_panel_3b.pdf"

echo "== Fig 6: single-reward vs CIRCA (no KL-Cov) =="
$PY "$PF/plot_single_multi_reward.py" --paper-panel \
    --base "$ROOT/eval_results_final/single_multi_reward_ablation" \
    --panel-tasks gsm8k,math500 \
    --paper-panel-out "$FIG/ablation/single_multi_reward_plot.pdf"

echo "== Fig 5: beta_cov sweep on CIRCA =="
$PY "$PF/plot_klcov_sweep.py" --paper-panel --panel-tasks gsm8k,math500 \
    --base "$ROOT/eval_results_final/klcov_ablation" \
    --paper-panel-out "$FIG/ablation/klcov_plot.pdf"

echo "== Fig 4: INTUITOR + KL-Cov =="
$PY "$PF/plot_intuitor_klcov.py" --paper-panel --panel-tasks gsm8k,math500 \
    --base "$ROOT/eval_results_final/intuitor_klcov_ablation" \
    --paper-panel-out "$FIG/ablation/intuitor_klcov_plot.pdf"

echo "== 7B greedy trajectory panel =="
$PY "$PF/plot_trajectory_panel.py" --panel \
    --base "$ROOT/eval_results_final/trajectory_7B" \
    --panel-tasks gsm8k,math500,aime2024,aime2025 \
    --model-label Qwen2.5-7B \
    --out "$FIG/7b/trajectory_panel_7b.pdf"

echo "== 7B pass@k panels =="
# summarize_passk_trajectory.py writes beside its inputs; copy the paper figure over.
$PY "$ROOT/evaluation/summarize_passk_trajectory.py" --scale 7b
for f in combined_trajectory_passk avg_pass1_trajectory passmax_trajectory; do
  for ext in pdf png; do
    src="$ROOT/eval_results_final/passk_trajectory_7B/plots/$f.$ext"
    [ -f "$src" ] && cp "$src" "$FIG/7b/${f}_7b.$ext"
  done
done

echo "== Qwen3-1.7B (A-2) and GRPO-GT +/- KL-Cov (A-4) =="
$PY "$ROOT/evaluation/plot_trajectory_a2.py" \
    --results_dir "$ROOT/eval-rebuttal-results" \
    --out "$FIG/qwen3/trajectory_panel_a2_qwen3.pdf"
$PY "$ROOT/evaluation/plot_trajectory_a4.py" \
    --results_dir "$ROOT/eval-rebuttal-results" \
    --out "$FIG/grpo_klcov/trajectory_panel_a4_klcov.pdf"

echo "== teaser (Figure 1 candidates) =="
$PY "$ROOT/evaluation/plot_teaser.py" --outdir "$FIG/teaser_variants"

echo
echo "done -> $FIG"
find "$FIG" -name "*.pdf" | sort
