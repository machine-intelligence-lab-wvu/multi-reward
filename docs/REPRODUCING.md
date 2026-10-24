# Reproducing the paper

Every experiment is one recipe trained with `scripts/train.sh` and evaluated with
`scripts/eval.sh`. Run directories are `outputs/<model>_<recipe>/`; evaluation results go to
`results/<model>/`. Commands below assume the repository root.

Training is stochastic (vLLM sampling), and every run in the paper is a single seed (7), so
expect run-to-run variation, especially in when single-reward runs collapse.

**Training horizon.** The Qwen2.5-1.5B/3B recipes have `max_steps: 500` (the cosine schedule
spans 500 steps) and the paper evaluates checkpoints 20–340, so training can be stopped after
step 340; do not lower `max_steps`, which would change the learning-rate schedule. The 7B
recipes have `max_steps: 340` and were evaluated through step 240; Qwen3-1.7B and the
GRPO-GT ± KL-Cov runs train for 200 steps.

## Main results (Qwen2.5-1.5B and 3B)

Runs, for `M` in `qwen2.5-1.5b` and `qwen2.5-3b`:

```bash
for R in grpo_gt intuitor multi_reward multi_reward_klcov; do bash scripts/train.sh recipes/$M/$R.yaml; done
```

| Result | Evaluation |
|---|---|
| Trajectory figure, stability table, retention table | `bash scripts/eval.sh outputs/${M}_$R` (GSM8K, MATH500, MMLU-Pro, every 20 steps) |
| Table 1 (step 160; math and code) | `TASKS=gsm8k,math500,mmlu_pro,lcb,cruxeval_o STEPS=160 bash scripts/eval.sh outputs/${M}_$R` |
| AIME table (step 160) | `TASKS=aime2024,aime2025 STEPS=160 bash scripts/eval.sh outputs/${M}_$R` |

```bash
python evaluation/paper_figures/plot_trajectory_panel.py --combined \
    --base results/qwen2.5-1.5b --base-3b results/qwen2.5-3b --exclude intuitor_klcov \
    --out figures/trajectory_panel.pdf
```

## Qwen2.5-7B (appendix)

Recipes `recipes/qwen2.5-7b/{grpo_gt,intuitor,multi_reward_klcov}.yaml`.

| Result | Evaluation |
|---|---|
| Greedy trajectories (steps 20–240) | `TASKS=gsm8k,math500,aime2024,aime2025 bash scripts/eval.sh outputs/qwen2.5-7b_$R` |
| Sampling (avg-pass@1, pass@K), GSM8K | `TASKS=gsm8k STEPS=40,80,120,160,200,240 EVAL_ARGS="--n_samples 8 --temperature 1.0" bash scripts/eval.sh outputs/qwen2.5-7b_$R results/qwen2.5-7b-t1` |
| Sampling, MATH500 | same with `TASKS=math500` and `--n_samples 16` |
| Sampling, AIME (step 240) | `TASKS=aime2024,aime2025 STEPS=240 EVAL_ARGS="--n_samples 32 --temperature 1.0"` |

```bash
python evaluation/paper_figures/plot_trajectory_panel.py --panel --base results/qwen2.5-7b \
    --model-label Qwen2.5-7B --panel-tasks gsm8k,math500 --out figures/trajectory_7b.pdf
```

## Qwen3-1.7B-Base (appendix)

Recipes `recipes/qwen3-1.7b/{grpo_gt,intuitor,multi_reward_klcov}.yaml` (`beta_cov: 0.02`).

```bash
TASKS=gsm8k,math500 bash scripts/eval.sh outputs/qwen3-1.7b_$R
TASKS=mmlu_pro STEPS=200 EVAL_ARGS="--max_samples 2000 --sample_seed 7" bash scripts/eval.sh outputs/qwen3-1.7b_$R
TASKS=gsm8k,math500 bash scripts/eval.sh Qwen/Qwen3-1.7B-Base results/qwen3-1.7b
TASKS=mmlu_pro EVAL_ARGS="--max_samples 2000 --sample_seed 7" bash scripts/eval.sh Qwen/Qwen3-1.7B-Base results/qwen3-1.7b

python evaluation/paper_figures/plot_qwen3_trajectory.py --results_dir results/qwen3-1.7b --out figures/trajectory_qwen3.pdf
python evaluation/significance.py --results-dir results/qwen3-1.7b --step 200     # Wilson CIs + McNemar
```

The MMLU-Pro subset (2,000 questions, seed 7) is drawn at random across all 14 categories.

## GRPO-GT with and without KL-Cov (appendix)

Recipes `recipes/qwen2.5-1.5b/grpo_gt_noklcov.yaml` and `grpo_gt_klcov.yaml`. The two
recipes are identical except for KL-Cov.

```bash
TASKS=gsm8k,math500 bash scripts/eval.sh outputs/qwen2.5-1.5b_grpo_gt_noklcov
TASKS=gsm8k,math500 bash scripts/eval.sh outputs/qwen2.5-1.5b_grpo_gt_klcov
python evaluation/paper_figures/plot_grpo_gt_klcov_trajectory.py --results_dir results/qwen2.5-1.5b \
    --out figures/trajectory_grpo_gt_klcov.pdf
python evaluation/significance.py --results-dir results/qwen2.5-1.5b --step 200 \
    --compare "grpo_gt_klcov:grpo_gt_noklcov:GRPO-GT+KL-Cov vs GRPO-GT"
```

## Ablations (Qwen2.5-1.5B)

All evaluated with `bash scripts/eval.sh outputs/qwen2.5-1.5b_$R` (GSM8K, MATH500, MMLU-Pro,
every 20 steps; the paper shows steps 20–280). Completion length and entropy come from each
run's `trainer_state.json`.

| Experiment | Recipes | Figure |
|---|---|---|
| Single- vs multi-reward (no KL-Cov) | `cluster_only`, `intuitor`, `multi_reward` | `plot_single_multi_reward.py --paper-panel --base results/qwen2.5-1.5b --train-dir outputs --paper-panel-out figures/single_multi_reward.pdf` |
| KL-Cov coefficient sweep | `multi_reward`, `multi_reward_klcov_beta0.0005`, `multi_reward_klcov`, `multi_reward_klcov_beta0.1` | `plot_klcov_sweep.py --paper-panel --base results/qwen2.5-1.5b --train-dir outputs --paper-panel-out figures/klcov_sweep.pdf` |
| INTUITOR + KL-Cov | `intuitor`, `intuitor_klcov` | `plot_intuitor_klcov.py --paper-panel --base results/qwen2.5-1.5b --train-dir outputs --paper-panel-out figures/intuitor_klcov.pdf` |
| Component ablation table | `multi_reward_klcov`, `multi_reward`, `intuitor`, `cluster_only` | peak and step-340 values from the trajectories |
| GDPO normalization | `gdpo_ablation_per_channel`, `gdpo_ablation_combine_then_norm` | `plot_gdpo_ablation.py accuracy --base results/qwen2.5-1.5b --out figures/gdpo_accuracy.pdf` and `plot_gdpo_ablation.py gradients --train-dir outputs --out figures/gdpo_gradients.pdf` |

(Figure scripts live in `evaluation/paper_figures/`.) The two GDPO-ablation recipes are the
main recipe plus the log-prob-space gradient probe (`grad_corr_enabled: true`); the second also
sets `gdpo_norm_after_combine: true`.
