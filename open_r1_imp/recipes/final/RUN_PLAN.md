# Final Runs — Plan

All configs here are post-fix (commit `bfb30a8`), matched to Intuitor hyperparameters
for apples-to-apples comparison, and write to
`${MR_OUTPUT_DIR}/multi_reward_final_*/` (no collisions with pre-fix runs).

Eval outputs go to `eval_results_final/`.

---

## Shared hyperparameters (all 7 configs)

| Knob | Value | Notes |
|---|---|---|
| dataset | `DigitalLearningGmbH/MATH-lighteval` | |
| base model | Qwen2.5-1.5B or -3B | |
| `learning_rate` | 3.0e-6 | matched to Intuitor |
| `epsilon` (PPO clip) | 0.2 | explicit |
| `beta` (KL-to-ref) | 0.005 | |
| `temperature` | 1.0 | |
| `num_generations` | 7 | |
| `max_completion_length` | 3072 | |
| `max_steps` | 500 | |
| effective batch | 882 (per_device × grad_accum × 3 GPUs) | |
| `w_cluster_format`, `w_selfcert` | 0.5, 0.5 | GDPO channel weights |
| `w_cluster`, `w_format` | 0.5, 0.1 | within-channel reward weights |
| `top_k_fraction` | 0.02 | for KL-Cov (ignored when `beta_cov=0`) |
| `min_completion_tokens` | 0 | disabled (matches Intuitor) |
| `thinking` | false | |
| `seed` | 7 | |

What varies per config: **model size** (1.5B vs 3B) and **`beta_cov`** (0, 0.0005, 0.02, 0.05, 0.1).

---

## Run list

### 1.5B suite (5 runs)

| # | Config | `beta_cov` | Purpose |
|---|---|---|---|
| 4 | `config_multi_reward_noklcov_qwen25_1.5b.yaml` | 0.0 | Pure multi-reward baseline |
| 5 | `config_multi_reward_klcov_0.02_qwen25_1.5b.yaml` | 0.02 | Canonical klcov comparison (matches Intuitor klcov 1.5B) |
| 7 | `config_multi_reward_klcov_0.0005_qwen25_1.5b.yaml` | 0.0005 | β_cov sweep: low end (near-exploit) |
| 8 | `config_multi_reward_klcov_0.05_qwen25_1.5b.yaml` | 0.05 | β_cov sweep: mid |
| 6 | `config_multi_reward_klcov_0.1_qwen25_1.5b.yaml` | 0.1 | β_cov sweep: high (most explore) |

### 3B suite (2 runs)

| # | Config | `beta_cov` | Purpose |
|---|---|---|---|
| 1 | `config_multi_reward_noklcov_qwen25_3b.yaml` | 0.0 | 3B no-KLCov baseline (NEW) |
| 2 | `config_multi_reward_klcov_0.05_qwen25_3b.yaml` | 0.05 | 3B canonical klcov (matches Intuitor klcov 3B) |

**Total: 7 training runs.**

Numbering matches the user's TODO list. Suggested execution order:

1. **Validation pass first** — run the 1.5B klcov(0.02) for ~20 steps and check
   wandb (see smoke-check below). Confirms the fix is active before spending compute
   on the full suite.
2. **3B suite next** (items 1, 2) — longest wall-clock; get them running early.
3. **1.5B β_cov sweep** (items 4, 5, 6, 7, 8) — can go in parallel or staggered.

---

## Launch commands

All use `open_r1_imp/scripts/submit.sh` with `CONFIG_OVERRIDE`:

```bash
# 3B suite
sbatch --export=ALL,CONFIG_OVERRIDE=open_r1_imp/recipes/final/config_multi_reward_noklcov_qwen25_3b.yaml,ACCEL_CONFIG_OVERRIDE=open_r1_imp/accelerate_configs/zero3_grad21.yaml \
    open_r1_imp/scripts/submit.sh
sbatch --export=ALL,CONFIG_OVERRIDE=open_r1_imp/recipes/final/config_multi_reward_klcov_0.05_qwen25_3b.yaml,ACCEL_CONFIG_OVERRIDE=open_r1_imp/accelerate_configs/zero3_grad21.yaml \
    open_r1_imp/scripts/submit.sh

# 1.5B suite
sbatch --export=ALL,CONFIG_OVERRIDE=open_r1_imp/recipes/final/config_multi_reward_noklcov_qwen25_1.5b.yaml \
    open_r1_imp/scripts/submit.sh
sbatch --export=ALL,CONFIG_OVERRIDE=open_r1_imp/recipes/final/config_multi_reward_klcov_0.02_qwen25_1.5b.yaml \
    open_r1_imp/scripts/submit.sh
sbatch --export=ALL,CONFIG_OVERRIDE=open_r1_imp/recipes/final/config_multi_reward_klcov_0.0005_qwen25_1.5b.yaml \
    open_r1_imp/scripts/submit.sh
sbatch --export=ALL,CONFIG_OVERRIDE=open_r1_imp/recipes/final/config_multi_reward_klcov_0.05_qwen25_1.5b.yaml \
    open_r1_imp/scripts/submit.sh
sbatch --export=ALL,CONFIG_OVERRIDE=open_r1_imp/recipes/final/config_multi_reward_klcov_0.1_qwen25_1.5b.yaml \
    open_r1_imp/scripts/submit.sh
```

(1.5B runs use `accelerate_configs/zero3.yaml` by default; 3B runs need the `zero3_grad21.yaml` override to match the grad_accum=21 split.)

---

## Output paths

All under `${MR_OUTPUT_DIR}/`:

```
outputs/
├── multi_reward_final_noklcov_qwen25_1.5b/
├── multi_reward_final_klcov_0.02_qwen25_1.5b/
├── multi_reward_final_klcov_0.05_qwen25_1.5b/
├── multi_reward_final_klcov_0.1_qwen25_1.5b/
├── multi_reward_final_klcov_0.0005_qwen25_1.5b/
├── multi_reward_final_noklcov_qwen25_3b/
└── multi_reward_final_klcov_0.05_qwen25_3b/
```

wandb run names follow `multi-reward-final-{kl}-qwen25-{size}`:
e.g. `multi-reward-final-klcov(0.02)-qwen25-1.5b`,
`multi-reward-final-noklcov-qwen25-3b`.

HF Hub: all push to `shourovj/multi-reward-grpo` with branches per
`run_name + step`. Set `push_to_hub: false` on any config you don't want pushed.

---

## Baseline / comparison runs (already exist, do NOT re-run)

These provide the "Intuitor vs. multi-reward" comparison:

| Baseline | Config | `beta_cov` | Status |
|---|---|---|---|
| Intuitor 1.5B | `open_r1_imp/recipes/config_intuitor_clariden.yaml` | — | done |
| Intuitor klcov 1.5B | `open_r1_imp/recipes/config_intuitor_klcov_clariden.yaml` | 0.02 | done |
| Intuitor 3B | `open_r1_imp/recipes/config_intuitor_3b.yaml` | — | done |
| Intuitor klcov 3B | `open_r1_imp/recipes/config_intuitor_klcov_3b.yaml` | 0.05 | done |
| GRPO-GT 1.5B | `open_r1_imp/recipes/config_grpo_gt_clariden.yaml` | — | done |
| GRPO-GT 3B | `open_r1_imp/recipes/config_grpo_gt_3b.yaml` | — | done |

---

## Evaluation

All final eval artifacts go to `eval_results_final/` (rename of the existing
`eval_results_h200/` convention).

Per-model evals:
- 1.5B: run `open_r1_imp/scripts/submit_eval.sh` with each 1.5B checkpoint path.
- 3B: same with 3B checkpoints.

The eval script should be pointed at `eval_results_final/` — see the one-line
edit in `submit_eval.sh`.

---

## Smoke-check (run once before spending compute on the full suite)

Launch `config_multi_reward_klcov_0.02_qwen25_1.5b.yaml` for ~20 steps and
verify on wandb:

1. **`rewards/cluster_voting/mean`** — varying across steps, not constant zero.
2. **`rewards/boxed_format/mean`** — same.
3. **`advantages/gdpo_combined_std`** — noticeably **larger** than the corresponding
   Intuitor run's `advantages/ch_sc_std`. If it's equal or smaller, the fix is
   not active — check `MultiRewardTrainer.__init__` is overriding `self.reward_weights`.
4. **Run log at startup** should contain:
   `MultiRewardTrainer: w_cf=0.5, w_sc=0.5, w_cluster=0.5, w_format=0.1, ...`
   (the `w_cluster` / `w_format` fields are only logged post-fix.)

---

## Pre-fix reference (do not re-run)

Pre-fix multi-reward checkpoints are effectively SC-only (see
`docs/MULTI_REWARD_FIX_ANALYSIS.md`). They live at:

- `${MR_OUTPUT_DIR}/run3_full_qwen2.5-1.5b_klcov-0.02/` — pre-fix 1.5B klcov 0.02
- `${MR_OUTPUT_DIR}/multi_reward_klcov_0.05_qwen25_3b/` — pre-fix 3B klcov 0.05
- `${MR_OUTPUT_DIR}/multi_reward_noklcov_qwen25_1.5b/` — pre-fix 1.5B noklcov

The "canonical" final runs in `outputs/multi_reward_final_*/` supersede these.
