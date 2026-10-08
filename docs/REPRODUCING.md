# Reproducing the paper

Every number in the paper comes from one config in
`open_r1_imp/recipes/final/`. This file maps results to configs.

## Before you start

```bash
export MR_SCRATCH=/your/scratch
export MR_OUTPUT_DIR=$MR_SCRATCH/outputs
export MR_DATA_DIR=$MR_SCRATCH/data
export MR_CACHE_DIR=$MR_SCRATCH/cache
export HF_HOME=$MR_SCRATCH/hf
export PYTHONPATH=$PWD:$PYTHONPATH
```

Training data is the MATH training split (`MATH-lighteval`), used **without
labels** for every unsupervised run. Ground truth is consumed only by the
supervised `grpo_gt` baseline and by evaluation of frozen checkpoints.

## Shared hyperparameters

Identical across every run unless a config says otherwise:

| Setting | Value |
|---|---|
| Optimizer | GRPO, lr `3e-6`, cosine schedule |
| Rollouts per prompt | `G = 7` |
| Max completion length | 3072 tokens |
| GRPO KL coefficient | `β = 0.005` |
| Effective batch | 882 prompts / step (6,174 rollouts) |
| Reward channel weights | `α = β = 0.5` |
| KL-Cov top-k fraction | `k = 0.02` (top 2% of tokens) |
| Default `β_cov` | `0.05` |
| Horizon | 340 steps (40 epochs); 240 at 7B, 200 for Qwen3 |

## Main results · stability under continued training

Trained on Qwen2.5-1.5B and 3B, evaluated every 20 steps.

| Row in paper | Config |
|---|---|
| GRPO-GT (supervised) | `config_grpo_gt_qwen25_1.5b.yaml` · `..._3b.yaml` |
| INTUITOR | `config_intuitor_qwen25_1.5b.yaml` · `..._3b.yaml` |
| CIRCA w/o KL-Cov | `config_multi_reward_noklcov_qwen25_1.5b.yaml` · `..._3b.yaml` |
| **CIRCA (ours)** | `config_multi_reward_klcov_0.05_qwen25_1.5b.yaml` · `..._3b.yaml` |

## Scaling · Qwen2.5-7B

| Row | Config |
|---|---|
| GRPO-GT | `config_grpo_gt_qwen25_7b.yaml` |
| INTUITOR | `config_intuitor_qwen25_7b.yaml` |
| CIRCA | `config_multi_reward_klcov_0.05_qwen25_7b.yaml` |

Horizon is 240 steps — the longest common budget under a 12 h walltime on
4×GH200.

## Ablations

**Complementary failure modes** — cluster-voting vs self-certainty vs both,
all with KL-Cov off:

| Arm | Config |
|---|---|
| Cluster-only (≈ EMPO objective on math) | `config_cluster_only_noklcov_qwen25_1.5b.yaml` |
| Self-certainty-only (INTUITOR) | `config_intuitor_qwen25_1.5b.yaml` |
| Multi-reward, no KL-Cov | `config_multi_reward_noklcov_qwen25_1.5b.yaml` |

**KL-Cov on a single signal** — does KL-Cov alone rescue INTUITOR?

| Arm | Config |
|---|---|
| INTUITOR + KL-Cov (1.5B) | `config_intuitor_klcov_0.02_qwen25_1.5b.yaml` |
| INTUITOR + KL-Cov (3B) | `config_intuitor_klcov_0.05_qwen25_3b.yaml` |

**β_cov sweep** — 200× range, 1.5B:

`config_multi_reward_klcov_{0.0005,0.02,0.05,0.1}_qwen25_1.5b.yaml`

## Rebuttal / extended experiments

`open_r1_imp/recipes/rebuttal/` holds the additional runs:

| Experiment | Config |
|---|---|
| Co-rewarding comparison | `A3_corewarding_qwen25_15b.yaml` |
| Global entropy bonus vs KL-Cov | `B7_multireward_globalentropy_qwen25_15b.yaml` |

See `open_r1_imp/recipes/rebuttal/RUN_PLAN.md` and `ICLR_RUN_PLAN.md` for the
protocol notes on those runs.

## Reduced-hardware variants

Not used for paper numbers — provided so the method can be run on smaller
machines:

- `*_local48g.yaml` — 2×48 GB
- `*_lora.yaml` — LoRA instead of full fine-tuning

## What is not in this repository

- Model checkpoints
- Raw evaluation dumps and per-checkpoint score files
- Paper sources and figures

Trajectory figures and the full result tables are on the
[project page](https://shourovj.github.io/projects-sites/unsupervised-rl/).

## Caveats

Each trajectory in the paper is a **single training run** — the compute cost of
multi-day runs put seed replication out of reach. Expect run-to-run variation,
particularly in the exact step at which a single-signal baseline collapses. The
*ordering* of methods was consistent at every checkpoint we evaluated.
