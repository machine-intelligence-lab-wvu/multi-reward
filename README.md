# Two is better than one: A Collapse-Resistant Multi-Reward RLIF Training Framework

Official code for the NeurIPS 2026 paper.

[**Paper**](https://arxiv.org/abs/2605.22620) · [**OpenReview**](https://openreview.net/forum?id=D8K5QK3kwT) · [**Project page**](https://machine-intelligence-lab-wvu.github.io/multi-reward/)

Reinforcement learning from internal feedback (RLIF) trains reasoning LLMs without labels,
verifiers or reward models, but single-signal RLIF collapses under continued training. The two
common intrinsic rewards fail in opposite directions: **cluster voting** (answer agreement) lets
the policy drift in reasoning space while funnelling answers into one cluster, and
**self-certainty** (token confidence) drives the policy to short, deterministic templates.

This repository trains with both rewards at once, normalizes each reward channel within its
prompt group (GDPO), and optionally adds **KL-Cov**, a KL penalty on the ~2% of tokens whose
log-probability covaries most with the advantage. Supervised GRPO with ground-truth rewards
(GRPO-GT) and INTUITOR (self-certainty only) are included as baselines.

## Contents

```
open_r1_imp/            training code (built on TRL's GRPOTrainer)
  train.py              single entry point; the recipe's `method` picks the trainer
  multi_reward_trainer.py   multi-reward advantage (cluster voting + self-certainty, GDPO)
  intuitor_trainer.py   INTUITOR baseline (self-certainty only)
  grpo_gt_trainer.py    GRPO-GT baseline (ground-truth accuracy reward)
  klcov.py              KL-Cov loss, shared by all trainers
  rewards.py            cluster_voting, boxed_format and accuracy rewards
  configs.py            recipe fields
recipes/<model>/        training recipes (one YAML per run)
scripts/train.sh        launch a recipe (starts the vLLM server + DeepSpeed training)
scripts/eval.sh         evaluate every checkpoint of a run
scripts/smoke_test.sh   CPU-only check of the training code
evaluation/run_task.py  benchmark evaluation with vLLM (GSM8K, MATH500, MMLU-Pro, AIME, LCB, CRUXEval)
evaluation/significance.py    Wilson CIs and exact McNemar tests
evaluation/paper_figures/     scripts for the paper's figures
docs/REPRODUCING.md     which recipe and evaluation produce each table and figure
tests/                  unit tests (pytest)
```

## Installation

Linux with NVIDIA GPUs, Python 3.10–3.12:

```bash
git clone https://github.com/machine-intelligence-lab-wvu/multi-reward.git
cd multi-reward
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation   # recipes use flash_attention_2
```

The trainers override parts of TRL's `GRPOTrainer`, so keep `trl==0.23.1`. Without
flash-attention, add `--attn_implementation sdpa` to the training command.

Check the installation without a GPU (a tiny model, two steps per method, a few minutes on CPU):

```bash
pytest tests/
bash scripts/smoke_test.sh
```

## Training

```bash
bash scripts/train.sh recipes/qwen2.5-1.5b/multi_reward_klcov.yaml
```

`train.sh` starts a vLLM generation server on the first GPU and trains with DeepSpeed ZeRO-3
on the others. The paper used 4 GPUs (1 vLLM + 3 training; NVIDIA GH200). With a different
number of GPUs (`NUM_GPUS=8 bash scripts/train.sh ...`) the script rescales gradient
accumulation so the effective batch stays the paper's 882 completions per step (126 prompts ×
7 samples). Checkpoints are written every 20 steps to `outputs/<model>_<recipe>/`; a run
resumes from its latest checkpoint when restarted. Any recipe field can be overridden on the
command line, e.g. `--max_steps 100 --output_dir outputs/test`. To log to Weights & Biases,
add `--report_to wandb`.

| Method | Recipe(s) |
|---|---|
| Multi-reward + KL-Cov (ours) | `recipes/{qwen2.5-1.5b,qwen2.5-3b,qwen2.5-7b,qwen3-1.7b}/multi_reward_klcov.yaml` |
| Multi-reward without KL-Cov | `recipes/{qwen2.5-1.5b,qwen2.5-3b}/multi_reward.yaml` |
| INTUITOR | `recipes/*/intuitor.yaml` |
| GRPO-GT (supervised) | `recipes/*/grpo_gt.yaml` |
| Cluster voting only | `recipes/qwen2.5-1.5b/cluster_only.yaml` |
| Ablations (Qwen2.5-1.5B) | `intuitor_klcov`, `multi_reward_klcov_beta{0.0005,0.1}`, `gdpo_ablation_{per_channel,combine_then_norm}`, `grpo_gt_noklcov`, `grpo_gt_klcov` |

All runs train on the MATH training split (`DigitalLearningGmbH/MATH-lighteval`, 7,500 problems)
with the system prompt in the recipe. Main settings: 7 samples per prompt, up to 3,072
completion tokens, learning rate 3e-6 (cosine), reference-KL 0.005, seed 7.

### What the recipe fields mean

The multi-reward advantage for completion *g* of a prompt is

```
A_g = w_cluster_format * z(w_cluster * cluster_voting + w_format * boxed_format)
    + w_selfcert       * z(self_certainty)
```

where `z` is the z-score within the prompt's group of 7 completions (the cluster and format
rewards are combined first and normalized together as one channel). With
`gdpo_norm_after_combine: true` (GDPO ablation) all rewards are summed first and normalized once.

KL-Cov (`beta_cov > 0`) gives the `top_k_fraction` of tokens with the highest covariance
between log-probability and advantage an unclipped surrogate plus the penalty
`beta_cov * |log pi_theta - log pi_old|`. The KL-Cov recipes set `loss_type: grpo`
(per-sequence averaging) and `vllm_importance_sampling_correction: false`, which is how those
runs were trained; the other recipes use TRL's defaults (`dapo` loss, importance-sampling
correction on).

`num_iterations` (μ, the number of optimizer passes per batch of rollouts) is set per recipe
and can be changed for further empirical analysis; we did not ablate it because of
computational cost.

## Evaluation

```bash
# greedy pass@1 on GSM8K, MATH500 and MMLU-Pro for every checkpoint (one GPU)
bash scripts/eval.sh outputs/qwen2.5-1.5b_multi_reward_klcov

# selected tasks / checkpoints / sampling settings
TASKS=aime2024,aime2025,lcb,cruxeval_o STEPS=160 bash scripts/eval.sh outputs/qwen2.5-1.5b_multi_reward_klcov
TASKS=gsm8k STEPS=240 EVAL_ARGS="--n_samples 8 --temperature 1.0 --seed 0" bash scripts/eval.sh outputs/qwen2.5-7b_multi_reward_klcov

# the untrained base model
bash scripts/eval.sh Qwen/Qwen3-1.7B-Base results/qwen3-1.7b
```

Results are written to `results/<model>/<recipe>_ckpt<step>/<recipe>_ckpt<step>__<task>.json`
with per-problem records. `python evaluation/summarize.py --output_dir results/qwen2.5-1.5b`
builds a summary table, and `evaluation/significance.py` computes confidence intervals and
McNemar tests. Math answers are scored with `math_verify`; MMLU-Pro uses TIGER-Lab's 5-shot CoT
prompt; LiveCodeBench (v6) and CRUXEval-O execute the generated code, so run code evaluations
in an isolated environment.

Paper figures are produced from these results and the training logs (`trainer_state.json`)
with the scripts in `evaluation/paper_figures/`; see [`docs/REPRODUCING.md`](docs/REPRODUCING.md).

## Citation

```bibtex
@inproceedings{joarder2026twoisbetter,
  title     = {Two is better than one: A Collapse-Resistant Multi-Reward {RLIF} Training Framework},
  author    = {Joarder, Shourov and Sikdar, Diganta and Akash, Ahsan Habib and Bhattarai, Binod and Gyawali, Prashnna},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

Apache 2.0 (see `LICENSE` and `NOTICE`). Built on [TRL](https://github.com/huggingface/trl) and
[Open-R1](https://github.com/huggingface/open-r1); the self-certainty reward follows
[INTUITOR](https://github.com/sunblaze-ucb/Intuitor) and KL-Cov follows Cui et al. (2025).
