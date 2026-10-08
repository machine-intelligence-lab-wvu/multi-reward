# Two is better than one: A Collapse-Resistant Multi-Reward RLIF Training Framework

Official code for the NeurIPS 2026 paper.

[**Paper**](https://arxiv.org/abs/2605.22620) · [**arXiv**](https://arxiv.org/abs/2605.22620) · [**OpenReview**](https://openreview.net/forum?id=D8K5QK3kwT) · [**Project page**](https://machine-intelligence-lab-wvu.github.io/multi-reward/)

---

Reinforcement learning from internal feedback (RLIF) trains reasoning LLMs with
no labels, no verifier and no reward model. It also collapses: every
single-signal method we tested reaches a strong peak and then falls to near-zero
accuracy under continued training.

The two dominant intrinsic rewards fail in **opposite directions**.
*Cluster voting* sees only the final answer, so the policy drifts in reasoning
space while funnelling every answer into one cluster — entropy and length
**rise** as accuracy crashes. *Self-certainty* sees only token confidence, so
the policy becomes deterministic — entropy falls to ~0.1 nats and completions
shrink to ~100-token templates.

**CIRCA** optimizes both at once, normalizes each channel within the prompt
group (GDPO), and adds **KL-Cov** — a KL penalty applied only to the ~2% of
tokens whose log-probability correlates most with advantage. It tracks
supervised RLVR across four backbones while single-signal baselines collapse.

---

## Install

```bash
git clone https://github.com/machine-intelligence-lab-wvu/multi-reward.git
cd multi-reward

conda create -n multi-reward python=3.11 -y
conda activate multi-reward
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```

Tested on 4×GH200 (96 GB) and 2×A6000 (48 GB). Full fine-tuning of Qwen2.5-1.5B
needs ~4 GPUs with one reserved for the vLLM generation server; LoRA configs run
on fewer.

### Environment variables

Paths are not hardcoded. Set these before running anything:

| Variable | Meaning |
|---|---|
| `MR_SCRATCH` | Root scratch directory |
| `MR_OUTPUT_DIR` | Where checkpoints and logs are written |
| `MR_DATA_DIR` | Datasets |
| `MR_CACHE_DIR` | Triton / compile caches |
| `HF_HOME` | Hugging Face cache |
| `MR_VENV` | Virtualenv root (SLURM scripts only) |

```bash
export MR_SCRATCH=/your/scratch
export MR_OUTPUT_DIR=$MR_SCRATCH/outputs
export MR_DATA_DIR=$MR_SCRATCH/data
export MR_CACHE_DIR=$MR_SCRATCH/cache
export HF_HOME=$MR_SCRATCH/hf
```

## Repository layout

```
open_r1_imp/              training package (built on HF Open-R1 + TRL)
├── multi_reward_trainer.py   CIRCA: GDPO normalization + KL-Cov          ★
├── intuitor_trainer.py       self-certainty-only baseline
├── grpo_gt_trainer.py        supervised GRPO-GT upper bound
├── rewards_multi.py          reward registry
├── recipes/final/            the 21 configs behind the paper's tables     ★
└── scripts/                  launch scripts (local + SLURM)

train_4090_imp/rewards/   reward implementations                           ★
├── cluster_voting.py         answer-level consensus reward
├── self_certainty.py         completion-level confidence reward
├── gdpo_advantage.py         per-channel group normalization
└── answer_extraction.py      boxed-answer parsing and equivalence

callbacks/klcov.py        standalone KL-Cov callback                       ★

data/                     dataset preparation (prepare_math.py, prepare_gsm8k.py)

evaluation/               benchmark harnesses (GSM8K, MATH-500, MMLU-Pro,
                          AIME, LiveCodeBench, CRUXEval) + trajectory tools

docs/                     project page (served by GitHub Pages) + REPRODUCING.md
```

The starred paths are the method. Everything else is infrastructure.

Prepare the training data once with `python data/prepare_math.py` (writes to
`$MR_DATA_DIR`).

## Training

Each paper run is one config in `open_r1_imp/recipes/final/`.

Every method shares the same entry-point shape — an `accelerate launch` of one
`*_train.py` with one `--config`:

```bash
CONFIG=open_r1_imp/recipes/final/config_multi_reward_klcov_0.05_qwen25_1.5b.yaml

accelerate launch --config_file open_r1_imp/accelerate_configs/zero2.yaml \
    --num_processes 3 \
    open_r1_imp/multi_reward_train.py --config $CONFIG
```

Swap the entry point for the baselines — `open_r1_imp/intuitor_train.py` or
`open_r1_imp/grpo_gt_train.py` — keeping `--config` pointed at the matching
recipe.

**Wrapper scripts** handle the vLLM generation server, GPU split and accelerate
config for you:

```bash
# local / interactive (CIRCA)
bash open_r1_imp/scripts/run_multi_reward.sh \
     --config open_r1_imp/recipes/final/config_multi_reward_klcov_0.05_qwen25_1.5b.yaml \
     --mode full --num_train_gpus 3

# single GPU, no vLLM server — smoke test
bash open_r1_imp/scripts/run_multi_reward.sh --mode lora --num_train_gpus 1 --no_vllm

# SLURM (baselines pass the config through an env var)
sbatch --export=ALL,CONFIG_OVERRIDE=open_r1_imp/recipes/final/config_intuitor_qwen25_1.5b.yaml \
       open_r1_imp/scripts/submit_intuitor.sh

sbatch --export=ALL,CONFIG_OVERRIDE=open_r1_imp/recipes/final/config_grpo_gt_qwen25_1.5b.yaml \
       open_r1_imp/scripts/submit_grpo_gt.sh
```

`--num_train_gpus N` uses N GPUs for training plus one for vLLM. Local variants
for the baselines are `run_intuitor_local.sh` and `run_multi_reward_local.sh`;
`smoke_test.sh` runs a minimal end-to-end check.

### Which config produces which result

| Paper result | Config |
|---|---|
| CIRCA, 1.5B | `config_multi_reward_klcov_0.05_qwen25_1.5b.yaml` |
| CIRCA, 3B | `config_multi_reward_klcov_0.05_qwen25_3b.yaml` |
| CIRCA, 7B | `config_multi_reward_klcov_0.05_qwen25_7b.yaml` |
| CIRCA w/o KL-Cov | `config_multi_reward_noklcov_qwen25_{1.5b,3b}.yaml` |
| INTUITOR | `config_intuitor_qwen25_{1.5b,3b,7b}.yaml` |
| INTUITOR + KL-Cov | `config_intuitor_klcov_0.02_qwen25_1.5b.yaml` |
| Cluster-only (≈ EMPO) | `config_cluster_only_noklcov_qwen25_1.5b.yaml` |
| GRPO-GT (supervised) | `config_grpo_gt_qwen25_{1.5b,3b,7b}.yaml` |
| β_cov sweep | `config_multi_reward_klcov_{0.0005,0.02,0.05,0.1}_qwen25_1.5b.yaml` |

`*_local48g.yaml` are reduced-memory variants for 2×48 GB; `*_lora.yaml` is LoRA.

See [`docs/REPRODUCING.md`](docs/REPRODUCING.md) for the full table-by-table mapping.

## Evaluation

```bash
# one checkpoint, one dataset  (--dataset: gsm8k | math500 | math)
python evaluation/eval_checkpoint.py \
       --checkpoint $MR_OUTPUT_DIR/<run>/checkpoint-240 \
       --dataset gsm8k

# several benchmarks for one model, sampled
python evaluation/eval_all.py \
       --model_path $MR_OUTPUT_DIR/<run>/checkpoint-240 \
       --benchmarks gsm8k math500 \
       --n_samples 16 --temperature 0.6

# vLLM-backed evaluation (much faster for full sweeps)
bash open_r1_imp/scripts/run_eval_vllm.sh
```

`eval_checkpoint.py` auto-detects LoRA vs full fine-tune (`--mode auto`); force
it with `--mode lora|full`. Add `--base_only` to score the untrained base model,
and `--wandb` to log results.

Math is scored with `math_verify`; MMLU-Pro uses TIGER-Lab's 5-shot CoT prompt;
LiveCodeBench follows the lighteval pipeline; CRUXEval-O uses ZeroEval's OEQA
template. Code-reasoning evaluation runs through
`open_r1_imp/scripts/submit_code_eval.sh`.

## Method in brief

For each prompt we sample `G` completions and score them twice:

**Cluster voting** — rollouts are grouped by answer equivalence and rewarded by
cluster size, `R_cluster(c_g) = |C(a_g)| / n_p`. A dense majority vote: minority
clusters get a small non-zero reward rather than zero.

**Self-certainty** — mean KL between the uniform distribution and the model's
next-token distribution, averaged over every token.

**GDPO normalization** — each channel is *z*-scored within the prompt group
*before* combining, `A_g = α·R̂_cluster + β·R̂_cert`. This is not cosmetic:
combined naively the self-certainty channel dominates the gradient by roughly an
order of magnitude, collapsing the objective back to a single signal.

**KL-Cov** — the top ~2% of tokens by covariance between log-probability and
advantage receive an extra KL penalty; the other ~98% keep the standard clipped
surrogate. Performance is stable across a 200× range of `β_cov`.

## Notes and limitations

- Evidence supports collapse **resistance** over the horizons, scales and priors
  tested — not universal immunity.
- The cluster channel needs an answer-equivalence relation: exact for math and
  code, semantic clustering for open-ended tasks (not evaluated here), undefined
  for genuinely multi-valued tasks.
- Each trajectory is a single training run; multi-day compute cost put seed
  replication out of reach.
- Checkpoints and raw evaluation dumps are not in this repository.

## Citation

```bibtex
@inproceedings{joarder2026two,
  title     = {Two is better than one: A Collapse-Resistant
               Multi-Reward {RLIF} Training Framework},
  author    = {Joarder, Shourov and Sikdar, Diganta and
               Akash, Ahsan Habib and Bhattarai, Binod and
               Gyawali, Prashnna},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  eprint    = {2605.22620},
  archivePrefix = {arXiv}
}
```

## License and attribution

Apache-2.0 — see [`LICENSE`](LICENSE).

Built on [Open-R1](https://github.com/huggingface/open-r1) and
[TRL](https://github.com/huggingface/trl) (both Apache-2.0), with the
self-certainty reward ported from [INTUITOR](https://github.com/sunblaze-ucb/Intuitor)
and the consensus reward following [EMPO](https://github.com/QingyangZhang/EMPO).
Full attribution in [`NOTICE`](NOTICE).

## Acknowledgments

This work was supported by the Defense Advanced Research Projects Agency (DARPA)
under Grant AI-CRAFT AWD16069. It was also supported as part of the Swiss AI
Initiative by a grant from the Swiss National Supercomputing Centre (CSCS) under
project ID a168 on Alps.
