# Multi-Reward GRPO — Cluster Training

Full-scale multi-reward GRPO training. Self-contained — depends only on TRL, no Open-R1 installation needed. Designed for multi-GPU clusters (4x H200 120GB, or any multi-GPU setup).

## Architecture

```
┌────────────────┐   completions    ┌──────────────────────────────────┐
│    GPU 0       │  <────────────── │    GPUs 1..N (DeepSpeed)         │
│  vLLM Server   │  ────────────────│    MultiRewardTrainer            │
│(trl vllm-serve)  + logprobs       │                                  │
│  - base model  │                  │  Reward Channels:                │
│  - dynamic LoRA│                  │    ├─ Cluster voting (EMPO)      │
│                │                  │    ├─ Format reward              │
│                │                  │    └─ Self-certainty KL(U||π)    │
│                │                  │                                  │
│                │                  │  GDPO: z-score per channel,      │
│                │                  │        weighted combine          │
│                │                  │                                  │
│                │                  │  Loss extensions:                │
│                │                  │    ├─ KL-Cov regularization      │
│                │                  │    └─ Entropy bonus              │
└────────────────┘                  └──────────────────────────────────┘
```

## Key Features

- **Self-contained** — imports from TRL directly, no Open-R1 package needed. `git clone` + `pip install trl` and run.
- **Multi-reward GDPO** — per-channel z-score normalization (cluster+format + self-certainty), weighted combination
- **EMPO semantic clustering** — DeBERTa NLI-based answer clustering with entropy filtering, falls back to string matching for numeric answers
- **Thinking mode** — Qwen3's `<think>` tags: self-certainty on reasoning only, rewards on answer only
- **KL-Cov regularization** — high-covariance tokens get unclipped loss + extra KL penalty
- **LoRA + Full fine-tuning** — toggle via `--mode lora|full`, auto-selects DeepSpeed ZeRO-2 or ZeRO-3
- **GPU-adaptive** — `--num_train_gpus N` controls training GPUs, vLLM always on GPU 0

## File Structure

```
open_r1_imp/
├── __init__.py                    # Package init
├── utils.py                       # GDPO normalize, reasoning mask, SC, model/dataset loaders
├── rewards_multi.py               # cluster_voting, empo_cluster, boxed_format rewards
├── configs_multi.py               # MultiRewardGRPOConfig, MultiRewardScriptArguments (extends TRL)
├── multi_reward_trainer.py        # MultiRewardTrainer (extends TRL's GRPOTrainer)
├── multi_reward_train.py          # Entry point — multi-reward training
├── configs_intuitor.py            # IntuitorGRPOConfig, IntuitorScriptArguments (extends TRL)
├── intuitor_trainer.py            # IntuitorTrainer — self-certainty as sole reward (INTUITOR paper)
├── intuitor_train.py              # Entry point — INTUITOR training
├── eval_callback.py               # In-training accuracy eval callback
├── hub_callback.py                # HF Hub push callback
├── recipes/
│   ├── config_math_lora.yaml      # Multi-reward LoRA
│   ├── config_math_full.yaml      # Multi-reward full FT (local)
│   ├── config_math_full_clariden.yaml       # Multi-reward full FT (Clariden)
│   ├── config_math_full_qwen25_1.5b.yaml    # Multi-reward Qwen2.5-1.5B
│   ├── config_math_thinking.yaml            # Multi-reward thinking mode
│   ├── config_intuitor_clariden.yaml        # INTUITOR full FT (paper reproduction)
│   ├── config_intuitor_lora_clariden.yaml   # INTUITOR LoRA
│   └── config_intuitor_qwen3_clariden.yaml  # INTUITOR Qwen3 thinking mode
├── accelerate_configs/
│   ├── zero2.yaml                 # DeepSpeed ZeRO-2 (LoRA)
│   └── zero3.yaml                 # DeepSpeed ZeRO-3 (full FT)
├── scripts/
│   ├── run_multi_reward.sh        # Interactive launch script
│   ├── submit.sh                  # SLURM: multi-reward training
│   ├── submit_eval.sh             # SLURM: checkpoint evaluation
│   └── submit_intuitor.sh         # SLURM: INTUITOR training
└── README.md

evaluation/
├── eval_checkpoint.py             # Evaluate a single checkpoint (GSM8K, MATH500, MATH)
└── eval_during_training.py        # Async eval — watches for new checkpoints
```

## Dependencies

```bash
pip install torch trl peft vllm datasets wandb deepspeed accelerate
```

No separate Open-R1 clone/install needed. Everything imports from TRL directly.

## Quick Start

### Local (Interactive)

#### LoRA Training (4 GPUs: 1 vLLM + 3 training)

```bash
bash open_r1_imp/scripts/run_multi_reward.sh --mode lora --num_train_gpus 3
```

#### Full Fine-Tuning

```bash
bash open_r1_imp/scripts/run_multi_reward.sh --mode full --num_train_gpus 3
```

#### Thinking Mode (Qwen3)

```bash
bash open_r1_imp/scripts/run_multi_reward.sh --mode lora --thinking --num_train_gpus 3
```

#### Full FT + Thinking Mode

```bash
bash open_r1_imp/scripts/run_multi_reward.sh --mode full --thinking --num_train_gpus 3
```

### SLURM Cluster (CSCS Clariden H200/GH200)

#### Submit Multi-Reward Training Job

```bash
# Full fine-tuning (default: 4 GPUs — 1 vLLM + 3 training)
sbatch open_r1_imp/scripts/submit.sh

# LoRA training
sbatch open_r1_imp/scripts/submit.sh --export=ALL,MODE=lora

# Full FT with custom GPU count
sbatch open_r1_imp/scripts/submit.sh --export=ALL,MODE=full,NUM_TRAIN_GPUS=3

# Thinking mode
sbatch open_r1_imp/scripts/submit.sh --export=ALL,THINKING=true
```

#### Submit INTUITOR Training Job

```bash
# Paper reproduction (Qwen2.5-1.5B, full FT)
sbatch open_r1_imp/scripts/submit_intuitor.sh

# LoRA mode
sbatch open_r1_imp/scripts/submit_intuitor.sh --export=ALL,MODE=lora

# Qwen3 with thinking mode (comparison with multi-reward)
sbatch open_r1_imp/scripts/submit_intuitor.sh --export=ALL,CONFIG_OVERRIDE=open_r1_imp/recipes/config_intuitor_qwen3_clariden.yaml
```

#### Submit Evaluation Job

Eval runs on a single GPU (model is small enough). Runs both GSM8K and MATH500 sequentially by default.

```bash
# Evaluate a checkpoint on both datasets
sbatch open_r1_imp/scripts/submit_eval.sh --export=ALL,CHECKPOINT=outputs/open_r1_multi_reward_full/checkpoint-200

# Evaluate base model only (no checkpoint)
sbatch open_r1_imp/scripts/submit_eval.sh

# Majority voting (16 samples, temperature=0.6)
sbatch open_r1_imp/scripts/submit_eval.sh --export=ALL,CHECKPOINT=outputs/open_r1_multi_reward_full/checkpoint-200,N_SAMPLES=16,TEMPERATURE=0.6

# Quick test (50 samples)
sbatch open_r1_imp/scripts/submit_eval.sh --export=ALL,CHECKPOINT=outputs/open_r1_multi_reward_full/checkpoint-200,MAX_SAMPLES=50

# Thinking mode eval
sbatch open_r1_imp/scripts/submit_eval.sh --export=ALL,CHECKPOINT=outputs/open_r1_multi_reward_thinking/checkpoint-100,THINKING=true
```

#### Parallel Evaluation (2 separate GPU jobs)

```bash
sbatch open_r1_imp/scripts/submit_eval.sh --export=ALL,CHECKPOINT=outputs/open_r1_multi_reward_full/checkpoint-200,DATASETS=gsm8k
sbatch open_r1_imp/scripts/submit_eval.sh --export=ALL,CHECKPOINT=outputs/open_r1_multi_reward_full/checkpoint-200,DATASETS=math500
```

Each `sbatch` is an independent SLURM job with its own GPU. Results saved to `eval_results_h200/`.

#### Monitor Jobs

```bash
squeue -u $USER          # list your running/queued jobs
scancel <job_id>         # cancel a job
cat logs/eval-<job_id>.out  # check eval output
cat logs/slurm-<job_id>.out # check training output
```

## GPU Configuration

The number of training GPUs is controlled from the launch script:

```bash
# 2 GPUs: 1 vLLM + 1 training
bash open_r1_imp/scripts/run_multi_reward.sh --mode lora --num_train_gpus 1

# 4 GPUs: 1 vLLM + 3 training
bash open_r1_imp/scripts/run_multi_reward.sh --mode lora --num_train_gpus 3

# 8 GPUs: 1 vLLM + 7 training (matches Intuitor paper)
bash open_r1_imp/scripts/run_multi_reward.sh --mode full --num_train_gpus 7
```

### Effective Batch Size

```
unique_prompts = (per_device_batch × num_train_gpus × ga_steps) / num_generations
```

| GPUs | per_device | ga_steps | num_gen | Unique prompts |
|------|-----------|----------|---------|----------------|
| 7 (paper) | 4 | 32 | 7 | **128** |
| 3 | 4 | 32 | 7 | **54** |
| 3 | 10 | 30 | 7 | **128** (match paper) |
| 1 | 4 | 32 | 7 | **18** |

### Parallel Experiments (4-GPU cluster)

```bash
# Experiment 1: GPU 0 (vLLM) + GPUs 1,2 (training)
bash open_r1_imp/scripts/run_multi_reward.sh --mode lora --num_train_gpus 2 \
    --run_name exp1_beta005 --beta 0.005

# Experiment 2: GPU 3 (vLLM) + GPU 4 would need manual setup
# (launch script assumes GPU 0 for vLLM)
```

## How It Works

### One Optimizer Step

```
1. GENERATE (vLLM, GPU 0)
   Main process sends 27 prompts to vLLM → 189 completions (27 × 7 gens)
   Broadcasts completions to all training GPUs

2. SCORE (GPUs 1-3, parallel)
   Each GPU scores ~63 completions: cluster_voting + boxed_format
   Gather rewards → z-score advantages across all GPUs

3. SELF-CERTAINTY (GPUs 1-3, parallel)
   Chunked forward pass (no grad) → logsumexp - mean per token
   Thinking mode: only on <think>...</think> tokens
   Penalize short completions (< 50 tokens)

4. GDPO NORMALIZE
   ch_cf = TRL's z-scored cluster+format advantages
   ch_sc = z-score(self_certainty) per prompt group
   advantages = 0.5 × ch_cf + 0.5 × ch_sc

5. TRAIN (32 gradient accumulation steps)
   Each step: 2 completions per GPU → forward → loss → backward
   Loss = clipped_surrogate + β × KL(ref||current)
   After 32 steps: all-reduce gradients → optimizer.step()

6. SYNC to vLLM
   Gather updated weights → send to vLLM server
```

### LoRA vs Full Fine-Tuning

| | LoRA | Full FT |
|---|---|---|
| **DeepSpeed** | ZeRO-2 (model replicated) | ZeRO-3 (model sharded) |
| **Trainable params** | ~100M (rank 64) | All 1.5B |
| **Reference model** | `disable_adapter()` trick | Separate frozen copy |
| **Memory per GPU** | ~20GB (1.5B model) | ~40GB (1.5B model) |
| **Config** | `config_math_lora.yaml` | `config_math_full.yaml` |
| **Accelerate** | `zero2.yaml` | `zero3.yaml` |

### Thinking Mode

When `--thinking` is enabled (requires Qwen3):

```
Completion: <think>Let me solve 2+3. The sum is 5.</think>The answer is \boxed{5}
                    ├── reasoning tokens ──────────┤├── answer tokens ────────┤

Self-certainty:  computed on reasoning tokens only
Cluster voting:  computed on answer text only (after </think>)
Format reward:   computed on answer text only
GRPO loss:       applied to ALL completion tokens
```

## KL-Cov Regularization

KL-Cov is an exploration mechanism that selectively removes the PPO/GRPO clip on the most important tokens, replacing it with a softer KL penalty.

### The Problem: Clipping Causes Mode Collapse

In standard GRPO, the clipped surrogate loss treats all tokens equally:

```
loss = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

Once the policy ratio exceeds the clip boundary (e.g., 1.2), the gradient becomes **zero** — the model stops learning for that token. This prevents large policy updates, which is good for stability but bad for exploration. Over time, the model finds one solution path, clipping locks it in, and it collapses into repeating that path.

### How KL-Cov Works

**Step 1: Identify decision-critical tokens via covariance**

```
Cov(y_i) = (log π_i - mean(log π)) × (A_i - mean(A))
```

Tokens with high covariance are the ones where changing their probability changes the reward the most — the "fork in the road" tokens that determine if a solution is correct.

**Step 2: Remove clipping on top-k% high-covariance tokens**

```
Normal tokens (98%):  loss = -min(ratio * A, clip(ratio) * A)     # standard clipped
High-cov tokens (2%): loss = -ratio * A + β_cov × |log(π/π_old)| # unclipped + KL penalty
```

Without the clip, the model can make larger policy updates on exactly the tokens that drive reward — exploring diverse solution strategies instead of getting stuck.

**Step 3: Soft KL penalty as a brake**

The KL penalty `β_cov × |log(π/π_old)|` penalizes large policy changes proportionally rather than hard-capping them. This is a softer constraint than clipping — it allows exploration but discourages the policy from changing too drastically.

### Why Unclipped + KL Penalty Instead of Just Clipping

| | Clipped (standard GRPO) | Unclipped + KL penalty (KL-Cov) |
|---|---|---|
| Gradient for key tokens | Hard wall at clip boundary (zero beyond) | Soft penalty, gradients can be large |
| Exploration | Conservative — locks into first good path | Aggressive at decision points |
| Stability | Very stable | Can cause high grad_norm (use `max_grad_norm`) |

### Concrete Example

Solving `3x + 5 = 20`:

```
Path A: "subtract 5 → 3x = 15 → x = 5"         (common strategy)
Path B: "divide by 3 → x + 5/3 = 20/3 → x = 5" (alternative strategy)
```

Without KL-Cov, the model discovers Path A, clipping locks it in, and it always generates Path A. If a harder problem needs Path B's approach, it fails.

With KL-Cov, the tokens at the "subtract vs divide" decision point are high-covariance. The clip is removed there, so the model keeps exploring both strategies, leading to better generalization on unseen problems.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta_cov` | 0.0 | KL-Cov penalty coefficient. 0 = disabled. Typical: 0.05 |
| `top_k_fraction` | 0.02 | Fraction of tokens treated as high-covariance (top 2%) |
| `max_grad_norm` | 1.0 | Gradient clipping to stabilize unclipped tokens |

### Empirical Results (Qwen2.5-1.5B, math-lighteval)

| Method | GSM8K | MATH500 |
|--------|-------|---------|
| Multi-reward (no KL-Cov) | 0.6702 | 0.4356 |
| Multi-reward (KL-Cov β=0.05) | 0.6884 | 0.4560 |
| Improvement | +1.8% | +2.0% |

Consistent ~2% improvement across both benchmarks, suggesting the model learns more diverse solution strategies rather than overfitting to one approach.

## Configuration Reference

### Training Config (MultiRewardGRPOConfig)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `w_cluster_format` | 0.5 | GDPO weight: cluster+format channel |
| `w_selfcert` | 0.5 | GDPO weight: self-certainty channel |
| `beta_cov` | 0.0 | KL-Cov penalty (0 = disabled) |
| `top_k_fraction` | 0.02 | Fraction of high-cov tokens |
| `entropy_bonus` | 0.0 | Entropy exploration bonus (0 = disabled) |
| `thinking` | false | SC on reasoning only, rewards on answer only |
| `min_completion_tokens` | 50 | Anti-template-collapse filter |
| `system_prompt` | null | System prompt for all conversations |
| `beta` | 0.005 | KL penalty vs reference model |
| `temperature` | 1.0 | Sampling temperature |
| `num_generations` | 7 | Completions per prompt |
| `max_completion_length` | 3072 | Max tokens per completion |
| `learning_rate` | 3e-6 | Peak learning rate |
| `lr_scheduler_type` | cosine | LR schedule |

### Script Args (MultiRewardScriptArguments)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reward_funcs` | [cluster_voting, boxed_format] | Reward function names |
| `reward_weights` | [0.0, 0.0] | TRL weights (0 = GDPO handles combination) |
| `w_cluster` | 0.5 | Cluster weight within ch_cf channel |
| `w_format` | 0.1 | Format weight within ch_cf channel |
| `use_semantic_clustering` | false | Enable DeBERTa EMPO clustering |
| `deberta_model` | microsoft/deberta-v3-large | NLI model for EMPO |
| `dataset_prompt_column` | problem | Dataset column with the prompt text |

## Evaluation

Supports LoRA checkpoints, full fine-tuned models, and thinking mode. Auto-detects checkpoint type: if `adapter_config.json` exists → LoRA, otherwise → full FT. Override with `--mode lora` or `--mode full`.

### Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| **pass@1** | Greedy accuracy (1 completion, temperature=0). "If the model gets one shot, how often is it right?" |
| **majority@N** | Majority vote accuracy (N completions, temperature>0, pick most common answer). "With N attempts, how often is the consensus correct?" |

### Evaluate a Single Checkpoint

```bash
# LoRA checkpoint — local path (auto-detected)
python evaluation/eval_checkpoint.py \
    --checkpoint outputs/open_r1_multi_reward_lora/checkpoint-50 \
    --dataset math500

# Full fine-tuned model — local path
python evaluation/eval_checkpoint.py \
    --checkpoint outputs/open_r1_multi_reward_full/checkpoint-50 \
    --dataset gsm8k

# Thinking mode (Qwen3)
python evaluation/eval_checkpoint.py \
    --checkpoint outputs/open_r1_multi_reward_thinking/checkpoint-50 \
    --thinking --dataset math500 --max_new_tokens 4096

# Base model comparison
python evaluation/eval_checkpoint.py --base_only --dataset math500

# Quick test (50 samples)
python evaluation/eval_checkpoint.py \
    --checkpoint outputs/checkpoint-50 --max_samples 50

# Majority vote (pass@16)
python evaluation/eval_checkpoint.py \
    --checkpoint outputs/checkpoint-50 --n_samples 16 --temperature 0.6
```

### Evaluate from HuggingFace Hub (Remote Checkpoints)

Checkpoints pushed to HF Hub can be evaluated from any machine — no need for direct access to the training server.

```bash
# Eval latest checkpoint from HF repo
python evaluation/eval_checkpoint.py \
    --checkpoint shourovj/multi-reward-grpo \
    --dataset math500

# Eval a specific checkpoint (by branch name)
python evaluation/eval_checkpoint.py \
    --checkpoint shourovj/multi-reward-grpo \
    --revision exp1_lora_math-step-100 \
    --dataset math500

# Eval a different run's checkpoint from the same repo
python evaluation/eval_checkpoint.py \
    --checkpoint shourovj/multi-reward-grpo \
    --revision exp2_full_ft-step-50 \
    --mode full \
    --dataset math500 gsm8k

# List available branches (checkpoints) in a repo
huggingface-cli repo info shourovj/multi-reward-grpo
```

### HuggingFace Hub Push (Training on Remote Cluster)

When training on a cluster without direct access, checkpoints are automatically pushed to HF Hub every N steps. Each checkpoint goes to a unique branch.

```yaml
# In recipe YAML:
push_to_hub: true
hub_model_id: shourovj/multi-reward-grpo    # your HF repo
hub_private_repo: true
hub_push_every_n_steps: 50                  # push every 50 steps (fixed)
save_steps: 10                              # local save every 10 steps (configurable)
run_name: exp1_lora_math                    # prefix for branch names
```

**HF repo structure after training:**
```
shourovj/multi-reward-grpo
├── main                              ← final model
├── exp1_lora_math-step-50            ← checkpoint at step 50
├── exp1_lora_math-step-100           ← checkpoint at step 100
├── exp1_lora_math-step-150           ← checkpoint at step 150
├── exp2_full_ft-step-50              ← different experiment, same repo
├── exp2_full_ft-step-100
└── ...
```

**Override repo or run name from CLI:**
```bash
bash open_r1_imp/scripts/run_multi_reward.sh --mode lora --num_train_gpus 3 \
    --hub_model_id shourovj/my-new-experiment \
    --run_name ablation_beta001
```

**Required setup on cluster (one-time):**
```bash
huggingface-cli login --token hf_your_token_here
# Or set in submit.sh:
export HF_TOKEN="hf_your_token_here"
```

### Async Evaluation During Training (Intuitor-style)

Watches the training output directory for new checkpoints and evaluates them automatically on a separate GPU. Training never pauses.

```
┌──────────────────────────┐         ┌──────────────────────────┐
│  Terminal 1 (GPUs 0-2)   │         │  Terminal 2 (GPU 3)      │
│  Training runs...        │         │  Eval watches...         │
│                          │  save   │                          │
│  checkpoint-10 ──────────│────────>│  detect → eval → log     │
│  checkpoint-20 ──────────│────────>│  detect → eval → log     │
│  checkpoint-30 ──────────│────────>│  detect → eval → log     │
│  ...                     │         │  ...                     │
└──────────────────────────┘         └──────────────────────────┘
```

**Terminal 1** — training:
```bash
bash open_r1_imp/scripts/run_multi_reward.sh --mode lora --num_train_gpus 3
```

**Terminal 2** — async eval on separate GPU:
```bash
CUDA_VISIBLE_DEVICES=3 python evaluation/eval_during_training.py \
    --output_dir outputs/open_r1_multi_reward_lora \
    --dataset math500 gsm8k \
    --max_samples 200 \
    --poll_interval 120
```

**With thinking mode:**
```bash
CUDA_VISIBLE_DEVICES=3 python evaluation/eval_during_training.py \
    --output_dir outputs/open_r1_multi_reward_thinking \
    --dataset math500 \
    --thinking --max_new_tokens 4096 \
    --poll_interval 120
```

**One-shot** — eval all existing checkpoints after training:
```bash
python evaluation/eval_during_training.py \
    --output_dir outputs/open_r1_multi_reward_lora \
    --dataset math500 gsm8k \
    --no_watch
```

**With wandb logging** (separate wandb run from training):
```bash
CUDA_VISIBLE_DEVICES=3 python evaluation/eval_during_training.py \
    --output_dir outputs/open_r1_multi_reward_lora \
    --dataset math500 gsm8k \
    --wandb_project multi-reward-eval \
    --wandb_run_name eval_lora_math
```

**Output summary:**
```
EVALUATION SUMMARY
============================================================

MATH500:
    Step    pass@1   correct
  ------  --------  --------
      10    0.0120     6/500
      20    0.0180     9/500
      30    0.0240    12/500
      50    0.0380    19/500

GSM8K:
    Step    pass@1   correct
  ------  --------  --------
      10    0.0150    20/1319
      20    0.0280    37/1319
      30    0.0410    54/1319
      50    0.0620    82/1319
```

Results saved as JSON: `outputs/.../eval_results/eval_step_10.json`, `eval_step_20.json`, etc.

### Eval Script Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--output_dir` | required | Training output dir with checkpoint-* folders |
| `--model_name` | auto-detect | Base model name (for LoRA checkpoints) |
| `--dataset` | math500 | Space-separated: `math500 gsm8k math` |
| `--max_samples` | all | Limit samples per dataset (for speed) |
| `--n_samples` | 1 | Completions per question (majority vote if >1) |
| `--temperature` | 0.0 | 0=greedy, >0 for majority voting |
| `--thinking` | off | Enable Qwen3 thinking mode |
| `--poll_interval` | 60 | Seconds between checkpoint checks |
| `--no_watch` | off | Eval existing checkpoints and exit |
| `--wandb_project` | none | Log eval results to separate wandb project |

## WandB Metrics

| Metric | Description |
|--------|-------------|
| `rewards/self_certainty` | Mean self-certainty across batch |
| `rewards/cluster_voting_reward/mean` | Mean cluster voting reward |
| `rewards/boxed_format_reward/mean` | Fraction with valid `\boxed{}` |
| `advantages/gdpo_combined_std` | Std of final GDPO advantages |
| `advantages/ch_sc_std` | Std of z-scored SC channel |
| `loss` | Training loss |
| `grad_norm` | Gradient norm before clipping |
| `kl` | KL divergence from reference |
| `entropy` | Token-level policy entropy |
| `klcov/high_cov_fraction` | Fraction flagged as high-cov (if enabled) |
| `reasoning/frac_with_think` | Fraction with `<think>` tags (thinking mode) |
| `reasoning/avg_reasoning_tokens` | Mean reasoning length (thinking mode) |

## Overriding Config from CLI

Any YAML parameter can be overridden:

```bash
bash open_r1_imp/scripts/run_multi_reward.sh --mode lora --num_train_gpus 3 \
    --model Qwen/Qwen2.5-3B \
    --learning_rate 1e-5 \
    --beta 0.01 \
    --max_steps 200 \
    --w_cluster_format 0.3 \
    --w_selfcert 0.7 \
    --run_name my_experiment \
    --wandb_project my-project
```
