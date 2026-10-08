# ICLR 2026 — Tier 1 (Llama) + Tier 2 (TriviaQA) Run Plan (Bridges-2)

Everything runs on THIS cluster via SLURM (`GPU-shared`, account `cis260218p`).
Infrastructure is the proven NeurIPS-rebuttal stack: `launch.sh` →
`submit_bridges2.sh` (GPU 0 = `trl vllm-serve`, rest = ZeRO-3 training).
Budget check 2026-08-31: **1,218 GPU-SU free**; full program ≈ 220 SU.

## One-time prereqs
1. **HF gating (B6 only):** accept the Meta license for `meta-llama/Llama-3.2-3B`
   on huggingface.co, then `export HF_TOKEN=hf_...` in the shell you `sbatch` from
   (submit scripts pass it through; never hardcode it).
2. `wandb login` once (or `export WANDB_MODE=offline`).
3. Optional pre-download so the job doesn't spend queue time downloading:
   `python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.2-3B')"`
   (with HF_HOME=${HF_HOME}, already set by the scripts).

## Tier 1 — B6 Llama-3.2-3B (2 arms now, 3rd if time)
Protocol = the paper's Qwen3-1.7B suite exactly (mu=2, effective batch 882,
200 steps, save every 20, seed 7). New vs the old B6 draft: **mu=2 fix**
(KL-Cov was a silent no-op at mu=1) and a **base-model chat template**
(Llama base ships none; template is saved into checkpoints, so eval inherits it).

```bash
export HF_TOKEN=hf_...
GPUS=4 bash open_r1_imp/scripts/rebuttal/launch.sh B6_llama32_3b_full
GPUS=4 bash open_r1_imp/scripts/rebuttal/launch.sh B6_llama32_3b_intuitor
# optional 3rd arm, only if time remains:
GPUS=4 bash open_r1_imp/scripts/rebuttal/launch.sh B6_llama32_3b_grpo_gt
```
B6 needs the 4-GPU shape (3B + ZeRO-3); `launch.sh` rejects GPUS=2 for B6.
Default GPU_TYPE=h100-80 (~14 h wall each, ~48 SU); `GPU_TYPE=l40s-48` if the
H100 queue is jammed (slower, cheaper/SU-equal). ~96–144 SU total.

**Risk plan (unchanged from ICLR-TODO):** Llama-3.2-3B base is weaker at
math/boxed-format — a cluster-channel cold start is itself reportable (it tests
the bootstrap-relaxation claim scoped in Limitations). Watch the first ~20 steps
in wandb: `rewards/cluster_voting` stuck at ~0 + near-zero format rate = cold
start. Fallback backbones: Llama-3.1-8B (needs GPUS=4 h100) or Gemma-2-9B.

## Tier 1 evaluation (after each arm finishes, or on partial checkpoints)
```bash
sbatch open_r1_imp/scripts/rebuttal/submit_eval_bridges2.sh          # b6 suite, gsm8k+math500, all ckpts + base row
# MMLU-Pro at step 200 only (A2 protocol: subsample WITH a seed):
sbatch --export=ALL,TASKS=mmlu_pro,STEPS=200,MAX_SAMPLES=2000,SAMPLE_SEED=7,BASE_MODEL=meta-llama/Llama-3.2-3B \
       open_r1_imp/scripts/rebuttal/submit_eval_bridges2.sh
```
Idempotent (skips existing JSONs). Results land in `eval-rebuttal-results/` —
same format the significance/plot tooling already consumes. Then:
- Table 2 (`tab:scaling`) gains the Llama row: retention = end/peak from the sweep.
- `evaluation/significance.py` pattern extends to the b6 records for McNemar/CIs.

## Tier 2 — T2 TriviaQA label-free probe (only if a card is free after Tier 1)
Unlabeled training on rc.nocontext *questions*; gold aliases used at EVAL time
only (`evaluation/tasks/triviaqa.py`, EM, boxed-else-last-line prediction).
```bash
bash open_r1_imp/scripts/rebuttal/launch.sh T2_triviaqa_full_qwen25_15b
bash open_r1_imp/scripts/rebuttal/launch.sh T2_triviaqa_intuitor_qwen25_15b
# eval (validation EM, 2000-question seeded subset keeps it cheap):
sbatch --export=ALL,RUNS="t2_full:T2_triviaqa_full_qwen25_15b t2_intuitor:T2_triviaqa_intuitor_qwen25_15b",TASKS=triviaqa,BASE_MODEL=Qwen/Qwen2.5-1.5B,BASE_NAME=qwen25_15b_base_tqa,MAX_SAMPLES=2000,SAMPLE_SEED=7 \
       open_r1_imp/scripts/rebuttal/submit_eval_bridges2.sh
```
1.5B fits the default 2-GPU shape (~30 SU/arm). max_completion 1024 (short answers).

## Save-everything policy (T0.4 — the missing-checkpoint problem must not recur)
- `save_steps 20`, `save_total_limit 50` → every checkpoint kept, on /ocean.
- Do NOT clean `outputs/rebuttal/B6_*` or `T2_*` until camera-ready + rebuttal end.
- Keep every per-item eval JSON in `eval-rebuttal-results/` (they feed McNemar).
- wandb on (`report_to: wandb`) — the training-signal series feed the
  early-stopping analysis if a reviewer asks for it on the new runs.

## Monitoring a run
`squeue -u $USER`, then `tail -f logs/slurm-<jobid>.out` (vLLM health line
first, then per-step training logs; wandb run name = the yaml `run_name`).
Resume after preemption/timeout: just relaunch the same command — the trainer
auto-resumes from the last checkpoint in output_dir.
