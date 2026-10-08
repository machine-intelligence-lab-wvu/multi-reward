"""
Multi-Reward Unsupervised GRPO — Persistent vLLM (Single 48GB GPU).

Same algorithm as train_from_scratch.py but keeps BOTH vLLM and PyTorch
loaded in GPU memory simultaneously.  No load/unload cycle = no memory
leak, much faster per-step time.

Requires >= 40GB VRAM (tested on RTX PRO 6000, 48GB).

Memory layout (Qwen2.5-1.5B, bf16):
  vLLM:     model 3GB + KV cache ~8GB = ~11GB  (gpu_memory_utilization=0.25)
  PyTorch:  model 3GB + LoRA 0.2GB + optimizer 0.6GB + activations ~10GB = ~14GB
  Total:    ~25GB (fits in 48GB with headroom)

Architecture
────────────
  STARTUP
    Load vLLM (persistent, with LoRA support)
    Load PyTorch model + LoRA + optimizer (persistent)

  FOR EACH STEP
    PHASE 1 — GENERATE  (vLLM, already loaded)
      Generate G completions per prompt using current LoRA
      Extract per-token log-probs (π_old)

    PHASE 2 — TRAIN  (PyTorch, already loaded)
      Pre-compute advantages (self-certainty + GDPO)
      Mini-batch forward + backward
      optimizer.step()
      Save LoRA checkpoint (for vLLM to pick up next step)

Shared functions imported from train_from_scratch.py:
  prepare_dataset, compute_all_rewards, gdpo_normalize, compute_advantages,
  precompute_advantages, prepare_training_batch, forward_with_mean_logprobs,
  compute_grpo_loss, compute_ref_logprobs, aggregate_loss
"""

import os, sys, gc, math, time, logging, argparse, json, random, shutil, glob
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_from_scratch import (
    TrainConfig, parse_args, prepare_dataset,
    compute_all_rewards, gdpo_normalize, compute_advantages,
    precompute_advantages, prepare_training_batch,
    forward_with_mean_logprobs, compute_grpo_loss, compute_ref_logprobs,
    aggregate_loss,
)
from train_thinking import (
    prepare_dataset_thinking, compute_rewards_thinking,
    prepare_training_batch_thinking, precompute_advantages_thinking,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# PERSISTENT TRAINER
# ════════════════════════════════════════════════════════════

class PersistentTrainer:
    """Keeps both vLLM and PyTorch model loaded throughout training.

    Unlike train_from_scratch.py which loads/unloads models each step
    (causing memory leaks and ~60s overhead), this keeps everything
    resident in GPU memory. Requires >= 40GB VRAM.
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if config.thinking:
            logger.info("Thinking mode ENABLED — SC on reasoning only, rewards on answer only")
            self.prompts = prepare_dataset_thinking(
                self.tokenizer, config.dataset,
                max_samples=config.max_train_samples)
        else:
            self.prompts = prepare_dataset(
                self.tokenizer, config.dataset,
                max_samples=config.max_train_samples)

        self.global_step = 0
        self.lora_path: Optional[str] = None
        self._run_number: Optional[int] = None

        # Handle resume
        if config.resume_from:
            ckpt = config.resume_from.strip()
            if not os.path.isdir(ckpt):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
            self.lora_path = ckpt
            ckpt_name = os.path.basename(ckpt.rstrip("/"))
            if ckpt_name.startswith("checkpoint-"):
                self.global_step = int(ckpt_name.split("-")[1])
            parent = os.path.basename(os.path.dirname(ckpt.rstrip("/")))
            if parent.startswith("run_"):
                try:
                    self._run_number = int(parent.split("_")[1])
                except ValueError:
                    pass
            logger.info(f"Resuming from {ckpt}, step={self.global_step}, run={self._run_number}")

        # ── Load PyTorch model (PERSISTENT) ──
        logger.info("Loading PyTorch training model (persistent)...")
        self.model = self._load_training_model()
        self.vocab_size = self.model.config.vocab_size

        # ── Create optimizer + scheduler ──
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable,
                               lr=config.learning_rate,
                               weight_decay=config.weight_decay,
                               betas=(0.9, 0.999))

        total_steps = self._total_steps()
        total_scheduler_steps = total_steps * config.mu
        warmup_steps = int(total_scheduler_steps * config.warmup_ratio)
        schedule_fn = (get_cosine_schedule_with_warmup if config.lr_scheduler == "cosine"
                       else get_linear_schedule_with_warmup)
        self.scheduler = schedule_fn(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_scheduler_steps,
        )

        # Restore optimizer/scheduler state if resuming
        if config.resume_from:
            opt_path = os.path.join(config.resume_from, "optimizer.pt")
            sch_path = os.path.join(config.resume_from, "scheduler.pt")
            if os.path.isfile(opt_path):
                self.optimizer.load_state_dict(
                    torch.load(opt_path, map_location="cuda", weights_only=True))
            if os.path.isfile(sch_path):
                self.scheduler.load_state_dict(
                    torch.load(sch_path, map_location="cpu", weights_only=True))

        # ── Load vLLM (PERSISTENT) ──
        logger.info("Loading vLLM generation engine (persistent)...")
        self.vllm_engine = self._load_vllm()

        # ── wandb ──
        if config.use_wandb:
            import wandb
            wandb_kwargs = dict(
                project=os.environ.get("WANDB_PROJECT", "multi-reward-persistent"),
                name=config.run_name,
                config=asdict(config),
            )
            if config.resume_from:
                wandb_kwargs["resume"] = "allow"
            wandb.init(**wandb_kwargs)

        logger.info(f"Persistent trainer ready. "
                    f"vLLM + PyTorch both loaded on GPU. "
                    f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated, "
                    f"{torch.cuda.memory_reserved()/1e9:.1f}GB reserved")

    def _load_training_model(self):
        """Load base model + LoRA for training. Stays on GPU permanently."""
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda")

        if self.lora_path is not None:
            model = PeftModel.from_pretrained(model, self.lora_path,
                                              is_trainable=True)
        else:
            lora_cfg = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
                lora_dropout=self.config.lora_dropout,
            )
            model = get_peft_model(model, lora_cfg)

        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.train()
        return model

    def _load_vllm(self):
        """Load vLLM engine. Stays loaded permanently alongside PyTorch model."""
        from vllm import LLM

        # Use a smaller fraction of GPU for vLLM since PyTorch also needs memory.
        # With 48GB: 0.25 × 48 = 12GB for vLLM (model 3GB + KV cache 9GB)
        # Remaining ~36GB for PyTorch (model 3GB + optimizer 0.6GB + activations)
        gpu_util = self.config.gpu_memory_utilization

        llm_kwargs = dict(
            model=self.config.model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=self.config.max_prompt_length + self.config.max_completion_length,
            gpu_memory_utilization=gpu_util,
            enforce_eager=True,
            seed=self.config.seed,
            enable_lora=True,
            max_lora_rank=self.config.lora_rank,
        )
        return LLM(**llm_kwargs)

    def _generate_rollouts(self, prompts: List[str]) -> Dict[str, Any]:
        """Generate completions using the persistent vLLM engine.

        Uses LoRARequest to load the latest LoRA weights dynamically.
        vLLM supports hot-swapping LoRA adapters without restarting.
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            n=self.config.num_generations,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_completion_length,
            logprobs=1,
        )

        lora_request = None
        if self.lora_path is not None:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest("active_lora", 1, self.lora_path)

        outputs = self.vllm_engine.generate(prompts, sampling_params,
                                            lora_request=lora_request)

        result: Dict[str, Any] = {
            "prompt_texts": [],
            "completion_texts": [],
            "completion_token_ids": [],
            "old_per_token_logps": [],
            "prompt_token_ids": [],
        }

        for req_output in outputs:
            result["prompt_texts"].append(req_output.prompt)
            result["prompt_token_ids"].append(list(req_output.prompt_token_ids))

            g_texts, g_ids, g_logps = [], [], []
            for comp in req_output.outputs:
                g_texts.append(comp.text)
                g_ids.append(list(comp.token_ids))

                token_logps = []
                if comp.logprobs is not None:
                    for t, lp_dict in enumerate(comp.logprobs):
                        tok_id = comp.token_ids[t]
                        if lp_dict and tok_id in lp_dict:
                            token_logps.append(lp_dict[tok_id].logprob)
                        else:
                            token_logps.append(0.0)
                g_logps.append(token_logps)

            result["completion_texts"].append(g_texts)
            result["completion_token_ids"].append(g_ids)
            result["old_per_token_logps"].append(g_logps)

        return result

    def _total_steps(self) -> int:
        if self.config.max_steps > 0:
            return self.config.max_steps
        n_prompts = len(self.prompts)
        steps_per_epoch = max(1, n_prompts // (self.config.train_batch_size * self.config.gradient_accumulation_steps))
        return steps_per_epoch * self.config.num_train_epochs

    def _save_checkpoint(self, step: int, permanent: bool = False):
        if permanent:
            if self._run_number is None:
                base_output = self.config.output_dir
                os.makedirs(base_output, exist_ok=True)
                existing_runs = glob.glob(os.path.join(base_output, "run_*"))
                run_numbers = [int(r.split("_")[-1]) for r in existing_runs
                               if r.split("_")[-1].isdigit()]
                self._run_number = max(run_numbers) + 1 if run_numbers else 0
                logger.info(f"  Starting run_{self._run_number}")

            run_dir = os.path.join(self.config.output_dir, f"run_{self._run_number}")
            save_dir = os.path.join(run_dir, f"checkpoint-{step}")
        else:
            save_dir = os.path.join(self.config.output_dir, "latest_lora_adapter")

        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        torch.save(self.optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))

        self.lora_path = save_dir
        if permanent:
            logger.info(f"  Saved checkpoint → {save_dir}")

    def _log_step(self, step: int, metrics: Dict[str, float]):
        parts = [f"step={step}"]
        key_order = ["loss", "grad_norm", "lr", "kl", "entropy",
                     "rewards/cluster_mean", "rewards/format_mean",
                     "rewards/self_certainty_mean",
                     "advantages/combined_mean",
                     "advantages/ch_cluster_format_std",
                     "advantages/ch_self_certainty_std",
                     "klcov/high_cov_fraction",
                     "klcov/mean_kl_penalty",
                     "step_time"]
        for k in key_order:
            if k in metrics:
                v = metrics[k]
                parts.append(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}")
        logger.info("  ".join(parts))

        if self.config.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def _print_rollouts(self, rollouts: Dict, rewards: Dict, step: int):
        from rewards.answer_extraction import extract_boxed_answer
        P = len(rollouts["completion_texts"])
        G = self.config.num_generations
        show_prompts = min(2, P)

        lines = ["", "=" * 80, f"  ROLLOUTS  (step {step})", "=" * 80]
        for p in range(show_prompts):
            prompt = rollouts["prompt_texts"][p]
            if len(prompt) > 300:
                prompt = prompt[:150] + " … " + prompt[-100:]
            lines += [f"\n{'─'*70}", f"PROMPT [{p}]: {prompt}", f"{'─'*70}"]

            for g in range(min(G, 4)):
                text = rollouts["completion_texts"][p][g]
                flat = p * G + g
                r_cl = rewards["cluster_rewards"][flat].item()
                r_fm = rewards["format_rewards"][flat].item()
                boxed = extract_boxed_answer(text)
                bstr = f"  \\boxed{{{boxed}}}" if boxed else "  [no \\boxed{}]"
                if len(text) > 500:
                    text = text[:250] + "\n  … [truncated] …\n  " + text[-200:]
                lines.append(
                    f"\n  ROLLOUT [{g}]  R_cluster={r_cl:.3f}  "
                    f"R_format={r_fm:.1f}{bstr}")
                lines.append(f"  {text}")
            if G > 4:
                lines.append(f"\n  … ({G - 4} more rollouts not shown)")
        lines.append(f"\n{'=' * 80}")
        logger.info("\n".join(lines))

    # ─── MAIN TRAINING LOOP ──────────────────────────────
    def train(self):
        total_steps = self._total_steps()
        config = self.config
        logger.info(f"=== Multi-Reward GRPO — Persistent vLLM ===")
        logger.info(f"Config: {asdict(config)}")
        logger.info(f"Total prompts: {len(self.prompts)}, total steps: {total_steps}")

        os.makedirs(config.output_dir, exist_ok=True)

        prompt_indices = list(range(len(self.prompts)))
        random.seed(config.seed)
        random.shuffle(prompt_indices)
        cursor = 0
        start_step = self.global_step

        # Fast-forward cursor for resume
        if start_step > 0:
            bs = config.train_batch_size
            prompts_per_step = bs * config.gradient_accumulation_steps
            for _ in range(start_step):
                cursor += prompts_per_step
                while cursor >= len(prompt_indices):
                    random.shuffle(prompt_indices)
                    cursor -= len(prompt_indices)
            logger.info(f"Skipped {start_step} steps, resuming from step {start_step}")

        ga_steps = config.gradient_accumulation_steps
        mu = config.mu
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler

        for step in range(start_step, total_steps):
            step_start = time.time()
            bs = config.train_batch_size

            step_metrics: Dict[str, List[float]] = defaultdict(list)
            all_full_batch_metrics: List[Dict[str, float]] = []

            # ════════════════════════════════════════════════════
            # PHASE 1: GENERATE all GA rounds ONCE
            # ════════════════════════════════════════════════════
            ga_round_data: List[Dict[str, Any]] = []

            for ga_idx in range(ga_steps):
                if cursor + bs > len(prompt_indices):
                    random.shuffle(prompt_indices)
                    cursor = 0
                batch_indices = prompt_indices[cursor:cursor + bs]
                cursor += bs
                batch_prompts = [self.prompts[i] for i in batch_indices]

                logger.info(f"\n\n Step {step}/{total_steps} "
                            f"[ga {ga_idx+1}/{ga_steps}]: generating "
                            f"{bs}×{config.num_generations} completions …")

                # No model swap needed — vLLM is always loaded!
                gen_start = time.time()
                rollouts = self._generate_rollouts(batch_prompts)
                gen_time = time.time() - gen_start
                logger.info(f"  Generation took {gen_time:.1f}s")

                # Branch: thinking mode uses answer-only rewards + reasoning-only SC
                if config.thinking:
                    rewards = compute_rewards_thinking(rollouts, config)
                else:
                    rewards = compute_all_rewards(rollouts, config)

                if (config.print_rollouts_every > 0
                        and step % config.print_rollouts_every == 0
                        and ga_idx == 0):
                    self._print_rollouts(rollouts, rewards, step)

                if config.thinking:
                    training_batch = prepare_training_batch_thinking(
                        rollouts, rewards, self.tokenizer, config)
                else:
                    training_batch = prepare_training_batch(
                        rollouts, rewards, self.tokenizer, config)
                del rollouts

                device = torch.device("cuda")
                for k, v in training_batch.items():
                    if isinstance(v, torch.Tensor):
                        training_batch[k] = v.to(device)

                B = training_batch["input_ids"].shape[0]

                if config.thinking:
                    precomputed_full = precompute_advantages_thinking(
                        model,
                        training_batch["input_ids"],
                        training_batch["attention_mask"],
                        training_batch["completion_mask"],
                        training_batch["reasoning_mask"],
                        training_batch["cluster_rewards"],
                        training_batch["format_rewards"],
                        training_batch["prompt_group_idx"],
                        training_batch["max_comp_len"],
                        self.vocab_size,
                        config,
                    )
                else:
                    precomputed_full = precompute_advantages(
                        model,
                        training_batch["input_ids"],
                        training_batch["attention_mask"],
                        training_batch["completion_mask"],
                        training_batch["cluster_rewards"],
                        training_batch["format_rewards"],
                        training_batch["prompt_group_idx"],
                        training_batch["max_comp_len"],
                        self.vocab_size,
                        config,
                    )

                batch_metrics = {
                    "rewards/self_certainty_mean": precomputed_full["self_certainty"].mean().item(),
                    "advantages/ch_cluster_format_std": (
                        precomputed_full["ch_cf"].std(correction=0).item()
                        if precomputed_full["ch_cf"].numel() > 1 else 0.0),
                    "advantages/ch_self_certainty_std": (
                        precomputed_full["ch_sc"].std(correction=0).item()
                        if precomputed_full["ch_sc"].numel() > 1 else 0.0),
                    "advantages/combined_mean": precomputed_full["advantages"].mean().item(),
                }
                if config.thinking:
                    r_mask = training_batch["reasoning_mask"]
                    batch_metrics["reasoning/frac_with_think"] = (
                        (r_mask.sum(dim=-1) > 0).float().mean().item())
                    batch_metrics["reasoning/avg_reasoning_tokens"] = (
                        r_mask.sum(dim=-1).mean().item())
                all_full_batch_metrics.append(batch_metrics)

                ga_round_data.append({
                    "training_batch": training_batch,
                    "precomputed": precomputed_full,
                    "B": B,
                })

            # ════════════════════════════════════════════════════
            # PHASE 2: TRAIN for μ iterations over the SAME data
            # ════════════════════════════════════════════════════
            mini_bs = 1
            actual_ga = ga_round_data[0]["B"] * ga_steps

            for mu_iter in range(mu):
                if mu > 1:
                    logger.info(f"    GRPO iteration {mu_iter + 1}/{mu}")

                optimizer.zero_grad()

                for ga_idx, rd in enumerate(ga_round_data):
                    training_batch = rd["training_batch"]
                    precomputed_full = rd["precomputed"]
                    B = rd["B"]

                    for mb_start in range(0, B, mini_bs):
                        mb_end = min(mb_start + mini_bs, B)
                        mb: Dict[str, Any] = {}
                        for k, v in training_batch.items():
                            if isinstance(v, torch.Tensor) and v.shape[0] == B:
                                mb[k] = v[mb_start:mb_end]
                            else:
                                mb[k] = v
                        mb["max_comp_len"] = training_batch["max_comp_len"]

                        mb_precomputed = {
                            k: v[mb_start:mb_end] for k, v in precomputed_full.items()
                        }

                        ref_logps = compute_ref_logprobs(
                            model, mb["input_ids"], mb["attention_mask"],
                            mb["max_comp_len"])

                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            loss, metrics = compute_grpo_loss(
                                model, mb, config, self.vocab_size,
                                ref_per_token_logps=ref_logps,
                                gradient_accumulation_steps=actual_ga,
                                precomputed=mb_precomputed,
                            )

                        loss.backward()
                        del loss, ref_logps, mb_precomputed, mb

                        for k, v in metrics.items():
                            step_metrics[k].append(v)
                        del metrics

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.max_grad_norm,
                ).item()

                optimizer.step()
                scheduler.step()

            # Free GA round data
            del ga_round_data
            gc.collect()
            torch.cuda.empty_cache()

            # ── Save checkpoint ──
            self._save_checkpoint(step + 1,
                                  permanent=((step + 1) % config.save_steps == 0))

            # ── Logging ──
            current_lr = scheduler.get_last_lr()[0]
            avg_metrics = {k: sum(v) / len(v) for k, v in step_metrics.items()}
            full_batch_metrics = {
                k: sum(d[k] for d in all_full_batch_metrics) / len(all_full_batch_metrics)
                for k in all_full_batch_metrics[0]
            }
            avg_metrics.update(full_batch_metrics)
            avg_metrics["grad_norm"] = grad_norm
            avg_metrics["lr"] = current_lr
            avg_metrics["step_time"] = time.time() - step_start

            free_mem, total_mem = torch.cuda.mem_get_info()
            avg_metrics["gpu/used_gb"] = (total_mem - free_mem) / 1e9
            avg_metrics["gpu/free_gb"] = free_mem / 1e9

            if step % config.logging_steps == 0:
                self._log_step(step, avg_metrics)

            self.global_step = step + 1

        # ── Final save ──
        final_dir = os.path.join(config.output_dir, "final")
        if self.lora_path and self.lora_path != final_dir:
            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)
            shutil.copytree(self.lora_path, final_dir)
            logger.info(f"Final model saved → {final_dir}")

        logger.info("Training complete!")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    config = parse_args()
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    trainer = PersistentTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
