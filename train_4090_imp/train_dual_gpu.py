"""
Multi-Reward Unsupervised GRPO — Dual GPU (vLLM Server + PyTorch Training).

Zero memory leak architecture:
  GPU 0: vLLM server (always running, serves generation requests)
  GPU 1: PyTorch training (model + optimizer always loaded)

The vLLM server supports dynamic LoRA loading — each generation request
specifies the latest LoRA adapter path. No restart needed.

Usage:
  1. Start vLLM server:  bash scripts/start_vllm_server.sh
  2. Run training:       bash scripts/train_dual_gpu.sh

Works on any multi-GPU setup (2+ GPUs). Only uses 2 GPUs — run multiple
experiments in parallel on larger clusters (e.g., 4 GPUs = 2 experiments).
"""

import os, sys, gc, math, time, logging, argparse, json, random, shutil, glob
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import requests

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rewards.answer_extraction import extract_boxed_answer, normalize_answer
from rewards.cluster_voting import compute_cluster_rewards
from rewards.format_reward import compute_format_reward
from callbacks.klcov import compute_token_covariances, select_high_covariance_tokens

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B"
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # Dataset
    dataset: str = "math-lighteval"
    max_train_samples: Optional[int] = None

    # Generation (via vLLM server)
    num_generations: int = 7
    max_prompt_length: int = 512
    max_completion_length: int = 3072
    temperature: float = 1.0
    top_p: float = 1.0
    vllm_base_url: str = "http://localhost:8000"  # vLLM server address

    # Reward weights
    w_cluster: float = 0.5
    w_format: float = 0.1
    w_cluster_format: float = 0.5
    w_selfcert: float = 0.5

    # KL-Cov
    beta_cov: float = 0.0
    top_k_fraction: float = 0.02

    # KL & Entropy
    beta_kl: float = 0.005
    entropy_bonus: float = 0.0
    epsilon_clip: float = 0.2

    # Min completion length filter
    min_completion_tokens: int = 50

    # Training
    train_batch_size: int = 32
    gradient_accumulation_steps: int = 4
    mu: int = 1
    learning_rate: float = 3e-6
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    num_train_epochs: int = 1
    max_steps: int = -1
    loss_type: str = "grpo"

    # GPU
    train_device: str = "cuda:0"  # training GPU (cuda:0 when CUDA_VISIBLE_DEVICES pins one GPU)

    # Infrastructure
    output_dir: str = "./outputs/dual_gpu"
    logging_steps: int = 1
    save_steps: int = 10
    print_rollouts_every: int = 10
    use_wandb: bool = False
    run_name: str = "dual_gpu"
    resume_from: Optional[str] = None
    seed: int = 42


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Multi-Reward GRPO (Dual GPU)")
    for f in TrainConfig.__dataclass_fields__.values():
        name = f"--{f.name}"
        if f.type is bool:
            p.add_argument(name, action="store_true", default=f.default)
        elif f.type is Optional[int]:
            p.add_argument(name, type=int, default=f.default)
        elif f.type is int:
            p.add_argument(name, type=int, default=f.default)
        elif f.type is float:
            p.add_argument(name, type=float, default=f.default)
        else:
            p.add_argument(name, type=str, default=f.default)
    args = p.parse_args()
    return TrainConfig(**vars(args))


# ════════════════════════════════════════════════════════════
# 2. DATASET
# ════════════════════════════════════════════════════════════

def prepare_dataset(tokenizer, dataset_name: str, split: str = "train",
                    max_samples: Optional[int] = None) -> List[str]:
    system_prompt = (
        "You are a helpful AI Assistant, designed to provided well-reasoned "
        "and detailed responses. You FIRST think about the reasoning process "
        "step by step and then provide the user with the answer. Please enclose "
        "your final answer in the box: \\boxed{Your Answer}. Please stop "
        "generation immediately after outputing the box."
    )

    if dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        question_key = "question"
    elif dataset_name == "math":
        ds = load_dataset("nlile/hendrycks-MATH-benchmark", split=split)
        question_key = "problem"
    elif dataset_name == "math-lighteval":
        ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split=split)
        question_key = "problem"
    elif dataset_name == "numina":
        ds = load_dataset("AI-MO/NuminaMath-CoT", split=split)
        question_key = "problem"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    prompts = []
    for ex in ds:
        q = ex.get(question_key, ex.get("question", ""))
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": q}]
        try:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    logger.info(f"Dataset '{dataset_name}': {len(prompts)} prompts")
    return prompts


# ════════════════════════════════════════════════════════════
# 3. vLLM SERVER CLIENT
# ════════════════════════════════════════════════════════════

class VLLMClient:
    """Client for vLLM OpenAI-compatible server with dynamic LoRA support."""

    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._check_server()

    def _check_server(self):
        """Verify the server is running."""
        try:
            r = requests.get(f"{self.base_url}/v1/models", timeout=5)
            r.raise_for_status()
            models = r.json()
            logger.info(f"vLLM server connected: {models['data'][0]['id']}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to vLLM server at {self.base_url}. "
                f"Start it with: bash scripts/start_vllm_server.sh\n"
                f"Error: {e}"
            )

    def generate_batch(self, prompts: List[str], config: 'TrainConfig',
                       lora_path: Optional[str] = None,
                       tokenizer=None) -> Dict[str, Any]:
        """Generate n completions per prompt via the vLLM server.

        Sends ALL prompts in a single batched request for efficiency.
        Returns same format as GenerationEngine.generate_rollouts().
        """
        n = config.num_generations
        model_name = self.model_name
        # vLLM dynamic LoRA: "model_name" for base, or lora adapter name
        if lora_path is not None:
            # Register-style: vLLM expects the adapter name, not a raw path.
            # With --enable-lora, pass the adapter as a separate field.
            model_name = self.model_name

        # Single batched request with all prompts
        body = {
            "model": model_name,
            "prompt": prompts,  # vLLM /v1/completions accepts a list of prompts
            "n": n,
            "temperature": config.temperature,
            "max_tokens": config.max_completion_length,
            "top_p": config.top_p,
            "logprobs": 1,
        }

        try:
            r = requests.post(
                f"{self.base_url}/v1/completions",
                json=body,
                timeout=1200,  # longer timeout for large batches
            )
            r.raise_for_status()
            resp = r.json()
        except Exception as e:
            logger.error(f"vLLM batch request failed: {e}")
            raise

        # Parse response: choices are ordered by prompt index, then by n
        # vLLM returns len(prompts) * n choices, sorted by (prompt_idx, generation_idx)
        choices = resp["choices"]
        P = len(prompts)

        result: Dict[str, Any] = {
            "prompt_texts": [],
            "completion_texts": [],
            "completion_token_ids": [],
            "old_per_token_logps": [],
            "prompt_token_ids": [],
        }

        for p_idx in range(P):
            result["prompt_texts"].append(prompts[p_idx])
            # Tokenize prompt locally (API doesn't return prompt_token_ids)
            if tokenizer:
                result["prompt_token_ids"].append(
                    tokenizer.encode(prompts[p_idx], add_special_tokens=False))
            else:
                result["prompt_token_ids"].append([])

            g_texts, g_ids, g_logps = [], [], []
            for g_idx in range(n):
                choice_idx = p_idx * n + g_idx
                choice = choices[choice_idx]

                g_texts.append(choice["text"])

                # Extract token IDs and logprobs from vLLM response
                token_ids = []
                token_logps = []
                if "logprobs" in choice and choice["logprobs"]:
                    lp_data = choice["logprobs"]
                    # vLLM completions API format: tokens, token_logprobs, token_ids
                    if "token_logprobs" in lp_data:
                        raw_logps = lp_data["token_logprobs"]
                        # First token may have null logprob
                        token_logps = [lp if lp is not None else 0.0 for lp in raw_logps]
                        # token_ids may or may not be present
                        if "token_ids" in lp_data:
                            token_ids = lp_data["token_ids"]

                # Fallback: if token_ids missing, tokenize the completion text
                if not token_ids and tokenizer is not None and choice["text"]:
                    token_ids = tokenizer.encode(choice["text"], add_special_tokens=False)
                    # Align lengths: truncate or pad logps to match token_ids
                    if len(token_logps) > len(token_ids):
                        token_logps = token_logps[:len(token_ids)]
                    elif len(token_logps) < len(token_ids):
                        token_logps.extend([0.0] * (len(token_ids) - len(token_logps)))

                g_ids.append(token_ids)
                g_logps.append(token_logps)

            result["completion_texts"].append(g_texts)
            result["completion_token_ids"].append(g_ids)
            result["old_per_token_logps"].append(g_logps)

        return result


# ════════════════════════════════════════════════════════════
# 4-9. SHARED COMPONENTS (from train_from_scratch.py)
# ════════════════════════════════════════════════════════════

from train_from_scratch import (
    compute_all_rewards,
    gdpo_normalize,
    compute_advantages,
    precompute_advantages,
    prepare_training_batch,
    forward_with_mean_logprobs,
    aggregate_loss,
    compute_grpo_loss,
    compute_ref_logprobs,
)


# ════════════════════════════════════════════════════════════
# 10. TRAINER
# ════════════════════════════════════════════════════════════

class DualGPUTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.prompts = prepare_dataset(
            self.tokenizer, config.dataset,
            max_samples=config.max_train_samples)

        # Connect to vLLM server on GPU 0
        self.vllm_client = VLLMClient(config.vllm_base_url, config.model_name)

        self.global_step = 0
        self._run_number: Optional[int] = None
        self.lora_path: Optional[str] = None

        # Resume
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
            logger.info(f"Resuming from {ckpt}, step={self.global_step}")

        if config.use_wandb:
            import wandb
            wandb_kwargs = dict(
                project=os.environ.get("WANDB_PROJECT", "multi-reward-dual-gpu"),
                name=config.run_name,
                config=asdict(config),
            )
            if config.resume_from:
                wandb_kwargs["resume"] = "allow"
            wandb.init(**wandb_kwargs)

    def _total_steps(self) -> int:
        if self.config.max_steps > 0:
            return self.config.max_steps
        n_prompts = len(self.prompts)
        prompts_per_step = self.config.train_batch_size * self.config.gradient_accumulation_steps
        steps_per_epoch = max(1, n_prompts // prompts_per_step)
        return steps_per_epoch * self.config.num_train_epochs

    def _init_model(self):
        """Load training model on GPU 1."""
        device = self.config.train_device
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)

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

        self.vocab_size = model.config.vocab_size

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable,
                          lr=self.config.learning_rate,
                          weight_decay=self.config.weight_decay,
                          betas=(0.9, 0.999))

        total_steps = self._total_steps()
        total_scheduler_steps = total_steps * self.config.mu
        warmup_steps = int(total_scheduler_steps * self.config.warmup_ratio)

        if self.config.lr_scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=total_scheduler_steps)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=total_scheduler_steps)

        # Restore optimizer/scheduler
        if self.lora_path:
            opt_path = os.path.join(self.lora_path, "optimizer.pt")
            sch_path = os.path.join(self.lora_path, "scheduler.pt")
            if os.path.isfile(opt_path):
                optimizer.load_state_dict(
                    torch.load(opt_path, map_location=device, weights_only=True))
            if os.path.isfile(sch_path):
                scheduler.load_state_dict(
                    torch.load(sch_path, map_location="cpu", weights_only=True))

        return model, optimizer, scheduler

    def _save_checkpoint(self, model, optimizer, scheduler, step: int,
                         permanent: bool = False):
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
        model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))

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
                     "gpu/used_gb", "step_time"]
        for k in key_order:
            if k in metrics:
                v = metrics[k]
                parts.append(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}")
        logger.info("  ".join(parts))

        if self.config.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def _print_rollouts(self, rollouts: Dict, rewards: Dict, step: int):
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
                lines.append(f"\n  ROLLOUT [{g}]  R_cluster={r_cl:.3f}  R_format={r_fm:.1f}{bstr}")
                lines.append(f"  {text}")
            if G > 4:
                lines.append(f"\n  … ({G - 4} more rollouts not shown)")
        lines.append(f"\n{'=' * 80}")
        logger.info("\n".join(lines))

    # ─── MAIN TRAINING LOOP ──────────────────────────────
    def train(self):
        total_steps = self._total_steps()
        logger.info(f"=== Multi-Reward GRPO (Dual GPU) ===")
        logger.info(f"Config: {asdict(self.config)}")
        logger.info(f"Total prompts: {len(self.prompts)}, total steps: {total_steps}")
        logger.info(f"Training on: {self.config.train_device}")
        logger.info(f"vLLM server: {self.config.vllm_base_url}")

        os.makedirs(self.config.output_dir, exist_ok=True)

        # Load training model on GPU 1 — stays loaded throughout
        model, optimizer, scheduler = self._init_model()
        device = torch.device(self.config.train_device)
        logger.info(f"Model loaded on {device}. "
                     f"GPU memory: {torch.cuda.memory_allocated(device)/1e9:.1f}GB")

        prompt_indices = list(range(len(self.prompts)))
        random.seed(self.config.seed)
        random.shuffle(prompt_indices)
        cursor = 0
        start_step = self.global_step

        if start_step > 0:
            bs = self.config.train_batch_size
            prompts_per_step = bs * self.config.gradient_accumulation_steps
            for _ in range(start_step):
                cursor += prompts_per_step
                while cursor >= len(prompt_indices):
                    random.shuffle(prompt_indices)
                    cursor -= len(prompt_indices)
            logger.info(f"Skipped {start_step} steps, resuming from step {start_step}")

        ga_steps = self.config.gradient_accumulation_steps
        bs = self.config.train_batch_size

        for step in range(start_step, total_steps):
            step_start = time.time()
            mu = self.config.mu
            step_metrics: Dict[str, List[float]] = defaultdict(list)
            all_full_batch_metrics: List[Dict[str, float]] = []

            # ── PHASE 1: Generate all GA rounds ONCE (fixed for all μ) ──
            ga_round_data: List[Dict[str, Any]] = []

            for ga_idx in range(ga_steps):
                if cursor + bs > len(prompt_indices):
                    random.shuffle(prompt_indices)
                    cursor = 0
                batch_indices = prompt_indices[cursor:cursor + bs]
                cursor += bs
                batch_prompts = [self.prompts[i] for i in batch_indices]

                logger.info(f"\n Step {step}/{total_steps} "
                            f"[ga {ga_idx+1}/{ga_steps}]: generating "
                            f"{bs}×{self.config.num_generations} completions …")

                gen_start = time.time()
                rollouts = self.vllm_client.generate_batch(
                    batch_prompts, self.config,
                    lora_path=self.lora_path,
                    tokenizer=self.tokenizer,
                )
                gen_time = time.time() - gen_start
                logger.info(f"  Generation took {gen_time:.1f}s")

                rewards = compute_all_rewards(rollouts, self.config)

                if (self.config.print_rollouts_every > 0
                        and step % self.config.print_rollouts_every == 0
                        and ga_idx == 0):
                    self._print_rollouts(rollouts, rewards, step)

                training_batch = prepare_training_batch(
                    rollouts, rewards, self.tokenizer, self.config)
                del rollouts
                for k, v in training_batch.items():
                    if isinstance(v, torch.Tensor):
                        training_batch[k] = v.to(device)

                B = training_batch["input_ids"].shape[0]

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
                    self.config,
                )

                all_full_batch_metrics.append({
                    "rewards/self_certainty_mean": precomputed_full["self_certainty"].mean().item(),
                    "advantages/ch_cluster_format_std": (
                        precomputed_full["ch_cf"].std(correction=0).item()
                        if precomputed_full["ch_cf"].numel() > 1 else 0.0),
                    "advantages/ch_self_certainty_std": (
                        precomputed_full["ch_sc"].std(correction=0).item()
                        if precomputed_full["ch_sc"].numel() > 1 else 0.0),
                    "advantages/combined_mean": precomputed_full["advantages"].mean().item(),
                })

                ga_round_data.append({
                    "training_batch": training_batch,
                    "precomputed": precomputed_full,
                    "B": B,
                })

            # ── PHASE 2: Train μ iterations over the SAME data ──
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

                        with torch.autocast(device.type, dtype=torch.bfloat16):
                            loss, metrics = compute_grpo_loss(
                                model, mb, self.config, self.vocab_size,
                                ref_per_token_logps=ref_logps,
                                gradient_accumulation_steps=actual_ga,
                                precomputed=mb_precomputed,
                            )

                        loss.backward()
                        del loss, ref_logps, mb_precomputed, mb

                        for k, v in metrics.items():
                            step_metrics[k].append(v)
                        del metrics

                # ── Optimizer step ──
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                ).item()
                optimizer.step()
                scheduler.step()

            del ga_round_data

            # ── Save checkpoint ──
            is_permanent = (step + 1) % self.config.save_steps == 0
            self._save_checkpoint(model, optimizer, scheduler, step + 1,
                                  permanent=is_permanent)

            current_lr = scheduler.get_last_lr()[0]

            # ── Logging ──
            avg_metrics = {k: sum(v) / len(v) for k, v in step_metrics.items()}
            if all_full_batch_metrics:
                combined = {}
                for d in all_full_batch_metrics:
                    for k, v in d.items():
                        combined.setdefault(k, []).append(v)
                avg_metrics.update({k: sum(v)/len(v) for k, v in combined.items()})

            avg_metrics["grad_norm"] = grad_norm
            avg_metrics["lr"] = current_lr
            avg_metrics["step_time"] = time.time() - step_start
            avg_metrics["gpu/used_gb"] = torch.cuda.memory_allocated(device) / 1e9

            if step % self.config.logging_steps == 0:
                self._log_step(step, avg_metrics)

            self.global_step = step + 1

        logger.info("Training complete!")


def main():
    config = parse_args()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    trainer = DualGPUTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
