"""
INTUITOR — Learning to Reason without External Rewards.

Exact reproduction of the Intuitor paper's method:
  - Self-certainty KL(U || π_θ) as the SOLE intrinsic reward
  - GRPO policy optimization with z-scored self-certainty advantages
  - KL penalty vs frozen reference model
  - NO entropy bonus, NO KL-Cov, NO cluster/format rewards

Architecture: same generate-then-train loop as train_from_scratch.py
  PHASE 1 — vLLM generation with optional LoRA
  PHASE 2 — PyTorch training with PEFT

Reference:
  Kang et al., "Learning to Reason without External Rewards" (2025)
  https://github.com/sunblaze-ucb/Intuitor
"""

import os, sys, gc, math, time, logging, argparse, json, random, shutil, glob
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
from torch.optim import AdamW

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset

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
class IntuitorConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B"
    lora_rank: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Dataset
    dataset: str = "math"
    max_train_samples: Optional[int] = None

    # Generation
    num_generations: int = 7
    max_prompt_length: int = 2048
    max_completion_length: int = 1024
    temperature: float = 1.0      # Paper default (not specified → TRL default 1.0)
    top_p: float = 1.0
    gpu_memory_utilization: float = 0.5

    # GRPO
    beta_kl: float = 0.005        # Paper: β = 0.005
    epsilon_clip: float = 0.2

    # Training
    train_batch_size: int = 16    # prompts per generation round
    gradient_accumulation_steps: int = 1  # rounds of generate→train per optimizer step
    # Paper: batch_size=4 × ga=32 = 128 effective prompts per update
    mu: int = 1                   # GRPO inner iterations (paper: num_iterations=1)
    learning_rate: float = 3e-6   # Paper: 3e-6
    lr_scheduler: str = "cosine"  # Paper: cosine
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    num_train_epochs: int = 1
    max_steps: int = -1

    # Infrastructure
    output_dir: str = "./outputs/intuitor"
    logging_steps: int = 1
    save_steps: int = 100
    print_rollouts_every: int = 10
    use_wandb: bool = False
    run_name: str = "intuitor"
    resume_from: Optional[str] = None
    seed: int = 7                 # Paper: seed 7


def parse_args() -> IntuitorConfig:
    p = argparse.ArgumentParser(description="INTUITOR: Self-Certainty GRPO")
    for f in IntuitorConfig.__dataclass_fields__.values():
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
    return IntuitorConfig(**vars(args))


# ════════════════════════════════════════════════════════════
# 2. DATASET
# ════════════════════════════════════════════════════════════

def prepare_dataset(tokenizer, dataset_name: str, split: str = "train",
                    max_samples: Optional[int] = None) -> List[str]:
    system_prompt = ("You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. "
                         "You FIRST think about the reasoning process step by step and then provide the user with the answer. "
                         "Please enclose your final answer in the box: \\boxed{Your Answer}. "
                         "Please stop generation immediately after outputing the box.")

    if dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        question_key = "question"
    elif dataset_name == "math":
        ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split=split)
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
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        prompts.append(prompt)

    logger.info(f"Dataset '{dataset_name}': {len(prompts)} prompts")
    return prompts


# ════════════════════════════════════════════════════════════
# 3. vLLM GENERATION
# ════════════════════════════════════════════════════════════

class GenerationEngine:
    def __init__(self, config: IntuitorConfig, lora_path: Optional[str] = None):
        from vllm import LLM

        free_mem, total_mem = torch.cuda.mem_get_info()
        max_util = (free_mem * 0.90) / total_mem
        gpu_util = min(config.gpu_memory_utilization, max_util)
        logger.info(f"  vLLM: free={free_mem/1e9:.1f}GB, util={gpu_util:.2f}")

        llm_kwargs = dict(
            model=config.model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=config.max_prompt_length + config.max_completion_length,
            gpu_memory_utilization=gpu_util,
            enforce_eager=True,
            seed=config.seed,
        )
        if lora_path is not None:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = config.lora_rank

        self.llm = LLM(**llm_kwargs)
        self.lora_path = lora_path
        self.config = config

    def generate_rollouts(self, prompts: List[str]) -> Dict[str, Any]:
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

        outputs = self.llm.generate(prompts, sampling_params,
                                    lora_request=lora_request)

        result: Dict[str, Any] = {
            "prompt_texts": [], "completion_texts": [],
            "completion_token_ids": [], "old_per_token_logps": [],
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

    def shutdown(self):
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════
# 4. SELF-CERTAINTY (Eq. 2-4)
# ════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_self_certainty(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    completion_mask: torch.Tensor, logits_to_keep: int,
    chunk_size: int = 2,
) -> torch.Tensor:
    """Compute per-sequence self-certainty = avg KL(U || π_θ).

    Uses RAW logits (no temperature) matching the paper's Eq. 2.
    The constant log|V| is included but cancels in z-scoring.
    """
    B = input_ids.shape[0]
    all_sc = []

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        outputs = model(
            input_ids=input_ids[start:end],
            attention_mask=attention_mask[start:end],
        )
        # RAW logits — paper computes self-certainty on the model's
        # true distribution, NOT the temperature-scaled sampling distribution.
        logits = outputs.logits[:, -(logits_to_keep + 1):-1, :]

        # KL(U||π) = logsumexp(logits) - mean(logits) - log|V|
        # Paper uses logsumexp - mean (drops -log|V| since it cancels in z-scoring)
        sce = logits.logsumexp(dim=-1) - logits.mean(dim=-1)

        mask_c = completion_mask[start:end]
        sc = (sce * mask_c).sum(dim=-1) / mask_c.sum(dim=-1).clamp(min=1)
        all_sc.append(sc)
        del outputs, logits, sce

    return torch.cat(all_sc)


def compute_advantages(
    self_certainty: torch.Tensor, group_idx: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Z-score self-certainty per prompt group (Eq. 6).

    Aˆ_i = (u_i - mean(u)) / std(u)  within each prompt group.
    """
    advantages = torch.zeros_like(self_certainty)
    for gid in group_idx.unique():
        mask = group_idx == gid
        group = self_certainty[mask]
        if group.numel() <= 1:
            continue
        mu = group.mean()
        sigma = group.std(correction=0)
        if sigma > eps:
            advantages[mask] = (group - mu) / (sigma + eps)
    return advantages


# ════════════════════════════════════════════════════════════
# 5. BATCH PREPARATION
# ════════════════════════════════════════════════════════════

def prepare_training_batch(
    rollouts: Dict, tokenizer, config: IntuitorConfig,
) -> Dict[str, Any]:
    P = len(rollouts["completion_texts"])
    G = config.num_generations
    B = P * G

    all_prompt_ids, all_comp_ids, all_old_logps = [], [], []
    for p_idx in range(P):
        prompt_ids = rollouts["prompt_token_ids"][p_idx]
        for g_idx in range(G):
            all_prompt_ids.append(prompt_ids)
            all_comp_ids.append(rollouts["completion_token_ids"][p_idx][g_idx])
            all_old_logps.append(rollouts["old_per_token_logps"][p_idx][g_idx])

    max_prompt_len = max(len(p) for p in all_prompt_ids)
    max_comp_len = max(len(c) for c in all_comp_ids) if all_comp_ids else 1

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    max_total = max_prompt_len + max_comp_len

    input_ids = torch.full((B, max_total), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_total), dtype=torch.long)
    completion_mask = torch.zeros((B, max_comp_len), dtype=torch.float32)
    old_per_token_logps = torch.zeros((B, max_comp_len), dtype=torch.float32)

    for i in range(B):
        p_ids, c_ids = all_prompt_ids[i], all_comp_ids[i]
        prompt_pad = max_prompt_len - len(p_ids)

        input_ids[i, prompt_pad:max_prompt_len] = torch.tensor(p_ids, dtype=torch.long)
        comp_end = max_prompt_len + len(c_ids)
        input_ids[i, max_prompt_len:comp_end] = torch.tensor(c_ids, dtype=torch.long)
        attention_mask[i, prompt_pad:comp_end] = 1
        completion_mask[i, :len(c_ids)] = 1.0
        old_lp = all_old_logps[i]
        old_per_token_logps[i, :len(old_lp)] = torch.tensor(old_lp, dtype=torch.float32)

    group_idx = torch.arange(B, dtype=torch.long) // G

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "old_per_token_logps": old_per_token_logps,
        "prompt_group_idx": group_idx,
        "max_comp_len": max_comp_len,
    }


# ════════════════════════════════════════════════════════════
# 6. FORWARD PASS
# ════════════════════════════════════════════════════════════

def forward_logprobs(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    logits_to_keep: int,
) -> torch.Tensor:
    """Per-token log-probs using RAW logits.

    vLLM V0 returns raw logprobs by default (logprobs_mode='raw_logprobs'),
    so training logprobs must also use raw logits for a consistent ratio.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, -(logits_to_keep + 1):-1, :]
    lse = logits.logsumexp(dim=-1)
    completion_ids = input_ids[:, -logits_to_keep:]
    per_token_logps = logits.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1) - lse
    return per_token_logps


# ════════════════════════════════════════════════════════════
# 7. REFERENCE LOG-PROBS
# ════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_ref_logprobs(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    logits_to_keep: int,
) -> torch.Tensor:
    had_adapters = False
    if hasattr(model, "disable_adapter_layers"):
        model.disable_adapter_layers()
        had_adapters = True

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, -(logits_to_keep + 1):-1, :]
    lse = logits.logsumexp(dim=-1)
    completion_ids = input_ids[:, -logits_to_keep:]
    ref_logps = logits.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1) - lse

    if had_adapters:
        model.enable_adapter_layers()
    return ref_logps


# ════════════════════════════════════════════════════════════
# 8. GRPO LOSS (Paper Eq. 5)
# ════════════════════════════════════════════════════════════

def compute_intuitor_loss(
    model,
    batch: Dict[str, torch.Tensor],
    config: IntuitorConfig,
    ref_per_token_logps: Optional[torch.Tensor],
    advantages: torch.Tensor,
    gradient_accumulation_steps: int = 1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Standard GRPO clipped surrogate + KL penalty. No extras."""

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    mask = batch["completion_mask"]
    old_per_token_logps = batch["old_per_token_logps"]
    max_comp_len = batch["max_comp_len"]

    # Forward pass — raw logits (matches vLLM old_per_token_logps)
    per_token_logps = forward_logprobs(model, input_ids, attention_mask, max_comp_len)

    # Importance sampling ratio (Eq. 5)
    log_ratio = per_token_logps - old_per_token_logps
    ratio = torch.exp(log_ratio)

    adv = advantages.unsqueeze(1)  # (B, 1) broadcast to (B, T)

    # Clipped surrogate (Eq. 5)
    clipped_ratio = torch.clamp(ratio, 1.0 - config.epsilon_clip,
                                       1.0 + config.epsilon_clip)
    per_token_loss = -torch.min(ratio * adv, clipped_ratio * adv)

    # KL penalty vs reference (Eq. 5)
    if config.beta_kl != 0.0 and ref_per_token_logps is not None:
        per_token_kl = (torch.exp(ref_per_token_logps - per_token_logps)
                        - (ref_per_token_logps - per_token_logps) - 1)
        per_token_loss = per_token_loss + config.beta_kl * per_token_kl
    else:
        per_token_kl = torch.zeros_like(per_token_logps)

    # GRPO aggregation: mean over tokens per sequence, then mean over batch
    loss = ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()
    loss_val = loss.detach().item()
    loss = loss / gradient_accumulation_steps

    # Metrics
    masked_kl = (per_token_kl * mask).sum() / mask.sum().clamp(min=1)
    metrics = {
        "loss": loss_val,
        "kl": masked_kl.item(),
        "ratio_mean": ratio.mean().item(),
    }
    return loss, metrics


# ════════════════════════════════════════════════════════════
# 9. TRAINER
# ════════════════════════════════════════════════════════════

class IntuitorTrainer:
    def __init__(self, config: IntuitorConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.prompts = prepare_dataset(
            self.tokenizer, config.dataset,
            max_samples=config.max_train_samples)

        self.vocab_size: Optional[int] = None
        self.global_step = 0
        self.lora_path: Optional[str] = None
        self._optimizer_path: Optional[str] = None
        self._scheduler_path: Optional[str] = None
        self._run_number: Optional[int] = None

        if config.resume_from:
            ckpt = config.resume_from.strip()
            if not os.path.isdir(ckpt):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
            self.lora_path = ckpt
            opt_path = os.path.join(ckpt, "optimizer.pt")
            sch_path = os.path.join(ckpt, "scheduler.pt")
            self._optimizer_path = opt_path if os.path.isfile(opt_path) else None
            self._scheduler_path = sch_path if os.path.isfile(sch_path) else None
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
            wkw = dict(
                project=os.environ.get("WANDB_PROJECT", "intuitor"),
                name=config.run_name, config=asdict(config),
            )
            if config.resume_from:
                wkw["resume"] = "allow"
            wandb.init(**wkw)

    def _total_steps(self) -> int:
        if self.config.max_steps > 0:
            return self.config.max_steps
        n = len(self.prompts)
        return max(1, n // self.config.train_batch_size) * self.config.num_train_epochs

    def _init_training_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda")

        if self.lora_path is not None:
            model = PeftModel.from_pretrained(model, self.lora_path, is_trainable=True)
        else:
            lora_cfg = LoraConfig(
                r=self.config.lora_rank, lora_alpha=self.config.lora_alpha,
                target_modules="all-linear", task_type="CAUSAL_LM",
                lora_dropout=self.config.lora_dropout,
            )
            model = get_peft_model(model, lora_cfg)

        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.train()
        self.vocab_size = model.config.vocab_size

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=self.config.learning_rate,
                          weight_decay=self.config.weight_decay, betas=(0.9, 0.999))

        total_steps = self._total_steps() * self.config.mu
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        if self.config.lr_scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=total_steps)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=total_steps)

        if self._optimizer_path and os.path.isfile(self._optimizer_path):
            optimizer.load_state_dict(
                torch.load(self._optimizer_path, map_location="cuda", weights_only=True))
        if self._scheduler_path and os.path.isfile(self._scheduler_path):
            scheduler.load_state_dict(
                torch.load(self._scheduler_path, map_location="cpu", weights_only=True))

        return model, optimizer, scheduler

    def _save_checkpoint(self, model, optimizer, scheduler, step: int,
                         permanent: bool = False):
        if permanent:
            if self._run_number is None:
                base = self.config.output_dir
                existing = glob.glob(os.path.join(base, "run_*"))
                nums = [int(r.split("_")[-1]) for r in existing if r.split("_")[-1].isdigit()]
                self._run_number = max(nums) + 1 if nums else 0
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
        self._optimizer_path = os.path.join(save_dir, "optimizer.pt")
        self._scheduler_path = os.path.join(save_dir, "scheduler.pt")

        if permanent:
            logger.info(f"  Saved checkpoint → {save_dir}")

    @staticmethod
    def _free_model(*objs):
        for o in objs:
            try:
                if hasattr(o, "cpu"):
                    o.cpu()
            except Exception:
                pass
            del o
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

    def _log_step(self, step: int, metrics: Dict[str, float]):
        parts = [f"step={step}"]
        for k in ["loss", "grad_norm", "lr", "kl", "self_certainty_mean",
                   "self_certainty_std", "advantage_std", "ratio_mean", "step_time"]:
            if k in metrics:
                v = metrics[k]
                parts.append(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}")
        logger.info("  ".join(parts))

        if self.config.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    # ─── MAIN TRAINING LOOP ──────────────────────────────
    def train(self):
        total_steps = self._total_steps()
        logger.info(f"=== INTUITOR Training ===")
        logger.info(f"Config: {asdict(self.config)}")
        logger.info(f"Prompts: {len(self.prompts)}, steps: {total_steps}")

        os.makedirs(self.config.output_dir, exist_ok=True)

        prompt_indices = list(range(len(self.prompts)))
        random.seed(self.config.seed)
        random.shuffle(prompt_indices)
        cursor = 0
        start_step = self.global_step

        if start_step > 0:
            bs = self.config.train_batch_size
            for _ in range(start_step):
                cursor += bs
                if cursor >= len(prompt_indices):
                    random.shuffle(prompt_indices)
                    cursor = cursor % len(prompt_indices)
            logger.info(f"Skipped {start_step} steps, resuming from step {start_step}")

        ga_steps = self.config.gradient_accumulation_steps
        effective_bs = self.config.train_batch_size * ga_steps
        logger.info(f"Effective batch: {self.config.train_batch_size} × {ga_steps} "
                     f"= {effective_bs} prompts per optimizer step")

        for step in range(start_step, total_steps):
            step_start = time.time()
            bs = self.config.train_batch_size

            # Load model ONCE per optimizer step
            model, optimizer, scheduler = self._init_training_model()
            step_metrics: Dict[str, List[float]] = defaultdict(list)
            all_full_metrics: List[Dict[str, float]] = []

            for mu_iter in range(self.config.mu):
                if self.config.mu > 1:
                    logger.info(f"    GRPO iteration {mu_iter + 1}/{self.config.mu}")

                optimizer.zero_grad()

                for ga_idx in range(ga_steps):
                    # ── Sample prompts ────────────────────
                    if cursor + bs > len(prompt_indices):
                        random.shuffle(prompt_indices)
                        cursor = 0
                    batch_indices = prompt_indices[cursor:cursor + bs]
                    cursor += bs
                    batch_prompts = [self.prompts[i] for i in batch_indices]

                    # ── PHASE 1: GENERATE (vLLM) ─────────
                    logger.info(f"\n Step {step}/{total_steps} "
                                f"[ga {ga_idx+1}/{ga_steps}]: generating "
                                f"{bs}×{self.config.num_generations} completions …")

                    # Free model so vLLM can use GPU memory
                    model.cpu()
                    torch.cuda.empty_cache()

                    gen = GenerationEngine(self.config, lora_path=self.lora_path)
                    rollouts = gen.generate_rollouts(batch_prompts)
                    gen.shutdown()
                    del gen

                    # Move model back to GPU
                    model.cuda()

                    # ── Prepare batch ─────────────────────
                    batch = prepare_training_batch(rollouts, self.tokenizer, self.config)
                    device = torch.device("cuda")
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(device)

                    B = batch["input_ids"].shape[0]
                    mini_bs = 1
                    # Loss divisor: B completions × ga_steps rounds
                    actual_ga = B * ga_steps

                    # Pre-compute self-certainty + advantages
                    sc = compute_self_certainty(
                        model, batch["input_ids"], batch["attention_mask"],
                        batch["completion_mask"], batch["max_comp_len"],
                    )
                    advantages = compute_advantages(sc, batch["prompt_group_idx"])

                    all_full_metrics.append({
                        "self_certainty_mean": sc.mean().item(),
                        "self_certainty_std": sc.std().item(),
                        "advantage_std": advantages.std().item() if advantages.numel() > 1 else 0.0,
                    })

                    logger.info(f"  SC: mean={sc.mean():.4f}, std={sc.std():.4f}, "
                                 f"adv_std={advantages.std():.4f}")

                    # ── Forward + backward ────────────────
                    for mb_start in range(0, B, mini_bs):
                        mb_end = min(mb_start + mini_bs, B)
                        mb = {}
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor) and v.shape[0] == B:
                                mb[k] = v[mb_start:mb_end]
                            else:
                                mb[k] = v
                        mb["max_comp_len"] = batch["max_comp_len"]
                        mb_adv = advantages[mb_start:mb_end]

                        ref_logps = compute_ref_logprobs(
                            model, mb["input_ids"], mb["attention_mask"],
                            mb["max_comp_len"])

                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            loss, metrics = compute_intuitor_loss(
                                model, mb, self.config,
                                ref_per_token_logps=ref_logps,
                                advantages=mb_adv,
                                gradient_accumulation_steps=actual_ga,
                            )

                        loss.backward()
                        for k, v in metrics.items():
                            step_metrics[k].append(v)

                    # Free this generation batch
                    del batch, rollouts, sc, advantages
                    gc.collect()
                    torch.cuda.empty_cache()

                # ── Optimizer step (after all ga rounds) ──
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                ).item()
                optimizer.step()
                scheduler.step()

            # ── Save ──────────────────────────────────────
            self._save_checkpoint(model, optimizer, scheduler, step + 1,
                                  permanent=((step + 1) % self.config.save_steps == 0))
            current_lr = scheduler.get_last_lr()[0]
            self._free_model(model, optimizer, scheduler)
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # ── Log ───────────────────────────────────────
            avg = {k: sum(v) / len(v) for k, v in step_metrics.items()}
            # Average full-batch metrics across ga rounds
            if all_full_metrics:
                full_avg = {
                    k: sum(d[k] for d in all_full_metrics) / len(all_full_metrics)
                    for k in all_full_metrics[0]
                }
                avg.update(full_avg)
            avg["grad_norm"] = grad_norm
            avg["lr"] = current_lr
            avg["step_time"] = time.time() - step_start

            if step % self.config.logging_steps == 0:
                self._log_step(step, avg)

            self.global_step = step + 1

        # ── Final save ────────────────────────────────────
        final_dir = os.path.join(self.config.output_dir, "final")
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
    trainer = IntuitorTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
