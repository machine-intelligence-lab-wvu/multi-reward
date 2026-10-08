"""
Multi-Reward Unsupervised GRPO — From Scratch (vLLM + PyTorch).

No TRL dependency.  Uses vLLM for fast batch generation and raw PyTorch +
PEFT for training on a single RTX 4090 (24 GB).

Components implemented:
  1. CLUSTER VOTING  (answer-level, Eq. 1)
  2. SELF-CERTAINTY  (reasoning-level, Eq. 2)  KL(U || π)
  3. GDPO NORMALIZATION (Eq. 3-4)
  4. KL-COV  (token-level covariance-aware regularization, Eq. 6-9)
  5. KL penalty vs frozen reference model
  6. Entropy bonus for exploration

Architecture
────────────
For each training step the script alternates between two GPU phases:

  PHASE 1 — GENERATE (vLLM)
    Load base model + LoRA → generate G completions per prompt
    Extract per-token log-probs (π_old)
    Shut down vLLM, free GPU

  PHASE 2 — TRAIN (PyTorch / PEFT)
    Load model + LoRA + optimizer + scheduler
    For each mini-batch (gradient accumulation):
      Forward pass → per_token_logps, entropies, mean_log_probs
      ref_logps via disable_adapter_layers()
      Self-certainty (Eq. 2), GDPO (Eq. 3-4), KL-Cov (Eq. 6-9)
      Clipped surrogate loss + KL penalty + entropy bonus
      loss.backward()
    optimizer.step(), scheduler.step()
    Save LoRA + optimizer + scheduler state
    Free GPU
"""

import os, sys, gc, math, time, logging, argparse, json, random, shutil, glob
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
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
    model_name: str = "Qwen/Qwen3-1.7B"
    lora_rank: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Dataset
    dataset: str = "gsm8k"
    max_train_samples: Optional[int] = None

    # Generation
    num_generations: int = 8
    max_prompt_length: int = 512
    max_completion_length: int = 3072
    temperature: float = 1.5
    top_p: float = 1.0
    gpu_memory_utilization: float = 0.6

    # Reward weights (TRL-compatible; used when combining cluster + format
    # for the cluster_format channel before GDPO)
    w_cluster: float = 0.5
    w_format: float = 0.1

    # GDPO channel weights (Eq. 4)
    w_cluster_format: float = 0.5
    w_selfcert: float = 0.5

    # KL-Cov
    beta_cov: float = 0.1
    top_k_fraction: float = 0.02

    # KL & Entropy
    beta_kl: float = 0.005
    entropy_bonus: float = 0.01
    epsilon_clip: float = 0.2
    min_completion_tokens: int = 50  # completions shorter than this get reward=0

    # Training
    # Effective batch = train_batch_size * num_generations completions per
    # optimizer.step().  To match TRL's per_device_train_batch_size=2 with
    # gradient_accumulation_steps=8, use train_batch_size=16 (= 2 * 8).
    # The within-step mini-batch loop processes 1 sample at a time for
    # memory efficiency but accumulates gradients over all completions.
    train_batch_size: int = 16          # prompts per outer step
    gradient_accumulation_steps: int = 1  # reserved for future cross-step accumulation
    mu: int = 1                         # GRPO iterations per step (Algorithm 1, line 10)
    learning_rate: float = 5e-6
    lr_scheduler: str = "linear"      # "linear" or "cosine"
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    num_train_epochs: int = 1
    max_steps: int = -1
    loss_type: str = "grpo"

    # Infrastructure
    output_dir: str = "./outputs/multi_reward_scratch"
    logging_steps: int = 1
    save_steps: int = 100
    print_rollouts_every: int = 1
    use_wandb: bool = False
    run_name: str = "multi_reward_scratch"
    resume_from: Optional[str] = None  # path to checkpoint dir (e.g. ./outputs/.../run_0/checkpoint-200)
    thinking: bool = False             # enable Qwen3 thinking mode (SC on reasoning only)
    seed: int = 42


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Multi-Reward GRPO from Scratch")
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
# 2. DATASET PREPARATION
# ════════════════════════════════════════════════════════════

def prepare_dataset(tokenizer, dataset_name: str, split: str = "train",
                    max_samples: Optional[int] = None) -> List[str]:
    """Return a list of chat-templated prompt strings."""
    system_prompt = ("You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. "
                         "You FIRST think about the reasoning process step by step and then provide the user with the answer. "
                         "Please enclose your final answer in the box: \\boxed{Your Answer}. "
                         "Please stop generation immediately after outputing the box.")

    if dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        question_key = "question"
        
    elif dataset_name == "math":
        try: ds = load_dataset("nlile/hendrycks-MATH-benchmark", split=split)
        except: print("Hendrycks-MATH-benchmark not FOUND")
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
# 3. vLLM GENERATION ENGINE
# ════════════════════════════════════════════════════════════

class GenerationEngine:
    """Wraps vLLM for fast batch generation with optional LoRA."""

    def __init__(self, config: TrainConfig, lora_path: Optional[str] = None):
        from vllm import LLM

        # Dynamically compute gpu_memory_utilization based on free memory.
        # Use 0.85 safety factor to leave headroom for PyTorch CUDA allocator
        # fragmentation (a known leak in generate-then-train architectures).
        free_mem, total_mem = torch.cuda.mem_get_info()
        max_util = (free_mem * 0.85) / total_mem
        gpu_util = min(config.gpu_memory_utilization, max_util)
        logger.info(f"  vLLM: free={free_mem/1e9:.1f}GB, "
                     f"gpu_memory_utilization={gpu_util:.2f}")

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
        """Generate G completions per prompt.

        Returns
        -------
        dict with keys:
            prompt_texts        : List[str]                P
            completion_texts    : List[List[str]]          P × G
            completion_token_ids: List[List[List[int]]]    P × G × T_i
            old_per_token_logps : List[List[List[float]]]  P × G × T_i
            prompt_token_ids    : List[List[int]]          P × T_prompt_i
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

        outputs = self.llm.generate(prompts, sampling_params,
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

    def shutdown(self):
        """Free all GPU memory held by vLLM."""
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════
# 4. REWARD COMPUTATION
# ════════════════════════════════════════════════════════════

def compute_all_rewards(rollouts: Dict, config: TrainConfig) -> Dict[str, torch.Tensor]:
    """Compute cluster-voting and format rewards.

    Returns dict with (P*G,) tensors:
        cluster_rewards, format_rewards
    """
    P = len(rollouts["completion_texts"])
    G = config.num_generations
    min_tokens = config.min_completion_tokens

    cluster_rewards: List[float] = []
    format_rewards: List[float] = []

    for p_idx in range(P):
        group_texts = rollouts["completion_texts"][p_idx]

        # Cluster voting (Eq. 1)
        group_cluster, _ = compute_cluster_rewards(group_texts)

        for g_idx, text in enumerate(group_texts):
            # Zero out rewards for short completions (prevents template collapse)
            comp_len = len(rollouts["completion_token_ids"][p_idx][g_idx])
            if min_tokens > 0 and comp_len < min_tokens:
                cluster_rewards.append(0.0)
                format_rewards.append(0.0)
            else:
                cluster_rewards.append(group_cluster[g_idx])
                format_rewards.append(compute_format_reward(text))

    return {
        "cluster_rewards": torch.tensor(cluster_rewards, dtype=torch.float32),
        "format_rewards": torch.tensor(format_rewards, dtype=torch.float32),
    }


# ════════════════════════════════════════════════════════════
# 5. GDPO NORMALIZATION  (Eq. 3-4)
# ════════════════════════════════════════════════════════════

def gdpo_normalize(rewards: torch.Tensor, group_idx: torch.Tensor,
                   eps: float = 1e-8) -> torch.Tensor:
    """Z-score normalise rewards within each prompt group."""
    normalised = torch.zeros_like(rewards)
    for gid in group_idx.unique():
        mask = group_idx == gid
        group = rewards[mask]
        if group.numel() <= 1:
            continue
        mu = group.mean()
        sigma = group.std(correction=0)
        if sigma > eps:
            normalised[mask] = (group - mu) / (sigma + eps)
    return normalised


def compute_advantages(cluster_rewards: torch.Tensor,
                       format_rewards: torch.Tensor,
                       self_certainty: torch.Tensor,
                       group_idx: torch.Tensor,
                       config: TrainConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GDPO per-channel z-score then weighted combine (Eq. 3-4).

    Returns (advantages, ch_cf_normed, ch_sc_normed).
    """
    # Multi-reward proposal: build the answer-level channel from
    # weighted cluster + format rewards before per-channel normalization.
    ch_cf_raw = (
        config.w_cluster * cluster_rewards.to(self_certainty.device)
        + config.w_format * format_rewards.to(self_certainty.device)
    )
    ch_cf = gdpo_normalize(ch_cf_raw, group_idx)
    ch_sc = gdpo_normalize(self_certainty.detach(), group_idx)
    advantages = config.w_cluster_format * ch_cf + config.w_selfcert * ch_sc
    return advantages, ch_cf, ch_sc


@torch.no_grad()
def precompute_advantages(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    cluster_rewards: torch.Tensor,
    format_rewards: torch.Tensor,
    group_idx: torch.Tensor,
    logits_to_keep: int,
    vocab_size: int,
    config: TrainConfig,
    chunk_size: int = 2,
) -> Dict[str, torch.Tensor]:
    """Pre-compute self-certainty and GDPO advantages over the full batch.

    Uses chunked no-grad forward passes so the full prompt group is visible
    for GDPO z-scoring, without the backward-pass memory cost.
    """
    B = input_ids.shape[0]
    log_V = math.log(vocab_size)
    all_sc = []

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        outputs = model(
            input_ids=input_ids[start:end],
            attention_mask=attention_mask[start:end],
        )
        logits = outputs.logits[:, -(logits_to_keep + 1):-1, :]
        # Self-certainty uses RAW logits (paper Eq. 2): KL(U || π_θ)
        # where π_θ is the model's true distribution, NOT temperature-scaled.
        lse = logits.logsumexp(dim=-1)
        mean_lp = logits.mean(dim=-1) - lse

        mask_c = completion_mask[start:end]
        kl_u_pi = (-log_V - mean_lp) * mask_c
        sc = kl_u_pi.sum(dim=-1) / mask_c.sum(dim=-1).clamp(min=1)
        all_sc.append(sc)
        del outputs, logits, lse, mean_lp

    self_certainty = torch.cat(all_sc)

    # Penalize short completions: set self-certainty to group minimum
    # so they never get positive advantage from the self-certainty channel.
    # This prevents the model from collapsing to short template outputs.
    if config.min_completion_tokens > 0:
        lengths = completion_mask.sum(dim=-1)  # (B,)
        short_mask = lengths < config.min_completion_tokens
        if short_mask.any():
            for gid in group_idx.unique():
                g_mask = group_idx == gid
                group_sc = self_certainty[g_mask]
                short_in_group = short_mask[g_mask]
                if short_in_group.any() and (~short_in_group).any():
                    # Set short completions to group min so z-score is negative
                    group_min = group_sc[~short_in_group].min()
                    self_certainty[g_mask & short_mask] = group_min
                elif short_in_group.all():
                    # All short → zero out (no signal from this group)
                    self_certainty[g_mask] = 0.0

    advantages, ch_cf, ch_sc = compute_advantages(
        cluster_rewards, format_rewards, self_certainty, group_idx, config,
    )
    return {
        "advantages": advantages,
        "self_certainty": self_certainty,
        "ch_cf": ch_cf,
        "ch_sc": ch_sc,
    }


# ════════════════════════════════════════════════════════════
# 6. TRAINING BATCH PREPARATION
# ════════════════════════════════════════════════════════════

def prepare_training_batch(
    rollouts: Dict,
    rewards: Dict[str, torch.Tensor],
    tokenizer,
    config: TrainConfig,
) -> Dict[str, Any]:
    """Convert ragged generation outputs into padded tensors.

    Left-pads prompts, right-pads completions so that completion tokens
    always occupy positions [max_prompt_len : max_prompt_len + max_comp_len].
    This lets us use the same ``[:, -(C+1):-1, :]`` logit indexing as the
    current train.py implementation.
    """
    P = len(rollouts["completion_texts"])
    G = config.num_generations
    B = P * G

    # Gather per-sample data
    all_prompt_ids: List[List[int]] = []
    all_comp_ids: List[List[int]] = []
    all_old_logps: List[List[float]] = []

    for p_idx in range(P):
        prompt_ids = rollouts["prompt_token_ids"][p_idx]
        for g_idx in range(G):
            comp_ids = rollouts["completion_token_ids"][p_idx][g_idx]
            old_logps = rollouts["old_per_token_logps"][p_idx][g_idx]

            all_prompt_ids.append(prompt_ids)
            all_comp_ids.append(comp_ids)
            all_old_logps.append(old_logps)

    max_prompt_len = max(len(p) for p in all_prompt_ids)
    max_comp_len = max(len(c) for c in all_comp_ids) if all_comp_ids else 1
    max_total_len = max_prompt_len + max_comp_len

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    input_ids = torch.full((B, max_total_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_total_len), dtype=torch.long)
    completion_mask = torch.zeros((B, max_comp_len), dtype=torch.float32)
    old_per_token_logps = torch.zeros((B, max_comp_len), dtype=torch.float32)

    for i in range(B):
        p_ids = all_prompt_ids[i]
        c_ids = all_comp_ids[i]
        prompt_pad = max_prompt_len - len(p_ids)

        # Left-pad prompt
        input_ids[i, prompt_pad:max_prompt_len] = torch.tensor(p_ids, dtype=torch.long)
        # Right-place completion
        comp_end = max_prompt_len + len(c_ids)
        input_ids[i, max_prompt_len:comp_end] = torch.tensor(c_ids, dtype=torch.long)
        # Attention mask
        attention_mask[i, prompt_pad:comp_end] = 1
        # Completion mask & old logps
        completion_mask[i, :len(c_ids)] = 1.0
        old_lp = all_old_logps[i]
        old_per_token_logps[i, :len(old_lp)] = torch.tensor(old_lp, dtype=torch.float32)

    # Group indices — consecutive groups of G (no TRL shuffle to worry about)
    prompt_group_idx = torch.arange(B, dtype=torch.long) // G

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "old_per_token_logps": old_per_token_logps,
        "cluster_rewards": rewards["cluster_rewards"],
        "format_rewards": rewards["format_rewards"],
        "prompt_group_idx": prompt_group_idx,
        "max_comp_len": max_comp_len,
    }


# ════════════════════════════════════════════════════════════
# 7. FORWARD PASS WITH MEAN LOG-PROBS
# ════════════════════════════════════════════════════════════

def forward_with_mean_logprobs(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    logits_to_keep: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass returning per-token log-probs, entropy, and mean log-probs.

    CRITICAL: In a causal LM, logits[:, t, :] predicts token at position t+1.
    To get P(completion_token_t | context), we need logits at position t-1.
    Correct indexing: ``logits[:, -(C+1):-1, :]``.
    Using ``[:, -C:, :]`` is OFF-BY-ONE and causes KL explosion.

    Memory optimisation: chunk-wise entropy avoids materialising full
    (B, S, V) probs tensor.  Saves ~2.4 GB vs naive.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[:, -(logits_to_keep + 1):-1, :]
    # NO temperature scaling here. vLLM V0 returns RAW logprobs
    # (logprobs_mode="raw_logprobs" is default), so per_token_logps
    # must also use raw logits for the ratio to be correct.
    # (This differs from TRL where both old and new logps use temperature.)
    B, S, V = logits.shape

    lse = logits.logsumexp(dim=-1)                          # (B, S)

    completion_ids = input_ids[:, -logits_to_keep:]
    per_token_logps = (logits.gather(-1, completion_ids.unsqueeze(-1))
                       .squeeze(-1) - lse)                  # (B, S)

    mean_log_probs = logits.mean(dim=-1) - lse              # (B, S)

    # Entropy H(π) = logsumexp - E_π[logits], chunked over vocab
    CHUNK = 32768
    weighted_sum = torch.zeros(B, S, device=logits.device, dtype=logits.dtype)
    for v_start in range(0, V, CHUNK):
        v_end = min(v_start + CHUNK, V)
        chunk_logits = logits[:, :, v_start:v_end]
        chunk_probs = torch.exp(chunk_logits - lse.unsqueeze(-1))
        weighted_sum += (chunk_probs * chunk_logits).sum(dim=-1)
        del chunk_probs, chunk_logits
    entropies = lse - weighted_sum
    del lse, weighted_sum

    return per_token_logps, entropies, mean_log_probs


# ════════════════════════════════════════════════════════════
# 8. GRPO LOSS  (all components)
# ════════════════════════════════════════════════════════════

def aggregate_loss(per_token_loss: torch.Tensor, mask: torch.Tensor,
                   loss_type: str, max_completion_length: int) -> torch.Tensor:
    if loss_type == "grpo":
        return ((per_token_loss * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1)).mean()
    elif loss_type in ("dapo", "cispo", "bnpo"):
        return (per_token_loss * mask).sum() / mask.sum().clamp(min=1)
    elif loss_type == "dr_grpo":
        return ((per_token_loss * mask).sum()
                / (per_token_loss.size(0) * max_completion_length))
    else:
        return ((per_token_loss * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1)).mean()


def compute_grpo_loss(
    model,
    batch: Dict[str, torch.Tensor],
    config: TrainConfig,
    vocab_size: int,
    ref_per_token_logps: Optional[torch.Tensor] = None,
    gradient_accumulation_steps: int = 1,
    precomputed: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Full GRPO loss with all multi-reward components.

    If ``precomputed`` is provided (from ``precompute_advantages``),
    steps 2-3 are skipped and the pre-computed advantages / metrics
    are used instead.  This allows mini_bs < G for gradient
    accumulation without breaking GDPO normalization.
    """

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    completion_mask = batch["completion_mask"]
    old_per_token_logps = batch["old_per_token_logps"]
    cluster_rewards = batch["cluster_rewards"]
    format_rewards = batch["format_rewards"]
    group_idx = batch["prompt_group_idx"]
    max_comp_len = batch["max_comp_len"]

    mask = completion_mask

    # ── STEP 1: Forward pass ──────────────────────────────
    per_token_logps, entropies, mean_log_probs = forward_with_mean_logprobs(
        model, input_ids, attention_mask, max_comp_len,
    )

    # ── STEP 2-3: Advantages (pre-computed or inline) ────
    if precomputed is not None:
        advantages = precomputed["advantages"]
        self_certainty = precomputed["self_certainty"]
        ch_cf = precomputed["ch_cf"]
        ch_sc = precomputed["ch_sc"]
    else:
        log_V = math.log(vocab_size)
        kl_uniform_pi = (-log_V - mean_log_probs) * mask
        lengths = mask.sum(dim=-1).clamp(min=1)
        self_certainty = kl_uniform_pi.sum(dim=-1) / lengths
        advantages, ch_cf, ch_sc = compute_advantages(
            cluster_rewards, format_rewards, self_certainty,
            group_idx, config,
        )

    # ── STEP 4: KL vs reference ──────────────────────────
    if config.beta_kl != 0.0 and ref_per_token_logps is not None:
        per_token_kl = (torch.exp(ref_per_token_logps - per_token_logps)
                        - (ref_per_token_logps - per_token_logps) - 1)
    elif config.beta_kl != 0.0:
        per_token_kl = (torch.exp(per_token_logps.detach() - per_token_logps)
                        - (per_token_logps.detach() - per_token_logps) - 1)
    else:
        per_token_kl = torch.zeros_like(per_token_logps)

    # ── STEP 5: KL-Cov ──────────────────────────────────
    covariances = compute_token_covariances(
        per_token_logps.detach(), advantages, mask)
    high_cov_mask = select_high_covariance_tokens(
        covariances, mask, config.top_k_fraction)
    normal_mask = mask * (1.0 - high_cov_mask)

    # ── STEP 6: Per-token loss ───────────────────────────
    # log(π_θ / π_θ_old) — the true importance sampling ratio.
    # When μ=1, π_θ = π_θ_old so log_ratio ≈ 0, ratio ≈ 1, clipping is a no-op.
    # When μ>1, π_θ drifts from π_θ_old across inner iterations, and clipping
    # prevents the policy from diverging too far from the generation policy.
    log_ratio = per_token_logps - old_per_token_logps
    ratio = torch.exp(log_ratio)

    adv = advantages.unsqueeze(1)  # (B, 1) → broadcast

    # Normal tokens: clipped surrogate (Eq. 21)
    clipped_ratio = torch.clamp(ratio,
                                1.0 - config.epsilon_clip,
                                1.0 + config.epsilon_clip)
    clipped_loss = -torch.min(ratio * adv, clipped_ratio * adv)

    # High-cov tokens: unclipped + KL-Cov penalty (Eq. 8-9)
    # Uses same log_ratio = log(π_θ / π_θ_old) for the anchor
    unclipped_loss = -ratio * adv
    kl_penalty = config.beta_cov * log_ratio.abs()
    penalised_loss = unclipped_loss + kl_penalty

    per_token_loss = clipped_loss * normal_mask + penalised_loss * high_cov_mask
    per_token_loss = per_token_loss + config.beta_kl * per_token_kl

    # Entropy bonus
    if config.entropy_bonus > 0:
        per_token_loss = per_token_loss - config.entropy_bonus * entropies

    loss = aggregate_loss(per_token_loss, mask,
                          config.loss_type, config.max_completion_length)
    loss_unscaled = loss.detach().item()
    loss = loss / gradient_accumulation_steps

    # ── STEP 7: Metrics ──────────────────────────────────
    n_high = high_cov_mask.sum().item()
    n_total = mask.sum().item()

    metrics: Dict[str, float] = {
        "loss": loss_unscaled,
        "rewards/self_certainty_mean": self_certainty.mean().item(),
        "rewards/cluster_mean": cluster_rewards.mean().item(),
        "rewards/format_mean": format_rewards.mean().item(),
        "advantages/ch_cluster_format_std": (
            ch_cf.std(correction=0).item() if ch_cf.numel() > 1 else 0.0),
        "advantages/ch_self_certainty_std": (
            ch_sc.std(correction=0).item() if ch_sc.numel() > 1 else 0.0),
        "advantages/combined_mean": advantages.mean().item(),
        "klcov/high_cov_fraction": n_high / max(n_total, 1),
        "klcov/mean_covariance": (
            (covariances * mask).sum().item() / max(n_total, 1)),
        "entropy": (
            (entropies * mask).sum().item() / max(n_total, 1)),
    }
    if config.beta_kl != 0.0:
        masked_kl = (per_token_kl * mask).sum() / mask.sum().clamp(min=1)
        metrics["kl"] = masked_kl.item()
    if n_high > 0:
        metrics["klcov/mean_kl_penalty"] = (
            (kl_penalty * high_cov_mask).sum().item() / n_high)

    return loss, metrics


# ════════════════════════════════════════════════════════════
# 9. REFERENCE MODEL LOG-PROBS
# ════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_ref_logprobs(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    logits_to_keep: int,
) -> torch.Tensor:
    """Compute per-token log-probs under the reference (base) model.

    For PEFT models we temporarily disable LoRA adapters so the forward
    pass uses only the frozen base weights.  This avoids loading a
    separate reference model copy.

    Uses RAW logits (no temperature) to match vLLM's raw logprobs.
    """
    had_adapters = False
    if hasattr(model, "disable_adapter_layers"):
        model.disable_adapter_layers()
        had_adapters = True

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, -(logits_to_keep + 1):-1, :]
    lse = logits.logsumexp(dim=-1)
    completion_ids = input_ids[:, -logits_to_keep:]
    ref_logps = (logits.gather(-1, completion_ids.unsqueeze(-1))
                 .squeeze(-1) - lse)

    if had_adapters:
        model.enable_adapter_layers()

    return ref_logps


# ════════════════════════════════════════════════════════════
# 10. TRAINER
# ════════════════════════════════════════════════════════════

class Trainer:
    def __init__(self, config: TrainConfig):
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
        self._run_number: Optional[int] = None  # set once on first permanent save

        # Resume from checkpoint if specified
        if config.resume_from:
            ckpt = config.resume_from.strip()
            if not os.path.isdir(ckpt):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
            self.lora_path = ckpt
            opt_path = os.path.join(ckpt, "optimizer.pt")
            sch_path = os.path.join(ckpt, "scheduler.pt")
            self._optimizer_path = opt_path if os.path.isfile(opt_path) else None
            self._scheduler_path = sch_path if os.path.isfile(sch_path) else None
            # Extract step number from checkpoint dir name (e.g. "checkpoint-200" → 200)
            ckpt_name = os.path.basename(ckpt.rstrip("/"))
            if ckpt_name.startswith("checkpoint-"):
                self.global_step = int(ckpt_name.split("-")[1])
            # Extract run number from parent dir (e.g. "run_3/checkpoint-200" → 3)
            parent = os.path.basename(os.path.dirname(ckpt.rstrip("/")))
            if parent.startswith("run_"):
                try:
                    self._run_number = int(parent.split("_")[1])
                except ValueError:
                    pass
            logger.info(f"Resuming from {ckpt}, step={self.global_step}, "
                        f"run={self._run_number}")

        if config.use_wandb:
            import wandb
            wandb_kwargs = dict(
                project=os.environ.get("WANDB_PROJECT", "multi-reward-scratch"),
                name=config.run_name,
                config=asdict(config),
            )
            if config.resume_from:
                wandb_kwargs["resume"] = "allow"
            wandb.init(**wandb_kwargs)

    # ─── total training steps ─────────────────────────────
    def _total_steps(self) -> int:
        if self.config.max_steps > 0:
            return self.config.max_steps
        n_prompts = len(self.prompts)
        steps_per_epoch = max(1, n_prompts // self.config.train_batch_size)
        return steps_per_epoch * self.config.num_train_epochs

    # ─── load model + LoRA + optim + scheduler ────────────
    def _init_training_model(self):
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

        self.vocab_size = model.config.vocab_size

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable,
                          lr=self.config.learning_rate,
                          weight_decay=self.config.weight_decay,
                          betas=(0.9, 0.999))

        total_steps = self._total_steps()
        # Scheduler sees total_steps * μ calls (one per GRPO inner iteration)
        total_scheduler_steps = total_steps * self.config.mu
        warmup_steps = int(total_scheduler_steps * self.config.warmup_ratio)
        schedule_fn = (get_cosine_schedule_with_warmup if self.config.lr_scheduler == "cosine"
                       else get_linear_schedule_with_warmup)
        scheduler = schedule_fn(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_scheduler_steps,
        )

        # Restore optimizer / scheduler state if available
        if self._optimizer_path and os.path.isfile(self._optimizer_path):
            optimizer.load_state_dict(
                torch.load(self._optimizer_path, map_location="cuda",
                           weights_only=True))
        if self._scheduler_path and os.path.isfile(self._scheduler_path):
            scheduler.load_state_dict(
                torch.load(self._scheduler_path, map_location="cpu",
                           weights_only=True))

        return model, optimizer, scheduler

    # ─── save LoRA + optimizer + scheduler ────────────────
    def _save_checkpoint(self, model, optimizer, scheduler, step: int,
                         permanent: bool = False):
        if permanent:
            # Determine run number ONCE per training run, reuse for all checkpoints.
            if self._run_number is None:
                base_output = self.config.output_dir
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

        opt_path = os.path.join(save_dir, "optimizer.pt")
        sch_path = os.path.join(save_dir, "scheduler.pt")
        torch.save(optimizer.state_dict(), opt_path)
        torch.save(scheduler.state_dict(), sch_path)

        self.lora_path = save_dir
        self._optimizer_path = opt_path
        self._scheduler_path = sch_path

        if permanent:
            logger.info(f"  Saved checkpoint → {save_dir}")

    # ─── free training model ──────────────────────────────
    @staticmethod
    def _free_model(*objs):
        # Clear optimizer state tensors explicitly (they hold CUDA refs)
        for o in objs:
            if hasattr(o, "state"):  # optimizer
                o.state.clear()
            if hasattr(o, "zero_grad"):
                try:
                    o.zero_grad(set_to_none=True)
                except Exception:
                    pass
        for o in objs:
            del o
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # Second pass — catches ref cycles
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # ─── pretty-print rollouts ────────────────────────────
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

                lines.append(
                    f"\n  ROLLOUT [{g}]  R_cluster={r_cl:.3f}  "
                    f"R_format={r_fm:.1f}{bstr}")
                lines.append(f"  {text}")

            if G > 4:
                lines.append(f"\n  … ({G - 4} more rollouts not shown)")
        lines.append(f"\n{'=' * 80}")
        logger.info("\n".join(lines))

    # ─── log step metrics ─────────────────────────────────
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

    # ─── MAIN TRAINING LOOP ──────────────────────────────
    def train(self):
        total_steps = self._total_steps()
        logger.info(f"=== Multi-Reward GRPO from Scratch ===")
        logger.info(f"Config: {asdict(self.config)}")
        logger.info(f"Total prompts: {len(self.prompts)}, "
                     f"total steps: {total_steps}")

        os.makedirs(self.config.output_dir, exist_ok=True)

        prompt_indices = list(range(len(self.prompts)))
        random.seed(self.config.seed)
        random.shuffle(prompt_indices)
        cursor = 0
        start_step = self.global_step

        # Fast-forward cursor to match resumed position (reproducible data order)
        # Each step consumes bs * ga_steps prompts (one bs per ga round)
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

        for step in range(start_step, total_steps):
            step_start = time.time()
            bs = self.config.train_batch_size

            # ── PHASE 3: TRAIN  (PyTorch) ────────────────
            # Load model ONCE per optimizer step, accumulate across ga_steps
            # generation rounds. Each round: generate → rewards → forward+backward.
            model, optimizer, scheduler = self._init_training_model()
            mu = self.config.mu
            step_metrics: Dict[str, List[float]] = defaultdict(list)
            all_full_batch_metrics: List[Dict[str, float]] = []

            # ════════════════════════════════════════════════════
            # PHASE 1: GENERATE all GA rounds ONCE (Algorithm 1, line 7-8)
            # Completions + old_logps + advantages are FIXED for all μ iterations.
            # ════════════════════════════════════════════════════
            ga_round_data: List[Dict[str, Any]] = []

            for ga_idx in range(ga_steps):
                # ── sample prompts ────────────────────
                if cursor + bs > len(prompt_indices):
                    random.shuffle(prompt_indices)
                    cursor = 0
                batch_indices = prompt_indices[cursor:cursor + bs]
                cursor += bs
                batch_prompts = [self.prompts[i] for i in batch_indices]

                # ── GENERATE (vLLM) ──────────────────
                logger.info(f"\n\n Step {step}/{total_steps} "
                            f"[ga {ga_idx+1}/{ga_steps}]: generating "
                            f"{bs}×{self.config.num_generations} completions …")

                # Free ALL GPU memory before vLLM
                model.cpu()
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and v.is_cuda:
                            state[k] = v.cpu()
                if hasattr(model, 'zero_grad'):
                    model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                    torch.cuda.reset_accumulated_memory_stats()
                gc.collect()
                torch.cuda.empty_cache()

                free_mem, total_mem = torch.cuda.mem_get_info()
                used_gb = (total_mem - free_mem) / 1e9
                logger.info(f"  GPU memory before vLLM: {used_gb:.1f}GB used, {free_mem/1e9:.1f}GB free")

                gen = GenerationEngine(self.config, lora_path=self.lora_path)
                rollouts = gen.generate_rollouts(batch_prompts)
                gen.shutdown()
                del gen

                # Move model back to GPU
                model.cuda()
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

                # ── REWARDS (CPU) ────────────────────
                rewards = compute_all_rewards(rollouts, self.config)

                if (self.config.print_rollouts_every > 0
                        and step % self.config.print_rollouts_every == 0
                        and ga_idx == 0):
                    self._print_rollouts(rollouts, rewards, step)

                # ── Prepare batch ────────────────────
                training_batch = prepare_training_batch(
                    rollouts, rewards, self.tokenizer, self.config)
                del rollouts  # free raw rollout data, keep tensors

                device = torch.device("cuda")
                for k, v in training_batch.items():
                    if isinstance(v, torch.Tensor):
                        training_batch[k] = v.to(device)

                B = training_batch["input_ids"].shape[0]

                # ── Pre-compute advantages (FIXED for all μ) ──
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

                # Store this round's data for reuse across μ iterations
                ga_round_data.append({
                    "training_batch": training_batch,
                    "precomputed": precomputed_full,
                    "B": B,
                })

            # ════════════════════════════════════════════════════
            # PHASE 2: TRAIN for μ iterations over the SAME data
            # (Algorithm 1, line 10). π_θ drifts, old_logps stay fixed.
            # ════════════════════════════════════════════════════
            mini_bs = 1
            actual_ga = ga_round_data[0]["B"] * ga_steps  # B × ga_steps

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

                # ── Optimiser step (after all ga rounds for this μ) ──
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                ).item()

                optimizer.step()
                scheduler.step()

            # Free all stored GA round data
            del ga_round_data
            gc.collect()
            torch.cuda.empty_cache()

            # ── Save LoRA + optimiser + scheduler ─────────
            self._save_checkpoint(model, optimizer, scheduler, step + 1,
                                  permanent=((step + 1) % self.config.save_steps == 0))

            current_lr = scheduler.get_last_lr()[0]
            self._free_model(model, optimizer, scheduler)
            # Delete local references — _free_model only deletes its own copies
            del model, optimizer, scheduler
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

            # ── Logging ───────────────────────────────────
            avg_metrics = {k: sum(v) / len(v) for k, v in step_metrics.items()}
            # Average full-batch metrics across ga rounds
            full_batch_metrics = {
                k: sum(d[k] for d in all_full_batch_metrics) / len(all_full_batch_metrics)
                for k in all_full_batch_metrics[0]
            }
            # Override with accurate full-batch metrics (mini-batch std is wrong)
            avg_metrics.update(full_batch_metrics)
            avg_metrics["grad_norm"] = grad_norm
            avg_metrics["lr"] = current_lr
            avg_metrics["step_time"] = time.time() - step_start

            # Track GPU memory usage
            free_mem, total_mem = torch.cuda.mem_get_info()
            avg_metrics["gpu/used_gb"] = (total_mem - free_mem) / 1e9
            avg_metrics["gpu/free_gb"] = free_mem / 1e9

            if step % self.config.logging_steps == 0:
                self._log_step(step, avg_metrics)

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
# 12. MAIN
# ════════════════════════════════════════════════════════════

def main():
    config = parse_args()
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
