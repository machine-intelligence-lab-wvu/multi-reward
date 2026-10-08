"""
Multi-Reward Unsupervised GRPO — Thinking Mode (Qwen3).

Same as train_persistent.py but with Qwen3's thinking mode enabled:
  - enable_thinking=True in chat template
  - Self-certainty computed on REASONING tokens only (<think>...</think>)
  - Cluster voting computed on ANSWER tokens only (after </think>)
  - Model generates: <think>reasoning...</think>answer with \\boxed{}

This separates the reward signals:
  - Self-certainty rewards REASONING CONFIDENCE (not answer confidence)
  - Cluster voting rewards ANSWER CONSISTENCY
  - Prevents template collapse (short \\boxed{0} has no reasoning → SC ≈ 0)

Requires Qwen3 models (Qwen/Qwen3-1.7B, Qwen/Qwen3-4B, etc.)
Requires >= 40GB VRAM (thinking mode generates longer completions).

Qwen3 special tokens:
  <think>   = 151667
  </think>  = 151668
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
    TrainConfig, parse_args, gdpo_normalize,
    forward_with_mean_logprobs, compute_grpo_loss, compute_ref_logprobs,
    aggregate_loss,
)
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

# Qwen3 special token IDs
THINK_START_ID = 151667  # <think>
THINK_END_ID = 151668    # </think>
THINK_END_STR = "</think>"


# ════════════════════════════════════════════════════════════
# THINKING-AWARE FUNCTIONS
# ════════════════════════════════════════════════════════════

def split_thinking_answer(text: str) -> Tuple[str, str]:
    """Split a thinking-mode completion into (reasoning, answer).

    Input:  "<think>Let me solve... 2+3=5.</think>The answer is \\boxed{5}"
    Output: ("Let me solve... 2+3=5.", "The answer is \\boxed{5}")

    If no </think> found, entire text is treated as reasoning, answer is empty.
    """
    idx = text.rfind(THINK_END_STR)
    if idx >= 0:
        reasoning = text[:idx]
        # Strip <think> tag from start if present
        if reasoning.startswith("<think>"):
            reasoning = reasoning[len("<think>"):]
        answer = text[idx + len(THINK_END_STR):]
        return reasoning.strip(), answer.strip()
    else:
        return text.strip(), ""


def compute_rewards_thinking(rollouts: Dict, config: TrainConfig) -> Dict[str, torch.Tensor]:
    """Compute rewards for thinking-mode completions.

    Cluster voting and format reward are computed on the ANSWER portion only
    (text after </think>), not the reasoning. This ensures:
      - Cluster voting compares final answers, not reasoning traces
      - Format reward checks \\boxed{} in the answer, not in reasoning
      - Ready for future semantic similarity models on the answer portion

    Self-certainty is handled separately in precompute_advantages_thinking
    (uses reasoning_mask on token-level logits).
    """
    P = len(rollouts["completion_texts"])
    G = config.num_generations
    min_tokens = config.min_completion_tokens

    cluster_rewards: List[float] = []
    format_rewards: List[float] = []

    for p_idx in range(P):
        group_texts = rollouts["completion_texts"][p_idx]

        # Extract ANSWER portion only for each completion
        group_answers = []
        for text in group_texts:
            _, answer = split_thinking_answer(text)
            group_answers.append(answer)

        # Cluster voting on answer portions only
        group_cluster, _ = compute_cluster_rewards(group_answers)

        for g_idx, text in enumerate(group_texts):
            comp_len = len(rollouts["completion_token_ids"][p_idx][g_idx])
            if min_tokens > 0 and comp_len < min_tokens:
                cluster_rewards.append(0.0)
                format_rewards.append(0.0)
            else:
                cluster_rewards.append(group_cluster[g_idx])
                # Format reward on answer portion only
                _, answer = split_thinking_answer(text)
                format_rewards.append(compute_format_reward(answer))

    return {
        "cluster_rewards": torch.tensor(cluster_rewards, dtype=torch.float32),
        "format_rewards": torch.tensor(format_rewards, dtype=torch.float32),
    }

def prepare_dataset_thinking(tokenizer, dataset_name: str, split: str = "train",
                             max_samples: Optional[int] = None) -> List[str]:
    """Prepare prompts with enable_thinking=True for Qwen3."""
    system_prompt = ("Please reason step by step, and put your final answer within \\boxed{}.")

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
        msgs = [
            {"role": "user", "content": q},
        ]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append(prompt)

    logger.info(f"Dataset '{dataset_name}' (thinking mode): {len(prompts)} prompts")
    return prompts


def build_reasoning_mask(completion_ids_tensor: torch.Tensor,
                         completion_mask: torch.Tensor) -> torch.Tensor:
    """Build a mask that's 1 for reasoning tokens (inside <think>...</think>), 0 elsewhere.

    Qwen3 with enable_thinking=True generates:
      <think> reasoning tokens... </think> answer tokens...

    The <think> token is at the start of generation (added by the template).
    We look for </think> (token 151668) to find where reasoning ends.

    If no </think> found (model didn't finish thinking), the entire
    completion is treated as reasoning.
    """
    B, S = completion_ids_tensor.shape
    reasoning_mask = torch.zeros_like(completion_mask)

    for i in range(B):
        # Find </think> position in this completion
        think_end_positions = (completion_ids_tensor[i] == THINK_END_ID).nonzero(as_tuple=True)[0]

        if len(think_end_positions) > 0:
            # Use the LAST </think> position
            end_pos = think_end_positions[-1].item()
            # Reasoning = everything before </think> (excluding the token itself)
            # Skip position 0 if it's <think> tag
            start_pos = 0
            if completion_ids_tensor[i, 0] == THINK_START_ID:
                start_pos = 1  # skip <think> token
            reasoning_mask[i, start_pos:end_pos] = 1.0
        else:
            # No </think> found — treat entire completion as reasoning
            reasoning_mask[i, :] = 1.0

    # Apply completion_mask to zero out padding
    reasoning_mask = reasoning_mask * completion_mask
    return reasoning_mask


def prepare_training_batch_thinking(
    rollouts: Dict,
    rewards: Dict[str, torch.Tensor],
    tokenizer,
    config: TrainConfig,
) -> Dict[str, Any]:
    """Like prepare_training_batch but also builds reasoning_mask.

    Returns all standard fields plus:
      reasoning_mask: (B, max_comp_len) — 1 for tokens inside <think>...</think>
    """
    P = len(rollouts["completion_texts"])
    G = config.num_generations
    B = P * G

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
    completion_ids_tensor = torch.full((B, max_comp_len), pad_id, dtype=torch.long)
    old_per_token_logps = torch.zeros((B, max_comp_len), dtype=torch.float32)

    for i in range(B):
        p_ids = all_prompt_ids[i]
        c_ids = all_comp_ids[i]
        prompt_pad = max_prompt_len - len(p_ids)

        input_ids[i, prompt_pad:max_prompt_len] = torch.tensor(p_ids, dtype=torch.long)
        comp_end = max_prompt_len + len(c_ids)
        input_ids[i, max_prompt_len:comp_end] = torch.tensor(c_ids, dtype=torch.long)
        attention_mask[i, prompt_pad:comp_end] = 1
        completion_mask[i, :len(c_ids)] = 1.0
        completion_ids_tensor[i, :len(c_ids)] = torch.tensor(c_ids, dtype=torch.long)
        old_lp = all_old_logps[i]
        old_per_token_logps[i, :len(old_lp)] = torch.tensor(old_lp, dtype=torch.float32)

    # Build reasoning mask from </think> positions
    reasoning_mask = build_reasoning_mask(completion_ids_tensor, completion_mask)

    # Log reasoning vs answer token stats
    n_reasoning = reasoning_mask.sum().item()
    n_total = completion_mask.sum().item()
    logger.info(f"  Thinking split: {n_reasoning:.0f} reasoning tokens, "
                f"{n_total - n_reasoning:.0f} answer tokens "
                f"({n_reasoning/max(n_total,1)*100:.1f}% reasoning)")

    prompt_group_idx = torch.arange(B, dtype=torch.long) // G

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "completion_ids": completion_ids_tensor,
        "reasoning_mask": reasoning_mask,
        "old_per_token_logps": old_per_token_logps,
        "cluster_rewards": rewards["cluster_rewards"],
        "format_rewards": rewards["format_rewards"],
        "prompt_group_idx": prompt_group_idx,
        "max_comp_len": max_comp_len,
    }


@torch.no_grad()
def precompute_advantages_thinking(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    reasoning_mask: torch.Tensor,
    cluster_rewards: torch.Tensor,
    format_rewards: torch.Tensor,
    group_idx: torch.Tensor,
    logits_to_keep: int,
    vocab_size: int,
    config: TrainConfig,
    chunk_size: int = 2,
) -> Dict[str, torch.Tensor]:
    """Pre-compute advantages with self-certainty on REASONING tokens only.

    Self-certainty is averaged over reasoning tokens (inside <think>...</think>).
    Completions with no reasoning tokens get SC = group minimum (penalized).
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
        lse = logits.logsumexp(dim=-1)
        mean_lp = logits.mean(dim=-1) - lse

        # Use REASONING mask instead of completion mask for self-certainty
        r_mask = reasoning_mask[start:end]
        kl_u_pi = (-log_V - mean_lp) * r_mask
        r_lengths = r_mask.sum(dim=-1).clamp(min=1)
        sc = kl_u_pi.sum(dim=-1) / r_lengths
        all_sc.append(sc)
        del outputs, logits, lse, mean_lp

    self_certainty = torch.cat(all_sc)

    # Completions with no reasoning tokens (reasoning_mask all zeros) get penalized
    no_reasoning = reasoning_mask.sum(dim=-1) == 0
    if no_reasoning.any():
        for gid in group_idx.unique():
            g_mask = group_idx == gid
            group_sc = self_certainty[g_mask]
            no_reason_in_group = no_reasoning[g_mask]
            if no_reason_in_group.any() and (~no_reason_in_group).any():
                group_min = group_sc[~no_reason_in_group].min()
                self_certainty[g_mask & no_reasoning] = group_min
            elif no_reason_in_group.all():
                self_certainty[g_mask] = 0.0

    # Also apply min_completion_tokens filter
    if config.min_completion_tokens > 0:
        lengths = completion_mask.sum(dim=-1)
        short_mask = lengths < config.min_completion_tokens
        if short_mask.any():
            for gid in group_idx.unique():
                g_mask = group_idx == gid
                group_sc = self_certainty[g_mask]
                short_in_group = short_mask[g_mask]
                if short_in_group.any() and (~short_in_group).any():
                    group_min = group_sc[~short_in_group].min()
                    self_certainty[g_mask & short_mask] = group_min
                elif short_in_group.all():
                    self_certainty[g_mask] = 0.0

    from train_from_scratch import compute_advantages
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
# PERSISTENT THINKING TRAINER
# ════════════════════════════════════════════════════════════

class ThinkingTrainer:
    """Persistent trainer with Qwen3 thinking mode.

    Self-certainty is computed on reasoning tokens only.
    Both vLLM and PyTorch stay loaded (requires >= 40GB VRAM).
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Verify model supports thinking mode
        if "qwen3" not in config.model_name.lower() and "qwen-3" not in config.model_name.lower():
            logger.warning(f"Model {config.model_name} may not support thinking mode. "
                          f"This script is designed for Qwen3 models.")

        self.prompts = prepare_dataset_thinking(
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

        # Load PyTorch model (persistent)
        logger.info("Loading PyTorch training model (persistent)...")
        self.model = self._load_training_model()
        self.vocab_size = self.model.config.vocab_size

        # Create optimizer + scheduler
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=config.learning_rate,
                               weight_decay=config.weight_decay, betas=(0.9, 0.999))

        total_steps = self._total_steps()
        total_scheduler_steps = total_steps * config.mu
        warmup_steps = int(total_scheduler_steps * config.warmup_ratio)
        schedule_fn = (get_cosine_schedule_with_warmup if config.lr_scheduler == "cosine"
                       else get_linear_schedule_with_warmup)
        self.scheduler = schedule_fn(self.optimizer,
                                     num_warmup_steps=warmup_steps,
                                     num_training_steps=total_scheduler_steps)

        if config.resume_from:
            opt_path = os.path.join(config.resume_from, "optimizer.pt")
            sch_path = os.path.join(config.resume_from, "scheduler.pt")
            if os.path.isfile(opt_path):
                self.optimizer.load_state_dict(
                    torch.load(opt_path, map_location="cuda", weights_only=True))
            if os.path.isfile(sch_path):
                self.scheduler.load_state_dict(
                    torch.load(sch_path, map_location="cpu", weights_only=True))

        # Load vLLM (persistent)
        logger.info("Loading vLLM generation engine (persistent, thinking mode)...")
        self.vllm_engine = self._load_vllm()

        # wandb
        if config.use_wandb:
            import wandb
            wandb_kwargs = dict(
                project=os.environ.get("WANDB_PROJECT", "multi-reward-thinking"),
                name=config.run_name, config=asdict(config),
            )
            if config.resume_from:
                wandb_kwargs["resume"] = "allow"
            wandb.init(**wandb_kwargs)

        logger.info(f"Thinking trainer ready. VRAM: "
                    f"{torch.cuda.memory_allocated()/1e9:.1f}GB allocated")

    def _load_training_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True).to("cuda")

        if self.lora_path is not None:
            model = PeftModel.from_pretrained(model, self.lora_path, is_trainable=True)
        else:
            lora_cfg = LoraConfig(
                r=self.config.lora_rank, lora_alpha=self.config.lora_alpha,
                target_modules="all-linear", task_type="CAUSAL_LM",
                lora_dropout=self.config.lora_dropout)
            model = get_peft_model(model, lora_cfg)

        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.train()
        return model

    def _load_vllm(self):
        from vllm import LLM
        return LLM(
            model=self.config.model_name,
            trust_remote_code=True, dtype="bfloat16",
            max_model_len=self.config.max_prompt_length + self.config.max_completion_length,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            enforce_eager=True, seed=self.config.seed,
            enable_lora=True, max_lora_rank=self.config.lora_rank,
        )

    def _generate_rollouts(self, prompts: List[str]) -> Dict[str, Any]:
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

    def _total_steps(self) -> int:
        if self.config.max_steps > 0:
            return self.config.max_steps
        n = len(self.prompts)
        spe = max(1, n // (self.config.train_batch_size * self.config.gradient_accumulation_steps))
        return spe * self.config.num_train_epochs

    def _save_checkpoint(self, step: int, permanent: bool = False):
        if permanent:
            if self._run_number is None:
                os.makedirs(self.config.output_dir, exist_ok=True)
                existing = glob.glob(os.path.join(self.config.output_dir, "run_*"))
                nums = [int(r.split("_")[-1]) for r in existing if r.split("_")[-1].isdigit()]
                self._run_number = max(nums) + 1 if nums else 0
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
        for k in ["loss", "grad_norm", "lr", "kl", "entropy",
                   "rewards/cluster_mean", "rewards/format_mean",
                   "rewards/self_certainty_mean",
                   "reasoning/frac_with_think", "reasoning/avg_reasoning_tokens",
                   "advantages/combined_mean",
                   "advantages/ch_cluster_format_std",
                   "advantages/ch_self_certainty_std",
                   "step_time"]:
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
        config = self.config
        logger.info(f"=== Multi-Reward GRPO — Thinking Mode ===")
        logger.info(f"Config: {asdict(config)}")
        logger.info(f"Total prompts: {len(self.prompts)}, total steps: {total_steps}")

        os.makedirs(config.output_dir, exist_ok=True)

        prompt_indices = list(range(len(self.prompts)))
        random.seed(config.seed)
        random.shuffle(prompt_indices)
        cursor = 0
        start_step = self.global_step

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

            # ── PHASE 1: GENERATE all GA rounds ──
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
                            f"{bs}×{config.num_generations} completions (thinking mode)…")

                gen_start = time.time()
                rollouts = self._generate_rollouts(batch_prompts)
                logger.info(f"  Generation took {time.time() - gen_start:.1f}s")

                rewards = compute_rewards_thinking(rollouts, config)

                # Use thinking-aware batch preparation
                training_batch = prepare_training_batch_thinking(
                    rollouts, rewards, self.tokenizer, config)
                del rollouts

                device = torch.device("cuda")
                for k, v in training_batch.items():
                    if isinstance(v, torch.Tensor):
                        training_batch[k] = v.to(device)

                B = training_batch["input_ids"].shape[0]

                # Pre-compute advantages with REASONING-ONLY self-certainty
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

                # Thinking-specific metrics
                reasoning_mask = training_batch["reasoning_mask"]
                completion_mask = training_batch["completion_mask"]
                n_with_think = (reasoning_mask.sum(dim=-1) > 0).float().mean().item()
                avg_reasoning_tokens = reasoning_mask.sum(dim=-1).mean().item()

                all_full_batch_metrics.append({
                    "rewards/self_certainty_mean": precomputed_full["self_certainty"].mean().item(),
                    "advantages/ch_cluster_format_std": (
                        precomputed_full["ch_cf"].std(correction=0).item()
                        if precomputed_full["ch_cf"].numel() > 1 else 0.0),
                    "advantages/ch_self_certainty_std": (
                        precomputed_full["ch_sc"].std(correction=0).item()
                        if precomputed_full["ch_sc"].numel() > 1 else 0.0),
                    "advantages/combined_mean": precomputed_full["advantages"].mean().item(),
                    "reasoning/frac_with_think": n_with_think,
                    "reasoning/avg_reasoning_tokens": avg_reasoning_tokens,
                })

                ga_round_data.append({
                    "training_batch": training_batch,
                    "precomputed": precomputed_full,
                    "B": B,
                })

            # ── PHASE 2: TRAIN for μ iterations ──
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
                    config.max_grad_norm).item()

                optimizer.step()
                scheduler.step()

            del ga_round_data
            gc.collect()
            torch.cuda.empty_cache()

            # ── Save + Log ──
            self._save_checkpoint(step + 1,
                                  permanent=((step + 1) % config.save_steps == 0))

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

        # Final save
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

    trainer = ThinkingTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
