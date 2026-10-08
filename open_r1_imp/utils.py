"""
Shared utilities for the Multi-Reward GRPO implementation.

Self-contained — no Open-R1 dependency. Only uses TRL, transformers, datasets, torch.

Contents:
  - Model/dataset loading wrappers (replaces Open-R1's utils)
  - GDPO normalization
  - Thinking mode utilities (reasoning mask, text splitting)
  - Self-certainty computation
  - KL-Cov (imported from callbacks/)
"""

import logging
import math
from typing import Tuple, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import KL-Cov functions directly from existing code (same repo)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from callbacks.klcov import compute_token_covariances, select_high_covariance_tokens

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# MODEL / DATASET / TOKENIZER LOADING
# (replaces open_r1.utils — simple wrappers, no external dependency)
# ════════════════════════════════════════════════════════════

def get_model(model_args, training_args):
    """Load a causal LM model.

    Replaces open_r1.utils.get_model(). Handles dtype and trust_remote_code.
    """
    model_kwargs = dict(trust_remote_code=True)
    dtype_str = getattr(model_args, "dtype", None)
    if dtype_str and dtype_str != "auto":
        model_kwargs["torch_dtype"] = getattr(torch, dtype_str, torch.bfloat16)
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    if hasattr(model_args, "attn_implementation") and model_args.attn_implementation:
        model_kwargs["attn_implementation"] = model_args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=getattr(model_args, "model_revision", "main"),
        **model_kwargs,
    )
    return model


def get_tokenizer(model_args, training_args):
    """Load tokenizer with padding configured.

    Replaces open_r1.utils.get_tokenizer().
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=getattr(model_args, "model_revision", "main"),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Base models without a chat template (e.g. meta-llama/Llama-3.2-3B) would
    # crash in TRL's maybe_apply_chat_template. A template set here is also
    # saved into every checkpoint, so downstream eval picks it up for free.
    ct = getattr(training_args, "chat_template", None)
    if ct:
        tokenizer.chat_template = ct
    elif tokenizer.chat_template is None:
        raise ValueError(
            f"{model_args.model_name_or_path} has no chat template. Set "
            "`chat_template:` in the recipe YAML (see B6_llama32_3b_*.yaml).")
    return tokenizer


def get_dataset(script_args):
    """Load dataset from HuggingFace hub.

    Replaces open_r1.utils.get_dataset(). Supports train/test splits.
    """
    dataset = load_dataset(
        script_args.dataset_name,
        name=getattr(script_args, "dataset_config", None),
    )
    return dataset

# Qwen3 special token IDs for thinking mode
THINK_START_ID = 151667  # <think>
THINK_END_ID = 151668    # </think>
THINK_END_STR = "</think>"


# ════════════════════════════════════════════════════════════
# GDPO NORMALIZATION (Eq. 3-4)
# ════════════════════════════════════════════════════════════

def gdpo_normalize(rewards: torch.Tensor, group_idx: torch.Tensor,
                   eps: float = 1e-8) -> torch.Tensor:
    """Z-score normalize rewards within each prompt group.

    For each group of G completions sharing the same prompt:
      z_i = (r_i - mean(r)) / (std(r) + eps)

    Groups with zero std or single elements get zero advantages.

    Args:
        rewards: (B,) reward values
        group_idx: (B,) integer group IDs (e.g., [0,0,0,0,0,0,0, 1,1,1,...])
        eps: small constant for numerical stability

    Returns:
        (B,) z-scored advantages
    """
    normalized = torch.zeros_like(rewards)
    for gid in group_idx.unique():
        mask = group_idx == gid
        group = rewards[mask]
        if group.numel() <= 1:
            continue
        mu = group.mean()
        sigma = group.std(correction=0)
        if sigma > eps:
            normalized[mask] = (group - mu) / (sigma + eps)
    return normalized


# ════════════════════════════════════════════════════════════
# THINKING MODE UTILITIES (Qwen3)
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
        if reasoning.startswith("<think>"):
            reasoning = reasoning[len("<think>"):]
        answer = text[idx + len(THINK_END_STR):]
        return reasoning.strip(), answer.strip()
    else:
        return text.strip(), ""


def build_reasoning_mask(completion_ids: torch.Tensor,
                         completion_mask: torch.Tensor) -> torch.Tensor:
    """Build a mask that's 1 for reasoning tokens (inside <think>...</think>), 0 elsewhere.

    Qwen3 with enable_thinking=True generates:
      <think> reasoning tokens... </think> answer tokens...

    The <think> token is typically the first completion token (added by template).
    We find </think> (token 151668) to determine where reasoning ends.

    If no </think> found (model didn't finish thinking), the entire
    completion is treated as reasoning.

    Args:
        completion_ids: (B, S) token IDs of completions
        completion_mask: (B, S) float mask (1 for real tokens, 0 for padding)

    Returns:
        (B, S) float reasoning mask
    """
    B, S = completion_ids.shape
    reasoning_mask = torch.zeros_like(completion_mask)

    for i in range(B):
        think_end_positions = (completion_ids[i] == THINK_END_ID).nonzero(as_tuple=True)[0]

        if len(think_end_positions) > 0:
            end_pos = think_end_positions[-1].item()
            start_pos = 0
            if completion_ids[i, 0] == THINK_START_ID:
                start_pos = 1  # skip <think> token itself
            reasoning_mask[i, start_pos:end_pos] = 1.0
        else:
            # No </think> found — treat entire completion as reasoning
            reasoning_mask[i, :] = 1.0

    reasoning_mask = reasoning_mask * completion_mask
    return reasoning_mask


# ════════════════════════════════════════════════════════════
# SELF-CERTAINTY COMPUTATION
# ════════════════════════════════════════════════════════════

def compute_self_certainty_per_token(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-token self-certainty from logits.

    Self-certainty = logsumexp(logits) - mean(logits) per position.
    This equals KL(U||π) + log|V| (constant cancels in z-scoring).

    Uses RAW logits (no temperature scaling), matching Intuitor's
    _get_per_token_self_certainty() implementation.

    Args:
        logits: (B, S, V) raw model logits

    Returns:
        (B, S) per-token self-certainty scores
    """
    return torch.logsumexp(logits, dim=-1) - logits.mean(dim=-1)


def aggregate_self_certainty(per_token_sc: torch.Tensor,
                             mask: torch.Tensor) -> torch.Tensor:
    """Average per-token self-certainty over masked positions.

    Args:
        per_token_sc: (B, S) per-token self-certainty
        mask: (B, S) float mask (completion_mask or reasoning_mask)

    Returns:
        (B,) averaged self-certainty per sample
    """
    lengths = mask.sum(dim=-1).clamp(min=1)
    return (per_token_sc * mask).sum(dim=-1) / lengths


def penalize_short_completions(self_certainty: torch.Tensor,
                               completion_mask: torch.Tensor,
                               group_idx: torch.Tensor,
                               min_tokens: int) -> torch.Tensor:
    """Set self-certainty of short completions to group minimum.

    Prevents template collapse where short confident outputs
    (e.g., "\\boxed{0}") get high self-certainty rewards.

    Args:
        self_certainty: (B,) self-certainty scores
        completion_mask: (B, S) completion mask
        group_idx: (B,) group indices
        min_tokens: minimum completion length threshold

    Returns:
        (B,) modified self-certainty (in-place modification)
    """
    if min_tokens <= 0:
        return self_certainty

    lengths = completion_mask.sum(dim=-1)
    short_mask = lengths < min_tokens

    if not short_mask.any():
        return self_certainty

    for gid in group_idx.unique():
        g_mask = group_idx == gid
        group_sc = self_certainty[g_mask]
        short_in_group = short_mask[g_mask]

        if short_in_group.any() and (~short_in_group).any():
            group_min = group_sc[~short_in_group].min()
            self_certainty[g_mask & short_mask] = group_min
        elif short_in_group.all():
            self_certainty[g_mask] = 0.0

    return self_certainty


def penalize_no_reasoning(self_certainty: torch.Tensor,
                          reasoning_mask: torch.Tensor,
                          group_idx: torch.Tensor) -> torch.Tensor:
    """Penalize completions with no reasoning tokens (thinking mode).

    Completions where reasoning_mask is all zeros get self-certainty
    set to group minimum (or 0 if all in group lack reasoning).

    Args:
        self_certainty: (B,) self-certainty scores
        reasoning_mask: (B, S) reasoning mask
        group_idx: (B,) group indices

    Returns:
        (B,) modified self-certainty
    """
    no_reasoning = reasoning_mask.sum(dim=-1) == 0

    if not no_reasoning.any():
        return self_certainty

    for gid in group_idx.unique():
        g_mask = group_idx == gid
        group_sc = self_certainty[g_mask]
        no_reason_in_group = no_reasoning[g_mask]

        if no_reason_in_group.any() and (~no_reason_in_group).any():
            group_min = group_sc[~no_reason_in_group].min()
            self_certainty[g_mask & no_reasoning] = group_min
        elif no_reason_in_group.all():
            self_certainty[g_mask] = 0.0

    return self_certainty
