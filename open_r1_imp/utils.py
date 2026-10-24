"""Shared helpers: model/tokenizer/dataset loading, GDPO normalization, self-certainty."""

import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ════════════════════════════════════════════════════════════
# LOADING
# ════════════════════════════════════════════════════════════

def get_model(model_args):
    dtype = getattr(model_args, "dtype", None)
    torch_dtype = getattr(torch, dtype) if dtype and dtype != "auto" else torch.bfloat16
    kwargs = dict(trust_remote_code=True, torch_dtype=torch_dtype)
    if getattr(model_args, "attn_implementation", None):
        kwargs["attn_implementation"] = model_args.attn_implementation
    return AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, revision=getattr(model_args, "model_revision", "main"), **kwargs)


def get_tokenizer(model_args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, revision=getattr(model_args, "model_revision", "main"),
        trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        raise ValueError(f"{model_args.model_name_or_path} has no chat template.")
    return tokenizer


def get_dataset(script_args):
    """Hub dataset name, or a local directory of data files. `$VARS` are expanded."""
    return load_dataset(os.path.expandvars(script_args.dataset_name),
                        name=getattr(script_args, "dataset_config", None))


# ════════════════════════════════════════════════════════════
# GDPO NORMALIZATION
# ════════════════════════════════════════════════════════════

def gdpo_normalize(rewards: torch.Tensor, group_idx: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Z-score `rewards` within each prompt group (population std).

    Groups with zero std (or a single element) get zero advantage.
    """
    normalized = torch.zeros_like(rewards)
    for gid in group_idx.unique():
        mask = group_idx == gid
        group = rewards[mask]
        if group.numel() <= 1:
            continue
        sigma = group.std(correction=0)
        if sigma > eps:
            normalized[mask] = (group - group.mean()) / (sigma + eps)
    return normalized


# ════════════════════════════════════════════════════════════
# SELF-CERTAINTY
# ════════════════════════════════════════════════════════════

def compute_self_certainty_per_token(logits: torch.Tensor) -> torch.Tensor:
    """logsumexp(logits) - mean(logits) = KL(U || pi) + log|V| (the constant
    cancels under group normalization). Raw logits, no temperature (as INTUITOR)."""
    return torch.logsumexp(logits, dim=-1) - logits.mean(dim=-1)


def aggregate_self_certainty(per_token_sc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean per-token self-certainty over the completion tokens."""
    return (per_token_sc * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)


@torch.no_grad()
def compute_self_certainty(model, prompt_ids, prompt_mask, completion_ids, completion_mask,
                           batch_size: int = 4) -> torch.Tensor:
    """Per-completion self-certainty from the current policy, (B,).

    Position t of the logits predicts token t+1, so the distributions over the
    completion tokens are logits[:, -(C+1):-1].
    """
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    C = completion_ids.size(1)
    was_training = model.training
    model.eval()
    chunks = []
    for start in range(0, input_ids.size(0), batch_size):
        logits = model(input_ids=input_ids[start:start + batch_size],
                       attention_mask=attention_mask[start:start + batch_size]).logits
        chunks.append(compute_self_certainty_per_token(logits[:, -(C + 1):-1, :]))
        del logits
    if was_training:
        model.train()
    return aggregate_self_certainty(torch.cat(chunks, dim=0), completion_mask)
