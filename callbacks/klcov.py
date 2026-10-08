"""
KL-Cov: Targeted Covariance-Aware Regularization.

Implements the KL-Cov mechanism from Cui et al. 2025
("The Entropy Mechanism of RL for Reasoning LMs", arXiv:2505.22617).

The mechanism:
1. Pool all response tokens in the batch into N tokens.
2. For each token y_i, compute Cov(y_i) = (log_pi - mean_log_pi) * (A_i - mean_A).
3. Select top-k% tokens by covariance as I_KL.
4. Normal tokens: standard clipped GRPO loss.
5. High-cov tokens: unclipped ratio * advantage + beta_cov * |log_pi - log_pi_old|.
"""
import torch
from typing import Tuple


def compute_token_covariances(log_probs, advantages, response_mask):
    """
    Compute per-token covariance between log-probability and advantage.
    
    Args:
        log_probs: (B, S) log-probs of chosen tokens under current policy.
        advantages: (B,) per-rollout advantages, broadcast to (B, S).
        response_mask: (B, S) binary mask for response tokens.
    Returns:
        covariances: (B, S) per-token covariance values.
    """
    B, S = log_probs.shape
    adv_per_token = advantages.unsqueeze(1).expand(B, S)
    total_tokens = response_mask.sum().clamp(min=1)
    mean_lp = (log_probs * response_mask).sum() / total_tokens
    mean_adv = (adv_per_token * response_mask).sum() / total_tokens
    centered_lp = (log_probs - mean_lp) * response_mask
    centered_adv = (adv_per_token - mean_adv) * response_mask
    return centered_lp * centered_adv


def select_high_covariance_tokens(covariances, response_mask, top_k_fraction=0.02):
    """Select top-k fraction of tokens by covariance magnitude."""
    flat_cov = covariances.view(-1)
    flat_mask = response_mask.view(-1)
    valid_indices = flat_mask.nonzero(as_tuple=True)[0]
    n_valid = valid_indices.shape[0]
    if n_valid == 0:
        return torch.zeros_like(response_mask)
    k = max(1, int(top_k_fraction * n_valid))
    valid_covs = flat_cov[valid_indices]
    _, topk_local = valid_covs.topk(k, largest=True)
    topk_global = valid_indices[topk_local]
    high_cov_flat = torch.zeros_like(flat_mask)
    high_cov_flat[topk_global] = 1.0
    return high_cov_flat.view_as(response_mask)


def compute_klcov_loss(
    log_probs, old_log_probs, advantages, response_mask,
    epsilon_clip=0.2, beta_cov=0.1, top_k_fraction=0.02,
) -> Tuple[torch.Tensor, dict]:
    """
    Full KL-Cov loss (Eq. 8-9 from proposal).
    
    Normal tokens (98%): clipped GRPO surrogate.
    High-cov tokens (2%): unclipped ratio*advantage + beta*|log_pi - log_pi_old|.
    """
    B, S = log_probs.shape
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    adv_per_token = advantages.unsqueeze(1).expand(B, S)

    # Step 1: token covariances
    covariances = compute_token_covariances(log_probs, advantages, response_mask)
    
    # Step 2: select high-cov tokens
    high_cov_mask = select_high_covariance_tokens(covariances, response_mask, top_k_fraction)
    normal_mask = response_mask * (1.0 - high_cov_mask)

    # Step 3: normal tokens - clipped loss
    surr1 = ratio * adv_per_token
    surr2 = torch.clamp(ratio, 1.0 - epsilon_clip, 1.0 + epsilon_clip) * adv_per_token
    clipped_loss = -torch.min(surr1, surr2)

    # Step 4: high-cov tokens - unclipped + KL penalty
    unclipped_loss = -ratio * adv_per_token
    kl_penalty = beta_cov * log_ratio.abs()
    penalized_loss = unclipped_loss + kl_penalty

    # Step 5: combine
    per_token_loss = clipped_loss * normal_mask + penalized_loss * high_cov_mask
    total_tokens = response_mask.sum().clamp(min=1)
    loss = per_token_loss.sum() / total_tokens

    n_high = high_cov_mask.sum().item()
    n_total = response_mask.sum().item()
    info = {
        "klcov/n_high_cov_tokens": n_high,
        "klcov/n_total_tokens": n_total,
        "klcov/high_cov_fraction": n_high / max(n_total, 1),
        "klcov/mean_covariance": (covariances * response_mask).sum().item() / max(n_total, 1),
        "klcov/max_covariance": (covariances * response_mask).max().item() if n_total > 0 else 0,
        "klcov/mean_kl_penalty": (kl_penalty * high_cov_mask).sum().item() / max(n_high, 1),
    }
    return loss, info
