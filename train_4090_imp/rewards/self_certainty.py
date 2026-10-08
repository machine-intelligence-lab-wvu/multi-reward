"""
Self-Certainty Reward (Eq. 2 from proposal).

R_cert(r_g) = (1/T_g) * sum_t D_KL(U || pi_theta(. | q, r_{g,<t}))

D_KL(U || pi) = sum_v (1/|V|) * log((1/|V|) / pi(v))
              = -log|V| - (1/|V|) * sum_v log pi(v)

Higher values indicate more confident (peaked) predictions — a proxy for
coherent reasoning. This signal is computed online from pi_theta, preventing
reward hacking against a frozen reward model.

Reference: Zhao et al. 2025 (INTUITOR, arXiv:2505.19590)
"""
import math
import torch
import torch.nn.functional as F


def compute_self_certainty_from_logits(logits, response_mask):
    """
    Compute KL(U || pi_theta) per position from raw logits.

    Args:
        logits: (B, S, V) or (S, V) raw model logits.
        response_mask: (B, S) or (S,) binary mask for response tokens.
    Returns:
        scores: (B,) or scalar — mean KL(U||pi) over response positions.
    """
    single = logits.dim() == 2
    if single:
        logits, response_mask = logits.unsqueeze(0), response_mask.unsqueeze(0)
    B, S, V = logits.shape

    log_probs = F.log_softmax(logits, dim=-1)

    # KL(U || pi) = -log|V| - (1/|V|) * sum_v log pi(v)
    #             = -log|V| - mean_v(log pi(v))
    mean_log_probs = log_probs.mean(dim=-1)  # (B, S) — uniform-weighted
    kl_per_position = -math.log(V) - mean_log_probs
    kl_per_position = kl_per_position * response_mask

    lengths = response_mask.sum(dim=-1).clamp(min=1)
    scores = kl_per_position.sum(dim=-1) / lengths
    return scores.squeeze(0) if single else scores
