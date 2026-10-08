"""
GDPO Advantage: per-channel normalization within prompt groups, then combine.
Reference: Liu et al. 2026 (arXiv:2601.05242)
"""
import torch
from typing import Dict

def normalize_within_groups(rewards, group_ids, eps=1e-8):
    normalized = torch.zeros_like(rewards)
    for gid in group_ids.unique():
        mask = group_ids == gid
        gr = rewards[mask]; mu = gr.mean(); sigma = gr.std()
        normalized[mask] = (gr - mu) / (sigma + eps) if sigma > eps else 0.0
    return normalized

def compute_gdpo_advantages(reward_channels, weights, group_ids, eps=1e-8):
    N = next(iter(reward_channels.values())).shape[0]
    device = next(iter(reward_channels.values())).device
    advantages = torch.zeros(N, dtype=torch.float32, device=device)
    for name, rewards in reward_channels.items():
        w = weights.get(name, 0.0)
        if w == 0.0: continue
        advantages += w * normalize_within_groups(rewards, group_ids, eps)
    return advantages
