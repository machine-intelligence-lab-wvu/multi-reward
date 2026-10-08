"""
Cluster Voting Reward (EMPO-style).
R_cluster(a_g) = |{g': a_g' equiv a_g}| / G
For math: extract boxed answer, normalize, group by string equality.
Reference: Zhang et al. 2025 (EMPO, NeurIPS 2025 Spotlight)
"""
from collections import defaultdict
from typing import List, Tuple, Dict
from rewards.answer_extraction import extract_boxed_answer, normalize_answer

def cluster_by_answer_equivalence(responses, extraction_fn=extract_boxed_answer):
    G = len(responses)
    normalized = [normalize_answer(extraction_fn(r)) for r in responses]
    clusters = defaultdict(list)
    unparseable = []
    for i, ans in enumerate(normalized):
        if ans is None: unparseable.append(i)
        else: clusters[ans].append(i)
    cluster_ids = [-1] * G
    for cid, (ans, indices) in enumerate(clusters.items()):
        for i in indices: cluster_ids[i] = cid
    largest = max((len(v) for v in clusters.values()), default=0)
    info = {"n_clusters": len(clusters), "n_unparseable": len(unparseable),
            "largest_cluster_frac": largest/G if G > 0 else 0, "total_responses": G}
    return cluster_ids, info


def compute_cluster_rewards(responses, extraction_fn=extract_boxed_answer):
    G = len(responses)
    if G == 0: return [], {}
    cluster_ids, info = cluster_by_answer_equivalence(responses, extraction_fn)
    cluster_sizes = defaultdict(int)
    for cid in cluster_ids:
        if cid >= 0: cluster_sizes[cid] += 1
    # Denominator = number of PARSEABLE completions (those with \boxed{}).
    # Using total G would dilute rewards when many completions lack answers,
    # penalizing correct completions for the model's formatting failures.
    n_parseable = sum(1 for cid in cluster_ids if cid >= 0)
    denom = max(n_parseable, 1)
    rewards = []
    for cid in cluster_ids:
        rewards.append(cluster_sizes[cid] / denom if cid >= 0 else 0.0)
    return rewards, info


def compute_cluster_rewards_batch(all_responses, extraction_fn=extract_boxed_answer):
    all_rewards, all_info = [], []
    for responses in all_responses:
        rewards, info = compute_cluster_rewards(responses, extraction_fn)
        all_rewards.append(rewards); all_info.append(info)
    return all_rewards, all_info
