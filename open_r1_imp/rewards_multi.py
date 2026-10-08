"""
Multi-reward functions for Open-R1 integration.

Provides three reward functions compatible with Open-R1's registry signature:
  func(completions, **kwargs) -> list[float]

1. cluster_voting_reward  — string-based cluster voting (fast, default)
2. empo_semantic_cluster_reward — DeBERTa NLI-based semantic clustering (EMPO)
3. boxed_format_reward — checks for valid \\boxed{} answer format

All three support thinking mode: when completions contain <think>...</think>,
rewards are computed on the ANSWER portion only (text after </think>).
"""

import math
import re
import logging
from typing import List, Dict, Optional, Callable, Tuple
from collections import defaultdict, Counter
from functools import partial, update_wrapper

import torch

# Import from existing reward modules (same repo)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train_4090_imp"))
from rewards.answer_extraction import extract_boxed_answer, normalize_answer, answers_equivalent
from rewards.cluster_voting import compute_cluster_rewards, cluster_by_answer_equivalence
from rewards.format_reward import compute_format_reward
from open_r1_imp.utils import split_thinking_answer

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

def _extract_completion_text(completion) -> str:
    """Extract text from Open-R1's completion format.

    Open-R1 passes completions as list[list[dict[str,str]]]:
      [[{"role": "assistant", "content": "..."}], ...]
    """
    if isinstance(completion, list):
        if len(completion) > 0 and isinstance(completion[0], dict):
            return completion[0].get("content", "")
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def _get_answer_text(text: str, thinking: bool = False) -> str:
    """Get the answer portion of a completion.

    If thinking=True, extracts text after </think>.
    Otherwise returns full text.
    """
    if thinking:
        _, answer = split_thinking_answer(text)
        return answer
    return text


# ════════════════════════════════════════════════════════════
# 1. CLUSTER VOTING REWARD (String-based)
# ════════════════════════════════════════════════════════════

_TRIVIA_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _trivia_normalize_answer(ans):
    """SQuAD-style normalization for short free-text answers (T2 TriviaQA):
    lowercase, strip punctuation and articles, collapse whitespace. Mirrors
    evaluation/tasks/triviaqa.py so training-time clusters and eval-time EM
    use the same equivalence. Returns None for empty results (-> reward 0)."""
    if ans is None:
        return None
    ans = re.sub(r"[^\w\s]", " ", ans.lower())
    ans = _TRIVIA_ARTICLES.sub(" ", ans)
    ans = " ".join(ans.split())
    return ans or None


def _trivia_extract(text):
    return _trivia_normalize_answer(extract_boxed_answer(text))


def cluster_voting_reward(completions: list, thinking: bool = False,
                          num_generations: int = None,
                          answer_normalizer: str = None,
                          **kwargs) -> list[float]:
    """Cluster voting reward based on \\boxed{} answer equivalence.

    Groups completions by extracted answer, rewards each completion
    proportional to its cluster size: R_i = |C_j| / n_parseable.

    Completions without \\boxed{} get reward 0.

    Args:
        completions: Open-R1 format [[{"role": "assistant", "content": "..."}], ...]
        thinking: if True, extract answer portion after </think> only
        num_generations: GRPO group size (must be bound via partial in the reward
            registry — TRL does not pass it through kwargs). If None, raises.
        **kwargs: ignored (TRL dataset columns)

    Returns:
        list[float] of rewards, one per completion
    """
    if num_generations is None:
        raise ValueError(
            "cluster_voting_reward requires num_generations to be bound at "
            "registry time (see get_multi_reward_funcs). Without it, the "
            "function degenerates to a boxed-format check."
        )
    if answer_normalizer not in (None, "trivia"):
        raise ValueError(f"Unknown answer_normalizer: {answer_normalizer!r} "
                         "(expected None or 'trivia')")
    texts = [_extract_completion_text(c) for c in completions]
    answers = [_get_answer_text(t, thinking) for t in texts]
    N = len(answers)

    rewards = [0.0] * N
    for start in range(0, N, num_generations):
        end = min(start + num_generations, N)
        group_answers = answers[start:end]
        if answer_normalizer == "trivia":
            group_rewards, _ = compute_cluster_rewards(
                group_answers, extraction_fn=_trivia_extract)
        else:
            group_rewards, _ = compute_cluster_rewards(group_answers)
        for i, r in enumerate(group_rewards):
            rewards[start + i] = r

    return rewards


# ════════════════════════════════════════════════════════════
# 1b. CO-REWARDING REWARD (cross-view consensus, Zhang et al. 2025)
# ════════════════════════════════════════════════════════════

def _co_rewarding_majority_vote(ans_list, empty_value=""):
    """Verbatim port of majority_vote() from the official Co-rewarding repo
    (Co-rewarding-I/verl/workers/reward_manager/co_reward.py)."""
    ans_list = [a for a in ans_list if a is not None and str(a).strip() != ""]
    if not ans_list:
        return empty_value
    return Counter(ans_list).most_common(1)[0][0]


def _co_rewarding_extract(text: str) -> str:
    """Last-\\boxed{} extraction matching the official repo's
    last_boxed_only_string + remove_boxed (no #### fallback, no normalization)."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    depth, start = 0, idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return ""


def co_rewarding_reward(completions: list, thinking: bool = False,
                        num_generations: int = None,
                        uid: list = None, view: list = None,
                        **kwargs) -> list[float]:
    """Co-rewarding-I reward — faithful port of the official implementation.

    Reference: "Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in
    Large Language Models", arXiv:2508.00410; official code
    github.com/tmlr-group/Co-rewarding (Co-rewarding-I/verl/workers/
    reward_manager/co_reward.py). Ported logic, verbatim semantics:

      ori_uid2pseudo = {uid: majority_vote(answers of AUG view)}
      aug_uid2pseudo = {uid: majority_vote(answers of ORI view)}
      reward_i = 1.0 if ans_i == pseudo_label and pseudo_label != '' else 0.0

    i.e. each response to the ORIGINAL question is rewarded iff its last-\\boxed{}
    answer equals the majority answer that the policy gives to the REPHRASED
    version of the same question (and vice versa). Uses the authors' released
    Qwen3-32B rephrasings (train_pairs.parquet) via the paired dataset built by
    scripts/rebuttal/make_corewarding_dataset.py.

    REQUIREMENTS (enforced by the A3_corewarding config + launch.sh):
      * dataset rows carry `uid` and `view` ('ori'/'aug') columns — TRL forwards
        them here as kwargs lists aligned with `completions`;
      * paired rows are adjacent, `shuffle_dataset: false`, and training runs on a
        SINGLE training process so both views of a uid share one reward batch.
        Completions whose counterpart view is absent get reward 0.0 and are
        counted + logged as a warning (should be 0 in a correct setup).
    """
    if uid is None or view is None:
        raise ValueError(
            "co_rewarding_reward requires `uid` and `view` dataset columns "
            "(use the paired dataset from make_corewarding_dataset.py)."
        )
    texts = [_extract_completion_text(c) for c in completions]
    answers = [_co_rewarding_extract(_get_answer_text(t, thinking)) for t in texts]

    # Group answer indices by (uid, view)
    by_uv: Dict[tuple, List[int]] = defaultdict(list)
    for i, (u, v) in enumerate(zip(uid, view)):
        by_uv[(u, v)].append(i)

    # Majority pseudo-label per (uid, view)
    pseudo = {key: _co_rewarding_majority_vote([answers[i] for i in idxs])
              for key, idxs in by_uv.items()}

    rewards = [0.0] * len(answers)
    missing = 0
    for i, (u, v) in enumerate(zip(uid, view)):
        other = ("aug" if v == "ori" else "ori")
        if (u, other) not in by_uv:
            missing += 1
            continue
        label = pseudo[(u, other)]
        rewards[i] = 1.0 if (answers[i] == label and label != "") else 0.0

    if missing:
        logger.warning(
            f"co_rewarding: {missing}/{len(answers)} completions had NO counterpart "
            f"view in the reward batch — pairing is broken. Check shuffle_dataset: "
            f"false, paired dataset ordering, and single-training-process launch."
        )
    return rewards


# ════════════════════════════════════════════════════════════
# 2. EMPO SEMANTIC CLUSTER REWARD (DeBERTa-based)
# ════════════════════════════════════════════════════════════

class _UnionFind:
    """Union-Find data structure for clustering."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def clusters(self, n: int) -> Dict[int, List[int]]:
        groups = defaultdict(list)
        for i in range(n):
            groups[self.find(i)].append(i)
        return groups


def _is_numeric(s: str) -> bool:
    """Check if a string represents a numeric value."""
    try:
        float(s.replace(",", "").strip())
        return True
    except (ValueError, AttributeError):
        return False


def _empo_cluster_group(
    group_answers: List[str],
    nli_model=None,
    nli_tokenizer=None,
    nli_threshold: float = 0.5,
    entail_idx: int = 2,
    open_ended: bool = False,
    nli_max_length: int = 256,
    nli_batch_size: int = 64,
) -> Tuple[List[float], float]:
    """Cluster a group of answers using EMPO method.

    Fast path: if all answers are numeric, use string equality.
    Slow path: use DeBERTa NLI for semantic similarity.

    open_ended=False (default, verifiable tasks): the clustering unit is the
      \\boxed{} answer; completions without one are unparseable and score 0.
    open_ended=True (non-verifiable tasks, e.g. A-5 on Alpaca): there is no
      boxed answer to extract, so the clustering unit is the completion TEXT and
      merging must be semantic. Without this, every completion is unparseable,
      the reward is identically 0 for the whole group, and the GRPO advantage
      (r - mean)/std is 0 -> no gradient signal at all (measured).

    Returns:
        (rewards, entropy) — per-answer rewards and group semantic entropy
    """
    G = len(group_answers)
    if G == 0:
        return [], 0.0

    # Extract and normalize answers
    extracted = []
    if open_ended:
        # Whole completion is the meaning unit. Everything is "parseable"; empty
        # generations remain None so they still land in the empty bucket.
        for ans in group_answers:
            t = (ans or "").strip()
            extracted.append(t if t else None)
    else:
        for ans in group_answers:
            boxed = extract_boxed_answer(ans)
            if boxed is not None:
                extracted.append(normalize_answer(boxed))
            else:
                extracted.append(None)

    # Count parseable
    parseable_indices = [i for i, e in enumerate(extracted) if e is not None]
    n_parseable = len(parseable_indices)

    if n_parseable == 0:
        return [0.0] * G, 0.0

    # Fast path: all parseable answers are numeric → string clustering
    parseable_answers = [extracted[i] for i in parseable_indices]
    all_numeric = all(_is_numeric(a) for a in parseable_answers if a is not None)

    if all_numeric or nli_model is None:
        # String equality clustering
        cluster_map = defaultdict(list)
        for i in parseable_indices:
            cluster_map[extracted[i]].append(i)

        # EMPO reward = meaning-cluster probability = cluster_size / G (total generations,
        # so empty/unparseable answers dilute the denominator — matches the official repo,
        # which divides counts by n_generations, NOT by the number of parseable answers).
        rewards = [0.0] * G
        for cluster_members in cluster_map.values():
            r = len(cluster_members) / G
            for i in cluster_members:
                rewards[i] = r

        # Semantic entropy over the meaning distribution (incl. the empty-answer bucket)
        probs = [len(v) / G for v in cluster_map.values()]
        n_empty = G - n_parseable
        if n_empty > 0:
            probs.append(n_empty / G)
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        return rewards, entropy

    # Slow path: DeBERTa NLI semantic clustering.
    # BATCHED. The previous version ran one nli_model() call per pair, which at
    # G=7 is 21 pairs/group x 126 groups = 2646 sequential DeBERTa-large forwards
    # per training step: measured 92 ms/pair on CPU = ~4.1 min/step, i.e. +146%
    # on a 2.8 min step. Batched it is ~19 ms/pair = ~0.8 min/step.
    uf = _UnionFind(G)

    todo = []  # (i, j) pairs that need an NLI verdict
    for i_idx, i in enumerate(parseable_indices):
        for j_idx in range(i_idx + 1, len(parseable_indices)):
            j = parseable_indices[j_idx]
            if extracted[i] == extracted[j]:
                uf.union(i, j)
            else:
                todo.append((i, j))

    if todo:
        # Bidirectional entailment (a=>b AND b=>a), which is what the standard
        # semantic-equivalence test does and what this function's docstring
        # always claimed; the previous code only checked one direction.
        pairs = [(extracted[i], extracted[j]) for i, j in todo] + \
                [(extracted[j], extracted[i]) for i, j in todo]
        scores = []
        try:
            for s in range(0, len(pairs), nli_batch_size):
                chunk = pairs[s:s + nli_batch_size]
                enc = nli_tokenizer([c[0] for c in chunk], [c[1] for c in chunk],
                                    return_tensors="pt", truncation=True,
                                    max_length=nli_max_length, padding=True)
                with torch.no_grad():
                    logits = nli_model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
                scores.extend(probs[:, entail_idx].tolist())
            n = len(todo)
            for k, (i, j) in enumerate(todo):
                if scores[k] > nli_threshold and scores[k + n] > nli_threshold:
                    uf.union(i, j)
        except Exception as e:
            # Do NOT silently degrade: string equality was already applied above,
            # so swallowing this would leave every paraphrase in its own cluster
            # and look like a legitimate all-disagree group.
            logger.warning("EMPO NLI batch failed (%s: %s); group falls back to "
                           "string-equality clustering only.", type(e).__name__, e)

    # Compute cluster rewards (= cluster_size / G, matching the official EMPO repo)
    clusters = uf.clusters(G)
    rewards = [0.0] * G
    for cluster_members in clusters.values():
        # Only count parseable members
        parseable_in_cluster = [m for m in cluster_members if m in parseable_indices]
        if not parseable_in_cluster:
            continue
        r = len(parseable_in_cluster) / G
        for i in parseable_in_cluster:
            rewards[i] = r

    # Semantic entropy over the meaning distribution (incl. empty-answer bucket)
    cluster_sizes = [len([m for m in members if m in parseable_indices])
                     for members in clusters.values()]
    cluster_sizes = [s for s in cluster_sizes if s > 0]
    probs = [s / G for s in cluster_sizes]
    n_empty = G - n_parseable
    if n_empty > 0:
        probs.append(n_empty / G)
    entropy = -sum(p * math.log(p) for p in probs if p > 0)

    return rewards, entropy


def get_empo_cluster_reward(
    thinking: bool = False,
    num_generations: int = None,
    deberta_model_name: str = "microsoft/deberta-large-mnli",
    use_semantic: bool = True,
    nli_threshold: float = 0.5,
    entropy_filter_low: float = 0.0,
    entropy_filter_high: float = float("inf"),
    open_ended: bool = False,
    nli_threads: int = 32,
) -> Callable:
    """Factory for EMPO semantic cluster reward function.

    Loads DeBERTa model on CPU at initialization time (once).
    For math problems with numeric answers, falls back to fast string clustering.

    Args:
        thinking: extract answer after </think> for thinking-mode models
        deberta_model_name: HuggingFace model ID for NLI
        use_semantic: if False, always use string clustering (faster)
        nli_threshold: entailment probability threshold for merging
        entropy_filter_low: skip groups with entropy below this (EMPO δ_low)
        entropy_filter_high: skip groups with entropy above this (EMPO δ_high)

    Returns:
        Callable compatible with Open-R1's reward registry
    """
    nli_model = None
    nli_tokenizer = None
    entail_idx = 2

    if use_semantic:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer as NLITokenizer
            logger.info(f"Loading NLI model: {deberta_model_name} (on CPU)")
            nli_model = AutoModelForSequenceClassification.from_pretrained(
                deberta_model_name).eval()
            nli_tokenizer = NLITokenizer.from_pretrained(deberta_model_name)
            torch.set_num_threads(nli_threads)

            # Resolve the entailment logit index from the checkpoint instead of
            # hardcoding 2. Label order is NOT consistent across NLI models:
            #   microsoft/deberta-large-mnli        -> [CONTRADICTION, NEUTRAL, ENTAILMENT]  (2)
            #   MoritzLaurer/DeBERTa-v3-large-mnli* -> [entailment, neutral, contradiction]  (0)
            # Hardcoding 2 with the latter would read CONTRADICTION as entailment
            # and cluster opposite answers together.
            id2label = getattr(nli_model.config, "id2label", None) or {}
            hits = [int(k) for k, v in id2label.items()
                    if str(v).upper().startswith("ENTAIL")]
            if hits:
                entail_idx = hits[0]
            else:
                # e.g. microsoft/deberta-v3-large: a plain encoder with no NLI
                # head (num_labels=2, id2label=None). AutoModelForSequenceClassification
                # gives it a RANDOM head, so its "entailment" score is noise.
                raise ValueError(
                    f"{deberta_model_name} has no entailment label in id2label "
                    f"({id2label!r}) — it is not an NLI checkpoint. Use e.g. "
                    f"microsoft/deberta-large-mnli.")
            logger.info(f"NLI model loaded; entailment index={entail_idx}, "
                        f"labels={id2label}, threads={nli_threads}")
        except Exception as e:
            logger.warning(f"Failed to load NLI model: {e}. Falling back to string clustering.")
            nli_model = None
            nli_tokenizer = None

    if open_ended and nli_model is None:
        # Worth shouting about: open-ended clustering by string equality merges
        # nothing, so every completion is its own cluster, group entropy hits
        # log(G), EMPO's own filter zeroes the group, and the run trains on nothing.
        logger.error(
            "empo_cluster: open_ended=True but no NLI model is loaded. String "
            "equality cannot merge paraphrases, so rewards will be degenerate. "
            "Set use_semantic_clustering=true with an MNLI checkpoint.")

    if num_generations is None:
        raise ValueError(
            "get_empo_cluster_reward requires num_generations to be specified "
            "at registry time. Without it, the reward degenerates to 1.0 per "
            "parseable completion (no actual clustering)."
        )

    def empo_reward(completions: list, **kwargs) -> list[float]:
        texts = [_extract_completion_text(c) for c in completions]
        answers = [_get_answer_text(t, thinking) for t in texts]
        N = len(answers)

        rewards = [0.0] * N
        for start in range(0, N, num_generations):
            end = min(start + num_generations, N)
            group_answers = answers[start:end]

            group_rewards, entropy = _empo_cluster_group(
                group_answers, nli_model, nli_tokenizer, nli_threshold,
                entail_idx=entail_idx, open_ended=open_ended)

            G = len(group_answers)
            # EMPO's intrinsic entropy filter (official repo: `if total_entropy <
            # math.log(n_generations)`): only train on groups whose semantic entropy is
            # BELOW the uniform maximum log(G); high-entropy (all-disagree) groups are zeroed.
            if G > 1 and entropy >= math.log(G):
                continue
            # Optional additional user-specified band (defaults to a no-op).
            if entropy < entropy_filter_low or entropy > entropy_filter_high:
                continue

            for i, r in enumerate(group_rewards):
                rewards[start + i] = r

        return rewards

    empo_reward.__name__ = "empo_semantic_cluster_reward"
    return empo_reward


# ════════════════════════════════════════════════════════════
# 2b. SOFT EMBEDDING-CONSENSUS REWARD (open-ended / A-5 v2)
# ════════════════════════════════════════════════════════════

def get_soft_consensus_reward(
    thinking: bool = False,
    num_generations: int = None,
    embed_model_name: str = "BAAI/bge-base-en-v1.5",
    embed_max_length: int = 512,
    embed_batch_size: int = 32,
    embed_threads: int = 32,
) -> Callable:
    """Soft embedding-consensus reward for open-ended prompts.

    reward_i = mean_{j != i} cos(e_i, e_j) over the GRPO group, where e_k is
    the L2-normalized [CLS] embedding (BGE convention) of completion k.

    This is the soft-kernel generalization of cluster voting: exact-match
    voting is a hard kernel on extracted answers; this is a continuous kernel
    on whole completions. It replaces hard NLI clustering on free-form text,
    where bidirectional entailment does not fire on long completions
    (docs/A5_FINDINGS.md §2). Because rewards are continuous, group std is
    nonzero in essentially every group — no threshold, no union-find chaining,
    no entropy gate, no zero-gradient cliff at either extreme.

    Known hack vector: mode collapse (all rollouts converge to one generic
    answer drives all rewards -> 1.0 -> std -> 0). That is answer-space drift,
    the failure mode the self-certainty channel + GDPO z-scoring + KL-Cov are
    designed to counter; watch rewards/soft_consensus_reward/mean -> 1.0 as
    the collapse signature.

    The encoder is lazy-loaded on first call (CPU), so building the reward
    registry does not pull the model for runs that never use this reward.
    Empty completions and singleton groups keep reward 0.
    """
    if num_generations is None:
        raise ValueError(
            "get_soft_consensus_reward requires num_generations at registry "
            "time; without it, completions cannot be grouped by prompt.")

    state = {"model": None, "tok": None, "failed": False}

    def _ensure_model() -> bool:
        if state["model"] is not None or state["failed"]:
            return state["model"] is not None
        try:
            from transformers import AutoModel, AutoTokenizer as EmbTokenizer
            logger.info(f"Loading consensus embedder: {embed_model_name} (on CPU)")
            state["tok"] = EmbTokenizer.from_pretrained(embed_model_name)
            state["model"] = AutoModel.from_pretrained(embed_model_name).eval()
            torch.set_num_threads(embed_threads)
        except Exception as e:
            # Do NOT silently train on a dead channel: fail loudly once.
            state["failed"] = True
            logger.error(
                "soft_consensus: failed to load embedder %s (%s: %s). "
                "Rewards will be identically 0 -> this channel has NO gradient.",
                embed_model_name, type(e).__name__, e)
        return state["model"] is not None

    def _embed(texts: List[str]) -> torch.Tensor:
        chunks = []
        with torch.no_grad():
            for s in range(0, len(texts), embed_batch_size):
                chunk = texts[s:s + embed_batch_size]
                enc = state["tok"](chunk, return_tensors="pt", truncation=True,
                                   max_length=embed_max_length, padding=True)
                out = state["model"](**enc).last_hidden_state[:, 0]  # [CLS]
                chunks.append(torch.nn.functional.normalize(out, dim=-1))
        return torch.cat(chunks, dim=0)

    def soft_consensus_reward(completions: list, **kwargs) -> list[float]:
        texts = [_extract_completion_text(c) for c in completions]
        answers = [_get_answer_text(t, thinking) for t in texts]
        N = len(answers)
        rewards = [0.0] * N
        if not _ensure_model():
            return rewards

        stripped = [(a or "").strip() for a in answers]
        nonempty = [i for i in range(N) if stripped[i]]
        if not nonempty:
            return rewards

        emb_all = _embed([stripped[i] for i in nonempty])
        row_of = {i: r for r, i in enumerate(nonempty)}

        for start in range(0, N, num_generations):
            end = min(start + num_generations, N)
            idx = [i for i in range(start, end) if i in row_of]
            G = len(idx)
            if G < 2:
                continue  # consensus undefined for empty/singleton groups
            E = emb_all[[row_of[i] for i in idx]]
            sim = E @ E.T
            # mean similarity to the OTHER members (self-sim = 1 on diagonal)
            mean_others = (sim.sum(dim=1) - 1.0) / (G - 1)
            for k, i in enumerate(idx):
                rewards[i] = float(mean_others[k])
        return rewards

    return soft_consensus_reward


# ════════════════════════════════════════════════════════════
# 3. FORMAT REWARD
# ════════════════════════════════════════════════════════════

def boxed_format_reward(completions: list, thinking: bool = False,
                        **kwargs) -> list[float]:
    """Check for valid \\boxed{} format in completion.

    Returns 1.0 if valid \\boxed{...} found, 0.0 otherwise.
    In thinking mode, only checks the answer portion (after </think>).

    Args:
        completions: Open-R1 format
        thinking: if True, check answer portion only
        **kwargs: unused

    Returns:
        list[float] of 0.0 or 1.0
    """
    rewards = []
    for comp in completions:
        text = _extract_completion_text(comp)
        answer_text = _get_answer_text(text, thinking)
        rewards.append(compute_format_reward(answer_text))
    return rewards


# ════════════════════════════════════════════════════════════
# 4. ACCURACY REWARD (Ground-truth supervised)
#    Matches the Intuitor paper's accuracy_reward implementation.
#    Uses math_verify when available, falls back to string matching.
# ════════════════════════════════════════════════════════════

def _accuracy_reward_math_verify(content: str, sol: str) -> Optional[float]:
    """Accuracy check using math_verify (Intuitor paper's exact method).

    Returns 1.0, 0.0, or None (skip this example).
    """
    try:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
    except ImportError:
        return None  # signal caller to use fallback

    gold_parsed = parse(sol, extraction_mode="first_match")
    if len(gold_parsed) == 0:
        return None  # unparseable gold → skip

    answer_parsed = parse(
        content,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )
    try:
        return float(verify(gold_parsed, answer_parsed))
    except Exception as e:
        logger.debug(f"math_verify failed: {e}")
        return None


# Check once at import time whether math_verify is available
_MATH_VERIFY_AVAILABLE = False
try:
    from math_verify import parse, verify  # noqa: F401
    from latex2sympy2_extended import NormalizationConfig  # noqa: F401
    _MATH_VERIFY_AVAILABLE = True
    logger.info("math_verify available — using Intuitor paper's exact accuracy reward")
except ImportError:
    logger.info("math_verify not installed — using string-matching fallback for accuracy reward")


def _extract_boxed(text: str) -> Optional[str]:
    """Extract content from \\boxed{...}, handling nested braces."""
    import re as _re
    idx = text.rfind("\\boxed{")
    if idx == -1:
        m = _re.search(r"####\s*(.+?)(?:\n|$)", text)
        return m.group(1).strip() if m else None
    depth, start = 0, idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None


def _normalize(answer: Optional[str]) -> Optional[str]:
    """Normalize a math answer for comparison."""
    import re as _re
    from fractions import Fraction as _Fraction
    if answer is None:
        return None
    ans = answer.strip().replace("$", "").replace("\\%", "").replace("%", "").replace(",", "")
    if not ans:
        return None
    fm = _re.match(r"\\frac\{([^}]+)\}\{([^}]+)\}", ans)
    if fm:
        try:
            n, d = float(fm.group(1)), float(fm.group(2))
            if d != 0:
                ans = str(n / d)
        except (ValueError, ZeroDivisionError):
            pass
    try:
        import math as _math
        v = float(ans)
        if _math.isinf(v) or _math.isnan(v):
            return f"{v:.10g}"
        return str(int(v)) if v == int(v) else f"{v:.10g}"
    except (ValueError, OverflowError):
        pass
    fm = _re.match(r"^(-?\d+)\s*/\s*(-?\d+)$", ans)
    if fm:
        try:
            v = float(_Fraction(int(fm.group(1)), int(fm.group(2))))
            return str(int(v)) if v == int(v) else f"{v:.10g}"
        except (ValueError, ZeroDivisionError):
            pass
    return _re.sub(r"\s+", " ", ans.lower().strip())


def _answers_match(a1: str, a2: str) -> bool:
    """Check if two answers are equivalent after normalization."""
    n1, n2 = _normalize(a1), _normalize(a2)
    return n1 is not None and n2 is not None and n1 == n2


def accuracy_reward(completions: list, thinking: bool = False,
                    **kwargs) -> list[Optional[float]]:
    """Accuracy reward matching the Intuitor paper's implementation.

    Uses math_verify for mathematical equivalence checking when available.
    Falls back to string-based \\boxed{} extraction + normalization otherwise.

    TRL automatically passes extra dataset columns (like 'solution') as kwargs.

    Args:
        completions: Open-R1 format [[{"role": "assistant", "content": "..."}], ...]
        thinking: if True, extract answer portion after </think> only
        **kwargs: must contain 'solution' (list of ground-truth solutions)

    Returns:
        list[float | None] — 1.0 correct, 0.0 wrong, None to skip
    """
    solutions = kwargs.get("solution", [])
    if not solutions:
        logger.warning("accuracy_reward: no 'solution' column found in dataset kwargs")
        return [0.0] * len(completions)

    rewards = []
    for comp, sol in zip(completions, solutions):
        text = _extract_completion_text(comp)
        content = _get_answer_text(text, thinking)

        # Primary path: math_verify (Intuitor paper's exact method)
        if _MATH_VERIFY_AVAILABLE:
            reward = _accuracy_reward_math_verify(content, sol)
            rewards.append(reward)
            continue

        # Fallback: string-based extraction + normalization
        pred = _extract_boxed(content)
        gt = _extract_boxed(sol) if sol else None
        if gt is None:
            gt = sol.split("####")[-1].strip() if "####" in sol else sol.strip()
        if pred is not None and gt is not None and _answers_match(pred, gt):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


# ════════════════════════════════════════════════════════════
# REGISTRY HELPERS
# ════════════════════════════════════════════════════════════

def get_multi_reward_funcs(script_args, num_generations: int) -> Dict[str, Callable]:
    """Build registry of multi-reward functions.

    Called from the training entry points. `num_generations` must be the
    GRPO group size from training_args — it is bound into cluster-based rewards
    here because TRL's reward-function call path does not pass it through.

    Args:
        script_args: MultiRewardScriptArguments with multi-reward params
        num_generations: GRPO group size (training_args.num_generations)

    Returns:
        dict mapping reward name → callable
    """
    thinking = getattr(script_args, "thinking", False)

    registry = {
        "cluster_voting": update_wrapper(
            partial(cluster_voting_reward, thinking=thinking, num_generations=num_generations,
                    answer_normalizer=getattr(script_args, "cluster_answer_normalizer", None)),
            cluster_voting_reward,
        ),
        "boxed_format": update_wrapper(
            partial(boxed_format_reward, thinking=thinking),
            boxed_format_reward,
        ),
        "accuracy": update_wrapper(
            partial(accuracy_reward, thinking=thinking),
            accuracy_reward,
        ),
        "co_rewarding": update_wrapper(
            partial(co_rewarding_reward, thinking=thinking, num_generations=num_generations),
            co_rewarding_reward,
        ),
        "empo_cluster": get_empo_cluster_reward(
            thinking=thinking,
            num_generations=num_generations,
            deberta_model_name=getattr(script_args, "deberta_model", "microsoft/deberta-large-mnli"),
            use_semantic=getattr(script_args, "use_semantic_clustering", False),
            nli_threshold=getattr(script_args, "nli_threshold", 0.5),
            open_ended=getattr(script_args, "empo_open_ended", False),
            entropy_filter_low=getattr(script_args, "entropy_filter_low", 0.0),
            entropy_filter_high=getattr(script_args, "entropy_filter_high", float("inf")),
        ),
        "soft_consensus": get_soft_consensus_reward(
            thinking=thinking,
            num_generations=num_generations,
            embed_model_name=getattr(script_args, "consensus_embed_model", "BAAI/bge-base-en-v1.5"),
            embed_max_length=getattr(script_args, "consensus_max_length", 512),
        ),
    }

    return registry
