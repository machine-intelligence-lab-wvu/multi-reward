"""Reward functions used in the paper.

TRL calls each reward as `func(completions, **dataset_columns) -> list[float]`.

  cluster_voting  answer-level consensus reward (Eq. 2): each completion gets
                  |{g': a_g' == a_g}| / n_parseable, 0 if it has no answer.
  boxed_format    auxiliary format reward: 1.0 for a non-empty \\boxed{...},
                  0.5 for an empty/unterminated \\boxed{, else 0.0.
  accuracy        ground-truth reward for the supervised GRPO-GT baseline
                  (math_verify, as in the INTUITOR codebase).

The self-certainty reward is computed inside the trainers (it needs logits).
"""

import logging
import math
import re
from collections import defaultdict
from fractions import Fraction
from functools import partial, update_wrapper
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# ANSWER EXTRACTION / NORMALIZATION (string-level, used by the cluster reward)
# ════════════════════════════════════════════════════════════

def extract_boxed_answer(text: str) -> Optional[str]:
    """Content of the last \\boxed{...} (nested braces handled); falls back to
    a GSM8K-style `#### answer` line. None if neither is present."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        m = re.search(r"####\s*(.+?)(?:\n|$)", text)
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


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    """Light normalization: strip $ % , ; evaluate simple numbers and fractions;
    otherwise lowercase with collapsed whitespace."""
    if answer is None:
        return None
    ans = answer.strip().replace("$", "").replace("\\%", "").replace("%", "").replace(",", "")
    if not ans:
        return None
    fm = re.match(r"\\frac\{([^}]+)\}\{([^}]+)\}", ans)
    if fm:
        try:
            n, d = float(fm.group(1)), float(fm.group(2))
            if d != 0:
                ans = str(n / d)
        except (ValueError, ZeroDivisionError):
            pass
    try:
        v = float(ans)
        if math.isinf(v) or math.isnan(v):
            return f"{v:.10g}"
        return str(int(v)) if v == int(v) else f"{v:.10g}"
    except (ValueError, OverflowError):
        pass
    fm = re.match(r"^(-?\d+)\s*/\s*(-?\d+)$", ans)
    if fm:
        try:
            v = float(Fraction(int(fm.group(1)), int(fm.group(2))))
            return str(int(v)) if v == int(v) else f"{v:.10g}"
        except (ValueError, ZeroDivisionError):
            pass
    return re.sub(r"\s+", " ", ans.lower().strip())


def _completion_text(completion) -> str:
    """TRL passes conversational completions as [{"role": "assistant", "content": ...}]."""
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


# ════════════════════════════════════════════════════════════
# CLUSTER VOTING
# ════════════════════════════════════════════════════════════

def compute_cluster_rewards(responses: List[str]) -> List[float]:
    """Cluster rewards for one prompt group (Eq. 2)."""
    normalized = [normalize_answer(extract_boxed_answer(r)) for r in responses]
    sizes = defaultdict(int)
    for ans in normalized:
        if ans is not None:
            sizes[ans] += 1
    n_parseable = max(sum(sizes.values()), 1)
    return [sizes[a] / n_parseable if a is not None else 0.0 for a in normalized]


def cluster_voting_reward(completions: list, num_generations: int = None, **kwargs) -> List[float]:
    """Cluster-voting reward. TRL hands the local batch with each prompt's
    `num_generations` completions contiguous, so groups are consecutive chunks."""
    if num_generations is None:
        raise ValueError("cluster_voting_reward needs num_generations (bound in get_reward_funcs).")
    texts = [_completion_text(c) for c in completions]
    rewards = []
    for start in range(0, len(texts), num_generations):
        rewards.extend(compute_cluster_rewards(texts[start:start + num_generations]))
    return rewards


# ════════════════════════════════════════════════════════════
# FORMAT
# ════════════════════════════════════════════════════════════

def compute_format_reward(response: str) -> float:
    a = extract_boxed_answer(response)
    if a and len(a.strip()) > 0:
        return 1.0
    if "\\boxed{" in response:
        return 0.5
    return 0.0


def boxed_format_reward(completions: list, **kwargs) -> List[float]:
    return [compute_format_reward(_completion_text(c)) for c in completions]


# ════════════════════════════════════════════════════════════
# GROUND-TRUTH ACCURACY (GRPO-GT baseline only)
# ════════════════════════════════════════════════════════════

try:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
    _MATH_VERIFY_AVAILABLE = True
except ImportError:
    _MATH_VERIFY_AVAILABLE = False


def _accuracy_math_verify(content: str, sol: str) -> Optional[float]:
    """1.0 / 0.0, or None when the gold solution cannot be parsed (TRL skips it)."""
    gold_parsed = parse(sol, extraction_mode="first_match")
    if len(gold_parsed) == 0:
        return None
    answer_parsed = parse(
        content,
        extraction_config=[LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False, malformed_operators=False, basic_latex=True,
                equations=True, boxed="all", units=True),
            boxed_match_priority=0,
            try_extract_without_anchor=False,
        )],
        extraction_mode="first_match",
    )
    try:
        return float(verify(gold_parsed, answer_parsed))
    except Exception as e:
        logger.debug(f"math_verify failed: {e}")
        return None


def accuracy_reward(completions: list, solution: List[str] = None, **kwargs) -> List[Optional[float]]:
    """Ground-truth accuracy against the dataset's `solution` column."""
    if not solution:
        raise ValueError("accuracy_reward needs a `solution` column in the dataset.")
    if not _MATH_VERIFY_AVAILABLE:
        raise ImportError("accuracy_reward needs math-verify: pip install 'math-verify[antlr4_13_2]'")
    return [_accuracy_math_verify(_completion_text(c), sol) for c, sol in zip(completions, solution)]


# ════════════════════════════════════════════════════════════
# REGISTRY
# ════════════════════════════════════════════════════════════

def get_reward_funcs(names: List[str], num_generations: int) -> List[Callable]:
    registry: Dict[str, Callable] = {
        "cluster_voting": update_wrapper(partial(cluster_voting_reward, num_generations=num_generations),
                                         cluster_voting_reward),
        "boxed_format": boxed_format_reward,
        "accuracy": accuracy_reward,
    }
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(f"Unknown reward function(s) {unknown}. Available: {list(registry)}")
    return [registry[n] for n in names]
