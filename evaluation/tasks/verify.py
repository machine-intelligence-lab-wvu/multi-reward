"""Self-contained answer extraction and verification utilities.

No dependency on train_4090_imp. Uses math_verify when available,
falls back to string-based normalization.
"""

import math
import re
from fractions import Fraction
from typing import Optional


# Detect math_verify once at import time
_MATH_VERIFY_AVAILABLE = False
try:
    from math_verify import LatexExtractionConfig, parse, verify  # noqa: F401
    from latex2sympy2_extended import NormalizationConfig  # noqa: F401
    _MATH_VERIFY_AVAILABLE = True
except ImportError:
    pass


# ════════════════════════════════════════════════════════════
# EXTRACTION
# ════════════════════════════════════════════════════════════

def extract_boxed(text: str) -> Optional[str]:
    """Extract content from the last \\boxed{...} in text, handling nested braces.

    Fallback: look for GSM8K-style `#### 42` at end of line.
    """
    if text is None:
        return None
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


def extract_integer(text: str) -> Optional[int]:
    """Extract an integer answer (for AIME-style numeric-only problems).

    Strategy: prefer \\boxed{n}; fallback to last integer in text.
    Returns None if nothing found.
    """
    boxed = extract_boxed(text)
    candidate = boxed if boxed is not None else text
    # Remove commas and find all integers
    cleaned = candidate.replace(",", "")
    matches = re.findall(r"-?\d+", cleaned)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except (ValueError, OverflowError):
        return None


def extract_mc_letter(text: str, max_letter: str = "J") -> Optional[str]:
    """Extract a multiple-choice letter answer (A..max_letter).

    Strategy:
      1. Look inside \\boxed{...} for a letter
      2. Fallback: look for common patterns like "answer is (A)", "answer: A"
      3. Fallback: last standalone capital letter A..max_letter in the string
    """
    if text is None:
        return None
    letters = f"A-{max_letter}"
    boxed = extract_boxed(text)
    if boxed:
        m = re.search(rf"\b([{letters}])\b", boxed)
        if m:
            return m.group(1)
    # Common natural-language patterns
    patterns = [
        rf"answer\s*(?:is|:)\s*\(?([{letters}])\)?\b",
        rf"answer\s*(?:is|:)\s*[\"']?([{letters}])[\"']?",
        rf"^\s*\(?([{letters}])\)?\s*$",
        rf"\b([{letters}])\b\s*(?:is|would be)\s*(?:the|correct)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    # Fallback: last standalone capital letter
    matches = re.findall(rf"\b([{letters}])\b", text)
    if matches:
        return matches[-1]
    return None


# ════════════════════════════════════════════════════════════
# NORMALIZATION (string fallback when math_verify unavailable)
# ════════════════════════════════════════════════════════════

def _normalize_math_str(answer: Optional[str]) -> Optional[str]:
    """Normalize a math answer string for equality comparison."""
    if answer is None:
        return None
    ans = answer.strip().replace("$", "").replace("\\%", "").replace("%", "").replace(",", "")
    if not ans:
        return None
    # \frac{a}{b} → float
    fm = re.match(r"\\frac\{([^}]+)\}\{([^}]+)\}", ans)
    if fm:
        try:
            n, d = float(fm.group(1)), float(fm.group(2))
            if d != 0:
                ans = str(n / d)
        except (ValueError, ZeroDivisionError):
            pass
    # numeric
    try:
        v = float(ans)
        if math.isinf(v) or math.isnan(v):
            return f"{v:.10g}"
        return str(int(v)) if v == int(v) else f"{v:.10g}"
    except (ValueError, OverflowError):
        pass
    # a/b fraction
    fm = re.match(r"^(-?\d+)\s*/\s*(-?\d+)$", ans)
    if fm:
        try:
            v = float(Fraction(int(fm.group(1)), int(fm.group(2))))
            return str(int(v)) if v == int(v) else f"{v:.10g}"
        except (ValueError, ZeroDivisionError):
            pass
    return re.sub(r"\s+", " ", ans.lower().strip())


# ════════════════════════════════════════════════════════════
# VERIFICATION
# ════════════════════════════════════════════════════════════

def math_answers_equal(pred: Optional[str], gold: Optional[str]) -> bool:
    """Check math-answer equivalence.

    Uses math_verify when available (Intuitor paper's exact method);
    falls back to string-based normalization.
    """
    if pred is None or gold is None:
        return False

    if _MATH_VERIFY_AVAILABLE:
        try:
            _cfg = [
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False, malformed_operators=False,
                        basic_latex=True, boxed="all", units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ]
            # Wrap both sides in \boxed{} so the LatexExtractionConfig with
            # boxed_match_priority=0 can find them. Without the wrapper, the
            # default parse() config extracts only numbers/simple exprs and
            # returns "3" for "3\sqrt{13}", which then fails verify().
            gold_wrapped = gold if "\\boxed" in gold else f"\\boxed{{{gold}}}"
            pred_wrapped = pred if "\\boxed" in pred else f"\\boxed{{{pred}}}"
            gold_parsed = parse(gold_wrapped, extraction_config=_cfg, extraction_mode="first_match")
            pred_parsed = parse(pred_wrapped, extraction_config=_cfg, extraction_mode="first_match")
            if gold_parsed and pred_parsed and bool(verify(gold_parsed, pred_parsed)):
                return True
            # math_verify said no — fall through to string match as a safety net
        except Exception:
            pass

    # String fallback (also catches cases where math_verify rejected a literal-string match)
    n_pred, n_gold = _normalize_math_str(pred), _normalize_math_str(gold)
    return n_pred is not None and n_gold is not None and n_pred == n_gold
