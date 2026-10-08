"""MMLU-Pro task — paper-faithful 5-shot CoT completion-style eval.

Mirrors TIGER-Lab's official evaluate_from_local.py:
  - Load validation split for 5 CoT exemplars per category
  - Plain-text completion prompt (NO chat template, NO system prompt)
  - Extract answer from "the answer is (X)" (fallback: last letter A-J)

The Intuitor paper says: "All evaluations are obtained with the chat inference
template, except for MMLU-Pro" — meaning MMLU-Pro uses the benchmark's original
(completion-style) prompt format. That's what this task implements.
"""

import re
import string
from collections import defaultdict
from typing import Dict, List, Optional

from datasets import load_dataset

from evaluation.tasks.base import Example, Task


LETTERS = string.ascii_uppercase  # A..Z (we only need A..J)


def _format_question(q: str, options: List[str]) -> str:
    lines = [f"Question: {q}", "Options:"]
    for i, opt in enumerate(options):
        lines.append(f"{LETTERS[i]}. {opt}")
    return "\n".join(lines)


def _format_fewshot_example(q: str, options: List[str], cot: str, answer: str) -> str:
    """Format one in-context exemplar (with its CoT and correct letter)."""
    body = _format_question(q, options)
    cot_clean = (cot or "").strip()
    return f"{body}\nAnswer: Let's think step by step. {cot_clean} The answer is ({answer})."


def _build_fewshot_prompts(val_ds, n_shots: int = 5) -> Dict[str, str]:
    """For each MMLU-Pro category, build a single 5-shot prefix string.

    Returns: dict {category -> prefix_text}.
    The prefix ends with a trailing newline — append the test question after.
    """
    by_cat: Dict[str, list] = defaultdict(list)
    for ex in val_ds:
        cat = ex.get("category", "unknown")
        by_cat[cat].append(ex)

    prefixes: Dict[str, str] = {}
    for cat, items in by_cat.items():
        shots = items[:n_shots]
        header = (
            f"The following are multiple choice questions (with answers) about {cat}. "
            f"Think step by step and then finish your answer with "
            f'"the answer is (X)" where X is the correct letter choice.\n\n'
        )
        body_parts = []
        for s in shots:
            body_parts.append(_format_fewshot_example(
                s["question"], s["options"], s.get("cot_content", ""), s["answer"],
            ))
        prefix = header + "\n\n".join(body_parts) + "\n\n"
        prefixes[cat] = prefix
    return prefixes


class MMLUProTask(Task):
    """MMLU-Pro with TIGER-Lab's official 5-shot CoT completion-style prompt."""

    name = "mmlu_pro"
    metric = "pass@1"
    max_new_tokens = 2048
    min_new_tokens = 1
    use_chat_template = False            # <-- raw completion, no chat template

    n_shots: int = 5

    def load(self, max_samples: Optional[int] = None) -> List[Example]:
        val_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
        test_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

        prefixes = _build_fewshot_prompts(val_ds, n_shots=self.n_shots)
        # Fallback if a test-category has no validation exemplars: use the
        # first prefix we have (rare, but MMLU-Pro covers all categories).
        fallback_prefix = next(iter(prefixes.values())) if prefixes else ""

        if max_samples:
            test_ds = test_ds.select(range(min(max_samples, len(test_ds))))

        examples = []
        for i, ex in enumerate(test_ds):
            cat = ex.get("category", "unknown")
            prefix = prefixes.get(cat, fallback_prefix)
            # Build the test question stub (ending with "Let's think step by step.")
            test_body = _format_question(ex["question"], ex["options"])
            test_stub = f"{test_body}\nAnswer: Let's think step by step."
            full_prompt = prefix + test_stub

            gold_letter = ex.get("answer") or LETTERS[ex["answer_index"]]
            examples.append(Example(
                id=f"mmlu_pro-{i}",
                question=full_prompt,   # FULL raw prompt — sent to vLLM verbatim
                gold=gold_letter,
                metadata={
                    "category": cat,
                    "n_options": len(ex["options"]),
                    "answer_index": ex.get("answer_index"),
                },
            ))
        return examples

    def score(self, completion: str, example: Example) -> float:
        pred = _extract_mmlu_pro_answer(completion, max_letter=LETTERS[example.metadata["n_options"] - 1])
        if pred is None:
            return 0.0
        return 1.0 if pred == example.gold else 0.0


def _extract_mmlu_pro_answer(text: str, max_letter: str = "J") -> Optional[str]:
    """Extract the MCQ letter. Primary pattern matches TIGER-Lab's scorer."""
    if not text:
        return None
    letters = f"A-{max_letter}"
    # TIGER-Lab's primary extraction: "the answer is (X)" / "the answer is X"
    patterns = [
        rf"[Tt]he answer is[:\s]*\(?([{letters}])\)?",
        rf"answer[:\s]*\(?([{letters}])\)?\b",
        rf"\b([{letters}])\s*(?:is|would be)\s*(?:the|correct)",
        rf"^\s*\(?([{letters}])\)?\s*$",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    # Fallback: last standalone A..max letter anywhere in the text
    matches = re.findall(rf"\b([{letters}])\b", text)
    if matches:
        return matches[-1].upper()
    return None
