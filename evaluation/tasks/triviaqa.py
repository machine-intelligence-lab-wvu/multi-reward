"""TriviaQA (rc.nocontext) — exact match against gold aliases.

Eval-time-only labels for the T-2 label-free non-math probe: training sees
only the *questions* (recipes/rebuttal/T2_triviaqa_*.yaml); this task scores
EM against the gold alias set on the validation split.

Prediction = the \\boxed{} answer if present, else the last non-empty line
(the INTUITOR arm may abandon the box under collapse — that behavior must
still be scored, not zeroed on a technicality). Matching uses SQuAD-style
normalization (lowercase, strip articles/punctuation/extra whitespace)
against the dataset's `normalized_aliases` (re-normalized on our side too,
so both sides pass through the identical function).
"""

import re
import string
from typing import List, Optional

from datasets import load_dataset

from evaluation.tasks.base import Example, Task
from evaluation.tasks.verify import extract_boxed

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    """SQuAD/TriviaQA-style answer normalization."""
    text = text.lower()
    text = text.translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _extract_pred(completion: str) -> Optional[str]:
    pred = extract_boxed(completion)
    if pred:
        return pred
    lines = [l.strip() for l in completion.strip().splitlines() if l.strip()]
    return lines[-1] if lines else None


class TriviaQATask(Task):
    name = "triviaqa"
    metric = "em"
    max_new_tokens = 1024

    def load(self, max_samples: Optional[int] = None) -> List[Example]:
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext",
                          split="validation")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        examples = []
        for i, ex in enumerate(ds):
            aliases = set(ex["answer"]["normalized_aliases"])
            aliases.add(ex["answer"]["value"])
            examples.append(Example(
                id=f"triviaqa-{i}",
                question=ex["question"],
                gold=sorted(_normalize(a) for a in aliases if a),
            ))
        return examples

    def score(self, completion: str, example: Example) -> float:
        pred = _extract_pred(completion)
        if not pred:
            return 0.0
        return 1.0 if _normalize(pred) in example.gold else 0.0
