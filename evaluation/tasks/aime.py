"""AIME 2024 and AIME 2025 tasks.

AIME answers are integers 0-999. Source: MathArena/aime_{year}.
"""

from typing import List, Optional

from datasets import load_dataset

from evaluation.tasks.base import Example, Task
from evaluation.tasks.verify import extract_integer


class _AIMEBase(Task):
    year: int = 0
    metric = "pass@1"
    max_new_tokens = 4096
    min_new_tokens = 64

    # Override in subclasses
    _hf_candidates: List[str] = []

    def load(self, max_samples: Optional[int] = None) -> List[Example]:
        ds = None
        last_err = None
        for repo in self._hf_candidates:
            try:
                ds = load_dataset(repo, split="train", trust_remote_code=True)
                break
            except Exception as e:
                try:
                    ds = load_dataset(repo, split="test", trust_remote_code=True)
                    break
                except Exception as e2:
                    last_err = (e, e2)
                    continue
        if ds is None:
            raise RuntimeError(f"Could not load {self.name}: {last_err}")

        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        examples = []
        for i, ex in enumerate(ds):
            problem = ex.get("problem") or ex.get("question") or ex.get("Problem") or ""
            gold = ex.get("answer") or ex.get("Answer") or ex.get("solution_number")
            try:
                gold_int = int(str(gold).strip())
            except (ValueError, TypeError):
                # Some datasets store answer as "042" → convert
                gold_int = int(str(gold).strip().lstrip("0") or "0")
            examples.append(Example(
                id=f"{self.name}-{i}",
                question=problem,
                gold=gold_int,
                metadata={"problem_id": ex.get("id") or ex.get("problem_id")},
            ))
        return examples

    def score(self, completion: str, example: Example) -> float:
        pred = extract_integer(completion)
        if pred is None:
            return 0.0
        return 1.0 if pred == int(example.gold) else 0.0


class AIME2024Task(_AIMEBase):
    name = "aime2024"
    year = 2024
    _hf_candidates = [
        "HuggingFaceH4/aime_2024",
        "Maxwell-Jia/AIME_2024",
        "MathArena/aime_2024",
    ]


class AIME2025Task(_AIMEBase):
    name = "aime2025"
    year = 2025
    _hf_candidates = [
        "yentinglin/aime_2025",
        "opencompass/AIME2025",
        "MathArena/aime_2025",
    ]
