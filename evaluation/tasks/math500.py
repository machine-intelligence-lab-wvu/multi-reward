"""MATH-500 task (HuggingFaceH4/MATH-500)."""

from typing import List, Optional

from datasets import load_dataset

from evaluation.tasks.base import Example, Task
from evaluation.tasks.verify import extract_boxed, math_answers_equal


class MATH500Task(Task):
    name = "math500"
    metric = "pass@1"
    max_new_tokens = 3072
    min_new_tokens = 64

    def load(self, max_samples: Optional[int] = None) -> List[Example]:
        # Try multiple sources
        for repo in ["HuggingFaceH4/MATH-500", "nlile/hendrycks-MATH-benchmark"]:
            try:
                ds = load_dataset(repo, split="test")
                break
            except Exception:
                continue
        else:
            raise RuntimeError("Could not load MATH-500 from any source")

        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        examples = []
        for i, ex in enumerate(ds):
            problem = ex.get("problem") or ex.get("question") or ""
            gold = ex.get("answer")
            if not gold:
                gold = extract_boxed(ex.get("solution", "")) or ""
            examples.append(Example(
                id=f"math500-{i}",
                question=problem,
                gold=str(gold),
                metadata={"level": ex.get("level"), "subject": ex.get("subject")},
            ))
        return examples

    def score(self, completion: str, example: Example) -> float:
        pred = extract_boxed(completion)
        return 1.0 if math_answers_equal(pred, example.gold) else 0.0
