"""GSM8K task."""

from typing import List, Optional

from datasets import load_dataset

from evaluation.tasks.base import Example, Task
from evaluation.tasks.verify import extract_boxed, math_answers_equal


class GSM8KTask(Task):
    name = "gsm8k"
    metric = "pass@1"
    max_new_tokens = 1024
    min_new_tokens = 32

    def load(self, max_samples: Optional[int] = None) -> List[Example]:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        examples = []
        for i, ex in enumerate(ds):
            # GSM8K gold answer is after "#### "
            gold = ex["answer"].split("####")[-1].strip().replace(",", "")
            examples.append(Example(
                id=f"gsm8k-{i}",
                question=ex["question"],
                gold=gold,
            ))
        return examples

    def score(self, completion: str, example: Example) -> float:
        pred = extract_boxed(completion)
        return 1.0 if math_answers_equal(pred, example.gold) else 0.0
