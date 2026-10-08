"""Task ABC for evaluation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Example:
    """One evaluation example."""
    id: str
    question: str              # user-message content (already formatted with options for MC tasks)
    gold: Any                  # ground-truth answer (type depends on task)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Task:
    """Abstract base class for an evaluation task.

    Subclasses set `name`, `metric`, `max_new_tokens` and implement
    `load()` and `score()`.

    If `use_chat_template = False`, the task builds its own raw completion
    prompt (no system prompt, no tokenizer.apply_chat_template). Used for
    benchmarks whose canonical eval is a few-shot completion, e.g. MMLU-Pro.
    """

    name: str = ""
    metric: str = "pass@1"
    max_new_tokens: int = 2048
    min_new_tokens: int = 0
    use_chat_template: bool = True      # False for raw completion prompts

    def load(self, max_samples: Optional[int] = None) -> List[Example]:
        """Load the dataset and return a list of Example objects.

        When use_chat_template=False, set example.question to the FULL raw
        prompt (including in-context exemplars) — it will be sent to vLLM
        verbatim without chat templating.
        """
        raise NotImplementedError

    def score(self, completion: str, example: Example) -> float:
        """Score a single completion against gold. Returns 0.0 or 1.0 (or a float for graded)."""
        raise NotImplementedError
