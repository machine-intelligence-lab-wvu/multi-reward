"""Task adapters for the evaluation pipeline.

Each task file implements a Task subclass that loads the dataset,
formats prompts, and scores completions.
"""

from evaluation.tasks.base import Task, Example
from evaluation.tasks.gsm8k import GSM8KTask
from evaluation.tasks.math500 import MATH500Task
from evaluation.tasks.aime import AIME2024Task, AIME2025Task
from evaluation.tasks.mmlu_pro import MMLUProTask
from evaluation.tasks.cruxeval_o import CRUXEvalOTask
from evaluation.tasks.lcb import LCBTask
from evaluation.tasks.triviaqa import TriviaQATask

TASK_REGISTRY = {
    "gsm8k": GSM8KTask,
    "math500": MATH500Task,
    "aime2024": AIME2024Task,
    "aime2025": AIME2025Task,
    "mmlu_pro": MMLUProTask,
    "cruxeval_o": CRUXEvalOTask,
    "lcb": LCBTask,
    "triviaqa": TriviaQATask,
}


def get_task(name: str) -> Task:
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {name}. Available: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[name]()


__all__ = ["Task", "Example", "get_task", "TASK_REGISTRY"]
