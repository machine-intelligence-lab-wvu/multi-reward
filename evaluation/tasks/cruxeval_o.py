"""CRUXEval-O via the ZeroEval framework — matching the Intuitor paper.

Dataset:  `flydust/zero-eval`, config `crux`, split `test` (the wrapped
copy ZeroEval uses; the `question` field is pre-formatted with the
function body and the `[PYTHON] ... assert f(...) == ??` assertion-style
instruction).

Prompt:  ZeroEval `OEQA` template (cot=True) wrapped around the dataset
question, sent as the user message under the training system prompt
(matching the paper's "same chat-style prompting format as in training").
The model is asked to emit a JSON object containing `reasoning` and
`answer` fields; the predicted output literal is read out of `answer`.

Scoring:  pass@1 — the predicted Python literal must equal the gold
literal under `python_objects_equal` (AST equality, so `'5'` vs `5`
won't false-fail).
"""

import json
import re
from typing import List, Optional

from evaluation.tasks.base import Example, Task
from evaluation.tasks.sandbox import python_objects_equal


# ZeroEval `OEQA` template (src/templates/OEQA.py), cot=True branch.
_OEQA_TEMPLATE = """
## Question:

{question}


## Instruction

Please answer this question by first reasoning and then providing your answer.
Present your reasoning and solution in the following json format.
Please show your final answer in the `answer` field, e.g.,`"answer": "42"`.

```json
{{
    "reasoning": "___",
    "answer": "___"
}}
```
"""


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_ANSWER_RE = re.compile(r'"answer"\s*:\s*(.+?)(?:,\s*"|\}\s*$|\}\s*```)', re.DOTALL)


def _extract_boxed(text: str) -> Optional[str]:
    """Extract the LAST `\\boxed{…}` content with balanced-brace matching.

    Handles nested braces like `\\boxed{{1: None, 2: None}}` (i.e. a dict
    inside a box). Returns the inner literal as a string, or None.
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start : i - 1]
    return None


def _extract_answer(completion: str) -> Optional[str]:
    """ZeroEval-style answer extraction with `\\boxed{}` fallback.

    1. Try every fenced ```json``` block, parse as JSON, return its `answer`.
    2. Try the last bare {...} that parses as JSON.
    3. Regex on `"answer": ...`.
    4. `\\boxed{…}` — picks up models that follow the training system prompt
       (`\\boxed{Your Answer}`) instead of the CRUX user-prompt's JSON form.

    Returns the predicted Python-literal string, or None.
    """
    # 1. Fenced JSON blocks (largest first — most likely the final answer).
    for match in reversed(list(_JSON_BLOCK_RE.finditer(completion))):
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict) and "answer" in obj:
                ans = obj["answer"]
                return ans if isinstance(ans, str) else json.dumps(ans)
        except json.JSONDecodeError:
            continue

    # 2. Last bare {...} block.
    last_brace = completion.rfind("}")
    while last_brace != -1:
        first_brace = completion.rfind("{", 0, last_brace)
        if first_brace == -1:
            break
        candidate = completion[first_brace : last_brace + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "answer" in obj:
                ans = obj["answer"]
                return ans if isinstance(ans, str) else json.dumps(ans)
        except json.JSONDecodeError:
            pass
        last_brace = completion.rfind("}", 0, last_brace)

    # 3. Regex on `"answer": …`.
    m = _ANSWER_RE.search(completion)
    if m:
        raw = m.group(1).strip()
        if raw.startswith('"') and raw.endswith('"'):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw[1:-1]
        return raw

    # 4. `\\boxed{…}` fallback.
    boxed = _extract_boxed(completion)
    if boxed is not None:
        return boxed.strip()

    return None


class CRUXEvalOTask(Task):
    name = "cruxeval_o"
    metric = "pass@1"
    max_new_tokens = 2048
    min_new_tokens = 16

    def load(self, max_samples: Optional[int] = None) -> List[Example]:
        from datasets import load_dataset
        ds = load_dataset("flydust/zero-eval", "crux", split="test")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        examples = []
        for ex in ds:
            user = _OEQA_TEMPLATE.replace("{question}", ex["question"])
            examples.append(Example(
                id=ex["id"],
                question=user,
                gold=ex["answer"],
                metadata={"task_type": ex.get("task_type")},
            ))
        return examples

    def score(self, completion: str, example: Example) -> float:
        pred = _extract_answer(completion)
        if pred is None:
            return 0.0
        return 1.0 if python_objects_equal(pred, example.gold) else 0.0
