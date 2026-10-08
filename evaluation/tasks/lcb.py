"""LiveCodeBench (LCB) v6 — matching the Intuitor paper / lighteval `extended|lcb:codegeneration_release_v6` task.

Dataset:  `lighteval/code_generation_lite`, subset `release_v6` (~1055 problems
released between May 2023 and April 2025).

Prompt format (from lighteval `prepare_prompt`, sent as the user message; the
training `TRAINING_SYSTEM_PROMPT` is applied as the system message via the
chat template, matching the paper's "same chat-style prompting format as in
training" sentence):

    You will be given a question (problem specification) and will generate
    a correct Python program that matches the specification and passes all
    tests.

    Question: {question_content}

    {starter or stdin instruction}

Code extraction (lighteval's `extract_code`): take the content between the
last two ``` fences in the completion. If fewer than two fences exist, score
zero (no parseable code).

Scoring: pass@1 — the model's solution must pass ALL (public + private)
tests. Each test runs in a fresh Python subprocess with a 6 s wall-clock
timeout (LCB's default).
"""

import base64
import json
import pickle
import zlib
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from evaluation.tasks.base import Example, Task
from evaluation.tasks.sandbox import run_python


def _decode_tests(raw: Any) -> List[Dict[str, str]]:
    """Decode public/private test cases. Private tests in v6 are
    base64(zlib(pickle(json)))-encoded."""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        try:
            return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode()))))
        except Exception:
            return []


def _extract_code(model_output: str) -> str:
    """Lighteval-equivalent extractor: content between the last two ``` fences.

    Mirrors `lighteval.tasks.tasks.lcb.codegen_metrics.extract_code`:
        outputlines = output.split("\\n")
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
        if len(indexlines) < 2: return ""
        return "\\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])
    """
    lines = model_output.split("\n")
    fence_idx = [i for i, line in enumerate(lines) if "```" in line]
    if len(fence_idx) < 2:
        return ""
    return "\n".join(lines[fence_idx[-2] + 1 : fence_idx[-1]])


def _build_functional_script(user_code: str, func_name: str, test_input: str, expected: str) -> str:
    """LeetCode-style: instantiate Solution() and call .{func_name}(*args)."""
    return (
        "import json, sys\n"
        "from typing import *\n"
        f"{user_code}\n\n"
        "try:\n"
        f"    _inp = json.loads({test_input!r})\n"
        "    _args = _inp if isinstance(_inp, list) else [_inp]\n"
        f"    _got = Solution().{func_name}(*_args)\n"
        f"    _expected = json.loads({expected!r})\n"
        "    print('__PASS__' if _got == _expected else '__FAIL__')\n"
        "except Exception as e:\n"
        "    print('__EXC__', type(e).__name__, str(e))\n"
    )


def _build_stdin_script(user_code: str) -> str:
    return user_code


# Lighteval's `prepare_prompt` — used verbatim as the user message.
_PROMPT_HEADER = (
    "You will be given a question (problem specification) and will generate "
    "a correct Python program that matches the specification and passes all tests.\n\n"
)
_STARTER_TAIL = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters.\n"
)
_STDIN_TAIL = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program runs, "
    "it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
    "```python\n# YOUR CODE HERE\n```\n\n"
)


class LCBTask(Task):
    name = "lcb"
    metric = "pass@1"
    # lighteval sets generation_size=32768; in practice greedy completions for
    # Qwen2.5-1.5B/3B finish far earlier on EOS. 8192 keeps wall time bounded
    # while still leaving enough room for any plausible solution.
    max_new_tokens = 8192
    min_new_tokens = 32

    release_tag = "release_v6"
    per_test_timeout: float = 6.0

    def load(self, max_samples: Optional[int] = None) -> List[Example]:
        ds = load_dataset(
            "lighteval/code_generation_lite",
            self.release_tag,
            split="test",
            trust_remote_code=True,
        )
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        examples = []
        for ex in ds:
            question = ex.get("question_content") or ex.get("question_title") or ""
            starter = ex.get("starter_code", "") or ""
            pub = _decode_tests(ex.get("public_test_cases", ""))
            prv = _decode_tests(ex.get("private_test_cases", ""))
            meta_raw = ex.get("metadata", "")
            try:
                meta = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
            except Exception:
                meta = {}

            # User message — lighteval's `prepare_prompt`, verbatim.
            query = _PROMPT_HEADER + f"Question: {question}\n\n"
            if starter.strip():
                query += _STARTER_TAIL + f"```python\n{starter}\n```\n\n"
            else:
                query += _STDIN_TAIL

            examples.append(Example(
                id=str(ex.get("question_id") or ex.get("question_title") or "lcb"),
                question=query,
                gold=None,
                metadata={
                    "public_tests": pub,
                    "private_tests": prv,
                    "starter_code": starter,
                    "func_name": meta.get("func_name"),
                    "platform": ex.get("platform"),
                    "difficulty": ex.get("difficulty"),
                },
            ))
        return examples

    def score(self, completion: str, example: Example) -> float:
        code = _extract_code(completion)
        if not code:
            return 0.0

        all_tests = (example.metadata["public_tests"] or []) + (example.metadata["private_tests"] or [])
        if not all_tests:
            return 0.0

        func_name = example.metadata.get("func_name")

        for t in all_tests:
            testtype = t.get("testtype", "stdin")
            if testtype == "functional" and func_name:
                script = _build_functional_script(code, func_name, t.get("input", ""), t.get("output", ""))
                res = run_python(script, timeout=self.per_test_timeout)
                if res.timeout or not res.passed or "__PASS__" not in (res.stdout or ""):
                    return 0.0
            else:
                script = _build_stdin_script(code)
                res = run_python(script, stdin_text=t.get("input", ""), timeout=self.per_test_timeout)
                if res.timeout or not res.passed:
                    return 0.0
                if (res.stdout or "").strip() != (t.get("output", "") or "").strip():
                    return 0.0
        return 1.0
