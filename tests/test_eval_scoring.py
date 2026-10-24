"""Benchmark scorers (evaluation/tasks) — no datasets or GPUs needed."""

from evaluation.tasks.aime import AIME2024Task
from evaluation.tasks.base import Example
from evaluation.tasks.cruxeval_o import CRUXEvalOTask
from evaluation.tasks.gsm8k import GSM8KTask
from evaluation.tasks.lcb import LCBTask
from evaluation.tasks.math500 import MATH500Task
from evaluation.tasks.mmlu_pro import MMLUProTask
from evaluation.tasks.sandbox import run_python


def ex(gold=None, **meta):
    return Example(id="x", question="", gold=gold, metadata=meta)


def test_gsm8k():
    t = GSM8KTask()
    assert t.score(r"so \boxed{18}", ex("18")) == 1.0
    assert t.score(r"\boxed{1,000}", ex("1000")) == 1.0
    assert t.score(r"\boxed{17}", ex("18")) == 0.0
    assert t.score("the answer is 18", ex("18")) == 0.0


def test_math500():
    t = MATH500Task()
    assert t.score(r"\boxed{0.5}", ex(r"\frac{1}{2}")) == 1.0
    assert t.score(r"\boxed{(3, \frac{\pi}{2})}", ex(r"\left( 3, \frac{\pi}{2} \right)")) == 1.0
    assert t.score(r"\boxed{4}", ex(r"3\sqrt{13}")) == 0.0


def test_aime():
    assert AIME2024Task().score(r"\boxed{033}", ex("33")) == 1.0


def test_mmlu_pro():
    assert MMLUProTask().score("... The answer is (C).", ex("C", n_options=10)) == 1.0


def test_cruxeval():
    t = CRUXEvalOTask()
    assert t.score('```json\n{"reasoning": "x", "answer": "[1]"}\n```', ex("[1]")) == 1.0
    assert t.score(r"\boxed{1}", ex("[1]")) == 0.0


SOL = "```python\nclass Solution:\n    def f(self, n: int, xs: List[int]) -> Tuple[int, int]:\n        return (n, sum(xs))\n```"


def test_lcb_functional_one_json_value_per_line():
    test = {"testtype": "functional", "input": "3\n[1, 2, 4]", "output": "[3, 7]"}
    assert LCBTask().score(SOL, ex(public_tests=[test], private_tests=[], func_name="f")) == 1.0
    wrong = {"testtype": "functional", "input": "3\n[1, 2, 4]", "output": "[3, 8]"}
    assert LCBTask().score(SOL, ex(public_tests=[wrong], private_tests=[], func_name="f")) == 0.0


def test_lcb_functional_single_list_argument_not_spread():
    sol = "```python\nclass Solution:\n    def g(self, grid: List[List[int]]) -> int:\n        return len(grid)\n```"
    test = {"testtype": "functional", "input": "[[1, 2], [3, 4]]", "output": "2"}
    assert LCBTask().score(sol, ex(public_tests=[test], private_tests=[], func_name="g")) == 1.0


def test_lcb_stdin_linewise_and_numeric():
    sol = "```python\nprint('1.50  2')\nprint('ok ')\n```"
    assert LCBTask().score(sol, ex(public_tests=[{"testtype": "stdin", "input": "", "output": "1.5 2\nok"}],
                                   private_tests=[], func_name=None)) == 1.0
    assert LCBTask().score(sol, ex(public_tests=[{"testtype": "stdin", "input": "", "output": "1.5 3\nok"}],
                                   private_tests=[], func_name=None)) == 0.0


def test_sandbox_hides_credentials(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    out = run_python("import os; print([k for k in os.environ if 'TOKEN' in k or 'WANDB' in k])").stdout
    assert out.strip() == "[]"
