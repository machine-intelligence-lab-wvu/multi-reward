"""Sandboxed Python execution utilities for code-eval tasks (CRUXEval-O, LCB).

Runs untrusted code in a subprocess with a hard timeout. Not a true security
sandbox — adequate for Clariden compute nodes (isolated, ephemeral).
"""

import os
import signal
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecResult:
    passed: bool
    stdout: str = ""
    stderr: str = ""
    timeout: bool = False
    exception: Optional[str] = None


def run_python(
    code: str,
    stdin_text: str = "",
    timeout: float = 10.0,
    env_overrides: Optional[dict] = None,
) -> ExecResult:
    """Run a Python snippet in a fresh subprocess.

    Args:
        code: full Python source to execute
        stdin_text: text piped to the subprocess stdin
        timeout: wall-clock timeout in seconds
        env_overrides: extra env vars to set

    Returns:
        ExecResult with stdout, stderr, and timeout/exception flags.
    """
    env = os.environ.copy()
    # Keep the subprocess lightweight
    env.pop("PYTHONPATH", None)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    env["CUDA_VISIBLE_DEVICES"] = ""  # never let model code touch the GPU
    if env_overrides:
        env.update(env_overrides)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        path = f.name
    try:
        try:
            proc = subprocess.run(
                [sys.executable, path],
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                check=False,
            )
            return ExecResult(
                passed=(proc.returncode == 0),
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                timeout=False,
            )
        except subprocess.TimeoutExpired as e:
            return ExecResult(
                passed=False,
                stdout=(e.stdout.decode(errors="replace") if e.stdout else ""),
                stderr=(e.stderr.decode(errors="replace") if e.stderr else ""),
                timeout=True,
            )
        except Exception as e:
            return ExecResult(passed=False, exception=repr(e))
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def python_objects_equal(pred_literal: str, gold_literal: str) -> bool:
    """Compare two Python literal strings by evaluating both with ast.literal_eval.

    Falls back to stripped string equality if either side won't parse.
    """
    import ast
    pred_s = pred_literal.strip()
    gold_s = gold_literal.strip()
    try:
        p = ast.literal_eval(pred_s)
        g = ast.literal_eval(gold_s)
        return p == g
    except (ValueError, SyntaxError, TypeError, RecursionError, MemoryError):
        # TypeError: e.g. set literal containing unhashable dicts
        # RecursionError/MemoryError: pathological adversarial inputs
        return pred_s == gold_s
