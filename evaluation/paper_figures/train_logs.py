"""Read training-time metrics (completion length, entropy, ...) from the
`trainer_state.json` files the HF Trainer writes into every run directory.

A method's training run is found under `train_dir` as a directory named
`<method>` or `<anything>_<method>`, e.g. outputs/qwen2.5-1.5b_multi_reward_klcov
for method `multi_reward_klcov`.
"""

import json
from pathlib import Path


def find_run_dir(train_dir: Path, method: str):
    train_dir = Path(train_dir)
    if not train_dir.is_dir():
        return None
    matches = [d for d in train_dir.iterdir() if d.is_dir() and (d.name == method or d.name.endswith("_" + method))]
    return min(matches, key=lambda d: len(d.name)) if matches else None


def _latest_state(run_dir: Path):
    final = run_dir / "trainer_state.json"
    if final.exists():
        return final
    ckpts = sorted(run_dir.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[-1]))
    return ckpts[-1] if ckpts else None


def load_metric(train_dir: Path, method: str, key: str, max_step: int = None, stride: int = 1):
    """-> [(step, value), ...] for a logged metric such as 'completions/mean_length' or 'entropy'."""
    run_dir = find_run_dir(train_dir, method)
    state = _latest_state(run_dir) if run_dir else None
    if state is None:
        return []
    out = {}
    for entry in json.loads(state.read_text()).get("log_history", []):
        step = entry.get("step")
        if key in entry and step is not None and (max_step is None or step <= max_step) and step % stride == 0:
            out[step] = float(entry[key])
    return sorted(out.items())
