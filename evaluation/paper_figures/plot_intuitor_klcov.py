"""Plot the INTUITOR + KL-Cov ablation (accuracy + completion length).

Reads:
  - JSONs under  eval_results_final/intuitor_klcov_ablation/{method}_ckptXXX/   (accuracy)
  - CSV         eval_results_final/intuitor_klcov_ablation/intuitor_and_intuitor-klcov(0.02).csv  (completion length)

Usage:
  # 2x2 paper figure (3 accuracy + 1 completion length) -> paper/ablation/intuitor_klcov_plot.pdf
  python plot_trajectory.py --paper-panel

  # 1x3 accuracy panel only:
  python plot_trajectory.py --panel

  # Completion-length only:
  python plot_trajectory.py --completion
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# Method colors/linestyles come from the shared palette so every figure in the
# paper maps a method to the same appearance (see evaluation/paper_figures/palette.py).
import sys as _sys
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from palette import label_color as _label_color, line_kw as _line_kw



DEFAULT_BASE = Path(__file__).resolve().parent
MAX_STEP_DEFAULT = 280

# Method keys must match the trajectory ckpt directory names.
# Colors match the canonical paper style for Intuitor lines.
STYLE_BY_METHOD = {k: _label_color(k) for k in (
    "intuitor",
    "intuitor_klcov",
)}

ORDER = [
    "intuitor",
    "intuitor_klcov",
]

# Map run names in the wandb-export CSV to the same method keys.
# Note: completion-length column is `mean_terminated_length` here (not `mean_length`).
CSV_RUN_TO_METHOD = {
    "run2_intuitor-qwen25-15b (betakl-0.005)": "intuitor",
    "intuitor-klcov(0.02)-qwen25-1.5b":        "intuitor_klcov",
}
COMPLETION_COLUMN = "train/completions/mean_terminated_length"


def collect(base: Path, task: str, max_step: int):
    """Return {method: [(step, acc), ...]} sorted by step, capped at max_step."""
    out = defaultdict(list)
    pat = re.compile(r"^(?P<method>.+)_ckpt(?P<step>\d+)$")
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if not m:
            continue
        method, step = m["method"], int(m["step"])
        if method not in STYLE_BY_METHOD or step > max_step:
            continue
        f = d / f"{d.name}__{task}.json"
        if not f.exists():
            continue
        payload = json.loads(f.read_text())
        acc = payload.get("accuracy")
        if acc is None:
            acc = payload.get("pass@1")
        if acc is None:
            continue
        out[method].append((step, acc))
    for method in out:
        out[method].sort(key=lambda t: t[0])
    return out


def collect_completion_lengths(csv_path: Path, max_step: int, base: Path):
    """Return {method: [(global_step, mean_length), ...]} sorted, capped at max_step.

    Same rescaling logic as klcov_ablation: wandb's `_step` is its internal
    log-call counter, not the training global_step. We rescale per-method by
    deriving each method's max_global_step from the eval ckpt dirs:
        global_step = _step * (max_ckpt_step / max_csv_step)
    """
    out = defaultdict(list)
    if not csv_path.exists():
        return out

    pat = re.compile(r"^(?P<method>.+)_ckpt(?P<step>\d+)$")
    max_ckpt = defaultdict(int)
    for d in base.iterdir():
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if m and m.group("method") in STYLE_BY_METHOD:
            max_ckpt[m.group("method")] = max(max_ckpt[m.group("method")], int(m.group("step")))

    raw = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            for run, method in CSV_RUN_TO_METHOD.items():
                len_key  = f"{run} - {COMPLETION_COLUMN}"
                step_key = f"{run} - _step"
                if len_key not in row or step_key not in row:
                    continue
                len_str  = row[len_key].strip()
                step_str = row[step_key].strip()
                if not len_str or not step_str:
                    continue
                try:
                    step = int(float(step_str))
                    val  = float(len_str)
                except ValueError:
                    continue
                raw[method].append((step, val))

    for method, pts in raw.items():
        if not pts or method not in max_ckpt or max_ckpt[method] == 0:
            continue
        max_csv = max(s for s, _ in pts)
        scale = max_ckpt[method] / max_csv
        rescaled = []
        for s, v in pts:
            gs = s * scale
            if gs > max_step:
                continue
            rescaled.append((gs, v))
        seen = {}
        for s, v in rescaled:
            seen.setdefault(round(s, 3), v)
        out[method] = sorted(seen.items())
    return out


def _parse_exclude(exclude: str):
    return {m.strip() for m in exclude.split(",") if m.strip()}


def _plot_paper_panel(base: Path, csv_path: Path, tasks, out_path: Path,
                      model_label: str, excluded, max_step: int,
                      completion_stride: int = 20):
    """One row of accuracy panels (one per task) plus a completion-length panel.

    The paper uses two tasks -> a 1x3 figure (GSM8K / MATH500 / mean completion
    length), matching the published layout. Passing three tasks gives 1x4.
    """
    if not tasks:
        raise SystemExit("--paper-panel needs at least one task")

    n = len(tasks) + 1
    fig, axes = plt.subplots(1, n, figsize=(6.6 * n, 6.0), dpi=170)
    legend_items = {}

    panel_specs = [("acc", t, axes[i]) for i, t in enumerate(tasks)]
    panel_specs.append(("comp", None, axes[-1]))

    completion_data = collect_completion_lengths(csv_path, max_step, base)
    completion_targets = [1] + list(range(completion_stride, max_step + 1, completion_stride))

    for kind, task, ax in panel_specs:
        if kind == "acc":
            data = collect(base, task, max_step)
            for method in ORDER:
                if method in excluded:
                    continue
                pts = data.get(method, [])
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                label, color = STYLE_BY_METHOD[method]
                line, = ax.plot(xs, ys, color=color, **_line_kw(method, markersize=5), label=label)
                if label not in legend_items:
                    legend_items[label] = line
            ax.set_title(task.upper(), fontsize=17)
            ax.set_ylabel("Accuracy (pass@1, greedy)", fontsize=15)
            ax.set_ylim(0, None)
        else:
            for method in ORDER:
                if method in excluded:
                    continue
                pts = completion_data.get(method, [])
                if not pts:
                    continue
                sampled = []
                for t in completion_targets:
                    s, v = min(pts, key=lambda p: abs(p[0] - t))
                    sampled.append((t, v))
                xs = [p[0] for p in sampled]
                ys = [p[1] for p in sampled]
                label, color = STYLE_BY_METHOD[method]
                line, = ax.plot(xs, ys, color=color, **_line_kw(method, markersize=5), label=label)
                if label not in legend_items:
                    legend_items[label] = line
            ax.set_title("Mean completion length", fontsize=17)
            ax.set_ylabel("Tokens", fontsize=15)
            ax.set_ylim(0, None)

        ax.set_xlabel("Training step", fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_step)

    # No suptitle: the paper carries the description in the LaTeX caption.
    if legend_items:
        labels = list(legend_items.keys())
        handles = [legend_items[k] for k in labels]
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=True,
                   bbox_to_anchor=(0.5, -0.01), fontsize=14)

    fig.tight_layout(rect=[0, 0.05, 1, 1.0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


def _plot_panel(base: Path, tasks, out_path: Path, model_label: str, excluded, max_step: int):
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.6 * len(tasks), 6.0), dpi=170, sharex=True)
    if len(tasks) == 1:
        axes = [axes]
    legend_items = {}
    for i, task in enumerate(tasks):
        ax = axes[i]
        data = collect(base, task, max_step)
        for method in ORDER:
            if method in excluded:
                continue
            pts = data.get(method, [])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            label, color = STYLE_BY_METHOD[method]
            line, = ax.plot(xs, ys, color=color, **_line_kw(method, markersize=5), label=label)
            if label not in legend_items:
                legend_items[label] = line
        ax.set_title(task.upper(), fontsize=17)
        ax.set_xlabel("Training step", fontsize=15)
        if i == 0:
            ax.set_ylabel("Accuracy (pass@1, greedy)", fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)
        ax.set_xlim(0, max_step)
    fig.suptitle(f"INTUITOR + KL-Cov ablation - {model_label}", y=0.99, fontsize=21)
    if legend_items:
        labels = list(legend_items.keys())
        handles = [legend_items[k] for k in labels]
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=True,
                   bbox_to_anchor=(0.5, -0.01), fontsize=14)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


def _plot_completion_length(base: Path, csv_path: Path, out_path: Path,
                            model_label: str, excluded, max_step: int,
                            stride: int = 20):
    data = collect_completion_lengths(csv_path, max_step, base)
    if not data:
        raise SystemExit(f"no completion-length data found in {csv_path}")
    fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=170)
    targets = [1] + list(range(stride, max_step + 1, stride))
    for method in ORDER:
        if method in excluded:
            continue
        pts = data.get(method, [])
        if not pts:
            continue
        sampled = [min(pts, key=lambda p: abs(p[0] - t)) for t in targets]
        sampled = [(t, v) for t, (_, v) in zip(targets, sampled)]
        xs = [p[0] for p in sampled]
        ys = [p[1] for p in sampled]
        label, color = STYLE_BY_METHOD[method]
        ax.plot(xs, ys, color=color, label=label, **_line_kw(method, markersize=6))
    ax.set_xlabel("Training step", fontsize=16)
    ax.set_ylabel("Mean completion length (tokens)", fontsize=16)
    ax.set_title(f"INTUITOR + KL-Cov - mean completion length ({model_label})", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=13)
    ax.set_xlim(0, max_step)
    ax.set_ylim(0, None)
    ax.set_xticks(list(range(0, max_step + 1, 20)))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--model-label", default="Qwen2.5-1.5B")
    ap.add_argument("--panel", action="store_true",
                    help="1x3 accuracy panel")
    ap.add_argument("--paper-panel", action="store_true",
                    help="2x2 paper figure (3 accuracy + 1 completion length)")
    ap.add_argument("--paper-panel-out", default=None)
    ap.add_argument("--panel-tasks", default="gsm8k,math500,mmlu_pro")
    ap.add_argument("--completion", action="store_true",
                    help="Single completion-length plot")
    ap.add_argument("--completion-csv",
                    default="intuitor_and_intuitor-klcov(0.02).csv")
    ap.add_argument("--max-step", type=int, default=MAX_STEP_DEFAULT)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = Path(args.base).resolve()
    excluded = _parse_exclude(args.exclude)
    csv_path = (base / args.completion_csv).resolve()

    if args.paper_panel:
        tasks = [t.strip() for t in args.panel_tasks.split(",") if t.strip()]
        repo_root = base.parent.parent  # eval_results_final/intuitor_klcov_ablation -> repo
        out_path = Path(args.paper_panel_out) if args.paper_panel_out else \
            repo_root / "paper" / "figures" / "ablation" / "intuitor_klcov_plot.pdf"
        _plot_paper_panel(base, csv_path, tasks, out_path, args.model_label,
                          excluded, args.max_step)
        if out_path.suffix == ".pdf":
            _plot_paper_panel(base, csv_path, tasks, out_path.with_suffix(".png"),
                              args.model_label, excluded, args.max_step)
        return

    if args.panel:
        tasks = [t.strip() for t in args.panel_tasks.split(",") if t.strip()]
        out_path = Path(args.out) if args.out else \
            base / "plots" / "trajectory_panel_gsm8k_math500_mmlu_pro.pdf"
        _plot_panel(base, tasks, out_path, args.model_label, excluded, args.max_step)
        if out_path.suffix == ".pdf":
            _plot_panel(base, tasks, out_path.with_suffix(".png"),
                        args.model_label, excluded, args.max_step)
        return

    if args.completion:
        out_path = Path(args.out) if args.out else base / "plots" / "completion_length_trajectory.pdf"
        _plot_completion_length(base, csv_path, out_path, args.model_label, excluded, args.max_step)
        if out_path.suffix == ".pdf":
            _plot_completion_length(base, csv_path, out_path.with_suffix(".png"),
                                    args.model_label, excluded, args.max_step)
        return

    raise SystemExit("Pass --paper-panel, --panel, or --completion")


if __name__ == "__main__":
    main()
