"""KL-Cov beta_cov sweep figure (accuracy + completion length), Qwen2.5-1.5B.

Reads eval JSONs from --base (scripts/eval.sh output, e.g. results/qwen2.5-1.5b)
and completion lengths from the training runs' trainer_state.json under --train-dir.

  python evaluation/paper_figures/plot_klcov_sweep.py --paper-panel \
      --base results/qwen2.5-1.5b --train-dir outputs --paper-panel-out figures/klcov_sweep.pdf
  # other modes: --panel (accuracy only), --completion, --task gsm8k
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# Method colors/linestyles come from the shared palette so every figure in the
# paper maps a method to the same appearance (see evaluation/paper_figures/palette.py).
import sys as _sys
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from train_logs import load_metric as _load_metric
from palette import label_color as _label_color, line_kw as _line_kw



DEFAULT_BASE = Path(__file__).resolve().parent
MAX_STEP_DEFAULT = 280

# Run names (recipe names, as written by scripts/eval.sh); older names also work.
ORDER = ["multi_reward", "multi_reward_klcov_beta0.0005", "multi_reward_klcov", "multi_reward_klcov_beta0.1",
         "multi_reward_final", "multi_reward_final_klcov0005", "multi_reward_final_klcov05", "multi_reward_final_klcov0.1"]
STYLE_BY_METHOD = {k: _label_color(k) for k in ORDER}
# This figure compares beta_cov values, so show it for the default arm too.
for _k in ("multi_reward_klcov", "multi_reward_final_klcov05"):
    STYLE_BY_METHOD[_k] = (r"Multi-reward + KL-Cov ($\beta_{cov}$=0.05)", STYLE_BY_METHOD[_k][1])


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


def collect_completion_lengths(train_dir: Path, max_step: int, base: Path = None):
    """{method: [(step, mean completion length), ...]} from the runs' trainer_state.json."""
    out = {}
    for method in ORDER:
        pts = _load_metric(train_dir, method, "completions/mean_length", max_step=max_step)
        if pts:
            out[method] = pts
    return out

def _parse_exclude(exclude: str):
    return {m.strip() for m in exclude.split(",") if m.strip()}


def _plot_single_task(base: Path, task: str, out_path: Path, model_label: str, excluded, max_step: int):
    data = collect(base, task, max_step)
    if not data:
        raise SystemExit(f"no JSONs found under {base}")

    fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=150)
    for method in ORDER:
        if method in excluded:
            continue
        pts = data.get(method, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        best = max(ys)
        best_step = xs[ys.index(best)]
        label, color = STYLE_BY_METHOD[method]
        ax.plot(xs, ys, color=color, **_line_kw(method, markersize=5), label=f"{label} (peak {best:.3f} @ step {best_step})")

    ax.set_xlabel("Training step", fontsize=16)
    ax.set_ylabel(f"{task} accuracy (pass@1, greedy)", fontsize=16)
    ax.set_title(f"KL-Cov sweep on multi-reward - {model_label} ({task})", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=13)
    ax.set_ylim(0, None)
    ax.set_xlim(0, max_step)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")


def _plot_panel(base: Path, tasks, out_path: Path, model_label: str, excluded, max_step: int):
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.6 * len(tasks), 6.0), dpi=170, sharex=True)
    if len(tasks) == 1:
        axes = [axes]

    legend_items = {}
    for i, task in enumerate(tasks):
        ax = axes[i]
        data = collect(base, task, max_step)
        if not data:
            ax.set_title(f"{task} (no data)")
            ax.axis("off")
            continue

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

    fig.suptitle(f"KL-Cov sweep on multi-reward - accuracy trajectories - {model_label}", y=0.99, fontsize=21)
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
        # Pick the closest data point to each target step (one point per stride).
        sampled = []
        for t in targets:
            s, v = min(pts, key=lambda p: abs(p[0] - t))
            sampled.append((t, v))
        xs = [p[0] for p in sampled]
        ys = [p[1] for p in sampled]
        label, color = STYLE_BY_METHOD[method]
        ax.plot(xs, ys, color=color, label=label, **_line_kw(method, markersize=6))

    ax.set_xlabel("Training step", fontsize=16)
    ax.set_ylabel("Mean completion length (tokens)", fontsize=16)
    ax.set_title(f"KL-Cov sweep on multi-reward - mean completion length ({model_label})", fontsize=18)
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


def _plot_paper_panel(base: Path, csv_path: Path, tasks, out_path: Path,
                      model_label: str, excluded, max_step: int,
                      completion_stride: int = 20):
    """One row of accuracy panels (one per task) plus a completion-length panel.

    The paper uses two tasks -> a 1x3 figure (GSM8K / MATH500 / mean completion
    length), matching the published layout. Passing three tasks gives 1x4.
    Shared legend at the bottom.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE),
                    help="Trajectory root directory containing method_ckptXXX folders")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model-label", default="Qwen2.5-1.5B")
    ap.add_argument("--panel", action="store_true",
                    help="Create one figure with multiple tasks in a row and one shared legend")
    ap.add_argument("--panel-tasks", default="gsm8k,math500,mmlu_pro",
                    help="Comma-separated tasks for --panel mode")
    ap.add_argument("--completion", action="store_true",
                    help="Plot mean completion length vs training step (from trainer_state.json)")
    ap.add_argument("--train-dir", default="outputs",
                    help="directory holding the training run folders (for completion length)")
    ap.add_argument("--max-step", type=int, default=MAX_STEP_DEFAULT,
                    help="Cap the x-axis at this training step (default 280)")
    ap.add_argument("--exclude", default="",
                    help="Comma-separated method keys to skip")
    ap.add_argument("--paper-panel", action="store_true",
                    help="Render the 2x2 paper figure (3 accuracy + 1 completion length) "
                         "into figures/klcov_plot.pdf")
    ap.add_argument("--paper-panel-out", default=None,
                    help="Override output path for --paper-panel")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    excluded = _parse_exclude(args.exclude)

    if args.paper_panel:
        tasks = [t.strip() for t in args.panel_tasks.split(",") if t.strip()]
        csv_path = Path(args.train_dir).resolve()
        out_path = Path(args.paper_panel_out) if args.paper_panel_out else \
            Path("figures") / "klcov_plot.pdf"
        _plot_paper_panel(base, csv_path, tasks, out_path, args.model_label,
                          excluded, args.max_step)
        if out_path.suffix == ".pdf":
            _plot_paper_panel(base, csv_path, tasks, out_path.with_suffix(".png"),
                              args.model_label, excluded, args.max_step)
        return

    if args.completion:
        csv_path = Path(args.train_dir).resolve()
        out_path = Path(args.out) if args.out else base / "plots" / "completion_length_trajectory.pdf"
        _plot_completion_length(base, csv_path, out_path, args.model_label, excluded, args.max_step)
        # Also write a PNG copy for quick viewing.
        if out_path.suffix == ".pdf":
            _plot_completion_length(base, csv_path, out_path.with_suffix(".png"),
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

    out_path = Path(args.out) if args.out else base / "plots" / f"trajectory_{args.task}.png"
    _plot_single_task(base, args.task, out_path, args.model_label, excluded, args.max_step)


if __name__ == "__main__":
    main()
