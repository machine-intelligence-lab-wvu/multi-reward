"""Plot single-vs-multi reward ablation trajectories.

Compares three Qwen2.5-1.5B runs:
  - cluster_only         (single reward: cluster-only)
  - intuitor             (single reward: self-certainty only)
  - multi_reward_final   (multi-reward, no KL-Cov)

Reads:
  - JSONs under  eval_results_final/single_multi_reward_ablation/{method}_ckptXXX/   (accuracy)
  - CSV         cluster_self-certainty_multi-reward_mean_completion.csv              (completion length)
  - CSV         cluster_self-certainty_multi-reward_mean_entropy.csv                 (entropy)

Usage:
  # 1x3 accuracy panel (GSM8K / MATH500 / MMLU-Pro):
  python plot_trajectory.py --panel

  # Mean completion length:
  python plot_trajectory.py --completion

  # Mean entropy:
  python plot_trajectory.py --entropy

  # Single accuracy task:
  python plot_trajectory.py --task gsm8k

  # 2x3 paper panel (3 accuracy + completion + entropy + shared legend):
  python plot_trajectory.py --paper-panel
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
STRIDE_DEFAULT = 20

# Method keys here MUST match the directory names produced by the orchestrator
# (orchestrate.py expands TRAJECTORY_METHODS into "{method}_ckpt{step:03d}/").
STYLE_BY_METHOD = {k: _label_color(k) for k in (
    "cluster_only",
    "intuitor",
    "multi_reward_final",
)}

# Plotting / legend order. Missing methods are skipped automatically.
ORDER = [
    "cluster_only",
    "intuitor",
    "multi_reward_final",
]

# Map run names that appear in the wandb-export CSVs to method keys.
CSV_RUN_TO_METHOD = {
    "cluster-only-noklcov-qwen25-1.5b":         "cluster_only",
    "multi-reward-final-noklcov-qwen25-1.5b":   "multi_reward_final",
    "run2_intuitor-qwen25-15b (betakl-0.005)":  "intuitor",
}

# Per-metric style: (csv_column_suffix, y_axis_label, plot_title_fragment).
METRICS = {
    "completion": ("train/completions/mean_length", "Mean completion length (tokens)", "mean completion length"),
    "entropy":    ("train/entropy",                 "Mean entropy",                    "mean entropy"),
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def collect_accuracy(base: Path, task: str, max_step: int):
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


def collect_metric(csv_path: Path, metric_col: str, max_step: int, stride: int):
    """Return {method: [(global_step, value), ...]} sampled at multiples of `stride`.

    Both wandb CSVs in this folder include a true `train/global_step` column
    (1-indexed: row N has train/global_step == N), so no rescaling is needed.
    We filter rows where train/global_step is a multiple of `stride` and within
    [stride, max_step]. The first checkpoint (step == stride) is always included.
    """
    out = defaultdict(list)
    if not csv_path.exists():
        return out

    targets = set(range(stride, max_step + 1, stride))

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gs = int(float(row["train/global_step"]))
            except (KeyError, TypeError, ValueError):
                continue
            if gs not in targets:
                continue
            for run, method in CSV_RUN_TO_METHOD.items():
                key = f"{run} - {metric_col}"
                if key not in row:
                    continue
                val_str = (row[key] or "").strip()
                if not val_str:
                    continue
                try:
                    val = float(val_str)
                except ValueError:
                    continue
                out[method].append((gs, val))

    for method in out:
        # Deduplicate (a step should appear at most once per run anyway).
        seen = {}
        for s, v in out[method]:
            seen.setdefault(s, v)
        out[method] = sorted(seen.items())
    return out


def _parse_exclude(exclude: str):
    return {m.strip() for m in exclude.split(",") if m.strip()}


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _plot_single_task(base: Path, task: str, out_path: Path, model_label: str,
                      excluded, max_step: int):
    data = collect_accuracy(base, task, max_step)
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
    ax.set_title(f"Single-reward vs CIRCA - {model_label} ({task})", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=13)
    ax.set_ylim(0, None)
    ax.set_xlim(0, max_step)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")


def _plot_panel(base: Path, tasks, out_path: Path, model_label: str,
                excluded, max_step: int):
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.6 * len(tasks), 6.0),
                             dpi=170, sharex=True)
    if len(tasks) == 1:
        axes = [axes]

    legend_items = {}
    for i, task in enumerate(tasks):
        ax = axes[i]
        data = collect_accuracy(base, task, max_step)
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

    fig.suptitle(f"Single-reward vs CIRCA accuracy trajectories - {model_label}",
                 y=0.99, fontsize=21)
    if legend_items:
        labels = list(legend_items.keys())
        handles = [legend_items[k] for k in labels]
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(4, len(labels)), frameon=True,
                   bbox_to_anchor=(0.5, -0.01), fontsize=14)

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


def _plot_metric(csv_path: Path, metric_key: str, out_path: Path,
                 model_label: str, excluded, max_step: int, stride: int):
    metric_col, ylabel, title_frag = METRICS[metric_key]
    data = collect_metric(csv_path, metric_col, max_step, stride)
    if not data:
        raise SystemExit(f"no {metric_key} data found in {csv_path}")

    fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=170)
    for method in ORDER:
        if method in excluded:
            continue
        pts = data.get(method, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        label, color = STYLE_BY_METHOD[method]
        ax.plot(xs, ys, color=color, label=label, **_line_kw(method, markersize=6))

    ax.set_xlabel("Training step", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(f"Single-reward vs CIRCA - {title_frag} ({model_label})", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=13)
    ax.set_xlim(0, max_step)
    ax.set_ylim(0, None)
    ax.set_xticks(list(range(0, max_step + 1, stride)))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


def _plot_paper_panel(base: Path, completion_csv: Path, entropy_csv: Path,
                      tasks, out_path: Path, model_label: str,
                      excluded, max_step: int, stride: int):
    """Single-row paper panel: one accuracy panel per task, then the two
    training-signal panels.

      [ GSM8K | MATH500 | Mean completion length | Mean entropy ]

    Drawn near its printed size (12.5in wide, included at ~0.96\\textwidth)
    so the in-figure text is not shrunk to illegibility by LaTeX's downscale;
    the earlier two-row version was drawn 18in wide and printed at 5.3in.
    """
    n_acc = len(tasks)
    n_panels = n_acc + 2

    TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS = 15, 13, 11, 13

    # Fixed-size canvas at the aspect the published PDF already has (3.7014),
    # so \includegraphics[width=\textwidth] keeps rendering it at the same
    # printed size while the internal spacing changes.
    fig, axes = plt.subplots(1, n_panels, figsize=(11.8445, 3.2), dpi=300)
    axes = list(axes)

    legend_items = {}
    completion_data = collect_metric(completion_csv, METRICS["completion"][0],
                                     max_step, stride)
    entropy_data    = collect_metric(entropy_csv,    METRICS["entropy"][0],
                                     max_step, stride)

    # Short y-labels: a 2.2in-tall panel cannot fit "Accuracy (pass@1, greedy)".
    SHORT_YLABEL = {"acc": "pass@1 (greedy)",
                    "completion": "Tokens",
                    "entropy": "Entropy"}

    panel_specs = ([("acc", t, ax) for t, ax in zip(tasks, axes[:n_acc])]
                   + [("completion", None, axes[n_acc]),
                      ("entropy",    None, axes[n_acc + 1])])

    for kind, task, ax in panel_specs:
        if kind == "acc":
            data = collect_accuracy(base, task, max_step)
            title = task.upper().replace("_", "-")
        else:
            data = completion_data if kind == "completion" else entropy_data
            title = METRICS[kind][2].capitalize()

        for method in ORDER:
            if method in excluded:
                continue
            pts = data.get(method, [])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            label, color = STYLE_BY_METHOD[method]
            line, = ax.plot(xs, ys, color=color,
                            **_line_kw(method, markersize=3.5), label=label)
            if label not in legend_items:
                legend_items[label] = line

        ax.set_title(title, fontsize=TITLE_FS)
        ax.set_ylabel(SHORT_YLABEL[kind], fontsize=LABEL_FS)
        ax.set_xlabel("Training step", fontsize=LABEL_FS)
        ax.set_ylim(0, None)
        ax.set_xlim(0, max_step)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.grid(True, alpha=0.3)
        # Full box + framed legend below: same conventions as Figure 3
        # (_plot_combined_panel in plot_trajectory_panel.py).

    # Reserve a band at the bottom for the x-labels AND the legend; the previous
    # bbox_inches="tight" + negative anchor pushed the legend hard against the
    # "Training step" labels.
    # bottom=0.34 (was 0.30): the x-axis "Training step" labels and the legend
    # row were sharing too little vertical band, so the legend frame nearly
    # touched the axes' bottom spine. More room here, and the legend anchored
    # further from the very edge of the fixed canvas below.
    fig.subplots_adjust(left=0.052, right=0.995, top=0.88, bottom=0.34,
                        wspace=0.42)
    if legend_items:
        labels = list(legend_items.keys())
        handles = [legend_items[k] for k in labels]
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(5, len(labels)), frameon=True, handlelength=1.6,
                   columnspacing=2.4, bbox_to_anchor=(0.5, 0.02),
                   prop={"weight": "bold", "size": LEGEND_FS})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    sibling = out_path.with_suffix(".png" if out_path.suffix == ".pdf" else ".pdf")
    fig.savefig(sibling)
    plt.close(fig)
    print(f"wrote {out_path} and {sibling}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE),
                    help="Trajectory root directory containing method_ckptXXX folders")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model-label", default="Qwen2.5-1.5B")
    ap.add_argument("--panel", action="store_true",
                    help="1x3 accuracy panel (one shared legend)")
    ap.add_argument("--panel-tasks", default="gsm8k,math500,mmlu_pro",
                    help="Comma-separated tasks for --panel and --paper-panel")
    ap.add_argument("--completion", action="store_true",
                    help="Plot mean completion length vs training step")
    ap.add_argument("--entropy", action="store_true",
                    help="Plot mean entropy vs training step")
    ap.add_argument("--completion-csv",
                    default="cluster_self-certainty_multi-reward_mean_completion.csv",
                    help="Completion-length CSV (relative to --base)")
    ap.add_argument("--entropy-csv",
                    default="cluster_self-certainty_multi-reward_mean_entropy.csv",
                    help="Entropy CSV (relative to --base)")
    ap.add_argument("--max-step", type=int, default=MAX_STEP_DEFAULT,
                    help=f"Cap the x-axis at this training step (default {MAX_STEP_DEFAULT})")
    ap.add_argument("--stride", type=int, default=STRIDE_DEFAULT,
                    help=f"Step stride for CSV-based metrics (default {STRIDE_DEFAULT})")
    ap.add_argument("--exclude", default="",
                    help="Comma-separated method keys to skip")
    ap.add_argument("--paper-panel", action="store_true",
                    help="Render the 2x3 paper figure (3 acc + completion + entropy)")
    ap.add_argument("--paper-panel-out", default=None,
                    help="Override output path for --paper-panel")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    excluded = _parse_exclude(args.exclude)

    if args.paper_panel:
        tasks = [t.strip() for t in args.panel_tasks.split(",") if t.strip()]
        completion_csv = (base / args.completion_csv).resolve()
        entropy_csv    = (base / args.entropy_csv).resolve()
        repo_root = base.parent.parent  # eval_results_final/single_multi_reward_ablation -> repo
        out_path = Path(args.paper_panel_out) if args.paper_panel_out else \
            repo_root / "paper" / "figures" / "ablation" / "single_multi_reward_plot.pdf"
        _plot_paper_panel(base, completion_csv, entropy_csv, tasks, out_path,
                          args.model_label, excluded, args.max_step, args.stride)
        if out_path.suffix == ".pdf":
            _plot_paper_panel(base, completion_csv, entropy_csv, tasks,
                              out_path.with_suffix(".png"),
                              args.model_label, excluded, args.max_step, args.stride)
        return

    if args.completion:
        csv_path = (base / args.completion_csv).resolve()
        out_path = Path(args.out) if args.out else \
            base / "plots" / "completion_length_trajectory.pdf"
        _plot_metric(csv_path, "completion", out_path, args.model_label,
                     excluded, args.max_step, args.stride)
        if out_path.suffix == ".pdf":
            _plot_metric(csv_path, "completion", out_path.with_suffix(".png"),
                         args.model_label, excluded, args.max_step, args.stride)
        return

    if args.entropy:
        csv_path = (base / args.entropy_csv).resolve()
        out_path = Path(args.out) if args.out else \
            base / "plots" / "entropy_trajectory.pdf"
        _plot_metric(csv_path, "entropy", out_path, args.model_label,
                     excluded, args.max_step, args.stride)
        if out_path.suffix == ".pdf":
            _plot_metric(csv_path, "entropy", out_path.with_suffix(".png"),
                         args.model_label, excluded, args.max_step, args.stride)
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

    out_path = Path(args.out) if args.out else \
        base / "plots" / f"trajectory_{args.task}.png"
    _plot_single_task(base, args.task, out_path, args.model_label,
                      excluded, args.max_step)


if __name__ == "__main__":
    main()
