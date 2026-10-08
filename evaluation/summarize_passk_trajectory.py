"""Aggregate + plot the pass@k trajectory results.

Reads JSONs under eval_results_final/passk_trajectory_{SCALE}/{method}_ckpt{step}/
and emits:
  - SUMMARY.csv           (long-form: scale, method, step, task, metric, k, value)
  - SUMMARY.md            (table per task: avg-pass@1 vs step + peak pass@k)
  - plots/avg_pass1_trajectory.{pdf,png}     (1×N_tasks, lines per method)
  - plots/passk_at_step{S}.{pdf,png}         (one per step, panels per task)

Usage:
    python -u evaluation/summarize_passk_trajectory.py --scale 7b
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys as _sys
_sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "paper_figures"))
from palette import style as _style

METHOD_STYLE = {k: (_style(k)[0], _style(k)[1], _style(k)[3])
                for k in ("grpo_gt", "intuitor", "multi_reward_klcov05")}
ORDER = ["grpo_gt", "intuitor", "multi_reward_klcov05"]
TASK_PRETTY = {"gsm8k": "GSM8K", "math500": "MATH500",
               "aime2024": "AIME 2024", "aime2025": "AIME 2025"}


def _load_root(root: Path):
    """{method}{step}{task} -> json dict."""
    out = defaultdict(lambda: defaultdict(dict))
    pat = re.compile(r"^(?P<method>.+)_ckpt(?P<step>\d+)$")
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("_") or d.name == "plots":
            continue
        m = pat.match(d.name)
        if not m:
            continue
        method, step = m["method"], int(m["step"])
        for jf in d.glob(f"{d.name}__*.json"):
            try:
                payload = json.loads(jf.read_text())
                out[method][step][payload["task"]] = payload
            except Exception as e:
                print(f"  skip {jf}: {e}")
    return out


def _fmt(v, nd=4):
    if v is None: return "—"
    return f"{v:.{nd}f}"


def _write_csv(data, out_path: Path):
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "step", "task", "metric", "k", "value", "n_samples"])
        for method, by_step in data.items():
            for step, by_task in by_step.items():
                for task, d in by_task.items():
                    w.writerow([method, step, task, "pass@1_avg", 1,
                                d.get("pass@1"), d.get("n_samples")])
                    for k, v in (d.get("pass@k_curve") or {}).items():
                        w.writerow([method, step, task, "pass@k", k, v,
                                    d.get("n_samples")])
                    for k, v in (d.get("maj@k_curve") or {}).items():
                        w.writerow([method, step, task, "maj@k", k, v,
                                    d.get("n_samples")])


def _write_md(data, scale, out_path: Path):
    """Per-task table: rows = step, columns = methods. Plus a per-method
    summary showing peak avg-pass@1 and the step it was reached."""
    lines = []
    lines.append(f"# pass@k trajectory — Qwen2.5-{scale.upper()}")
    lines.append("")
    lines.append("Sampling: T=1.0, top_p=1.0. avg-pass@1 = mean correctness "
                 "over N samples per problem (DeepSeek-R1 / Chen 2021 convention).")
    lines.append("")
    tasks_present = set()
    for by_step in data.values():
        for by_task in by_step.values():
            tasks_present.update(by_task.keys())
    tasks_present = [t for t in ("gsm8k","math500","aime2024","aime2025")
                     if t in tasks_present]

    for task in tasks_present:
        lines.append(f"## {TASK_PRETTY.get(task, task)} — avg-pass@1 vs step")
        lines.append("")
        steps_for_task = sorted({s for m in data for s in data[m] if task in data[m][s]})
        header = ["step"] + [METHOD_STYLE[m][0] for m in ORDER if m in data]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|---" * len(header) + "|")
        for s in steps_for_task:
            row = [str(s)]
            for method in ORDER:
                if method not in data: continue
                d = data[method].get(s, {}).get(task)
                row.append(_fmt(d.get("pass@1")) if d else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # Peak summary
        lines.append(f"### {TASK_PRETTY.get(task, task)} — peak avg-pass@1")
        lines.append("")
        lines.append("| method | peak avg-pass@1 | step |")
        lines.append("|---|---|---|")
        for method in ORDER:
            if method not in data: continue
            best = None
            for s, by_task in data[method].items():
                d = by_task.get(task)
                if not d: continue
                v = d.get("pass@1")
                if v is None: continue
                if best is None or v > best[0]:
                    best = (v, s)
            if best is None:
                lines.append(f"| {METHOD_STYLE[method][0]} | — | — |")
            else:
                lines.append(f"| {METHOD_STYLE[method][0]} | {best[0]:.4f} | {best[1]} |")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


def _plot_avg_pass1_trajectory(data, scale, out_path: Path):
    tasks_present = set()
    for by_step in data.values():
        for by_task in by_step.values():
            tasks_present.update(by_task.keys())
    tasks = [t for t in ("gsm8k","math500","aime2024","aime2025") if t in tasks_present]
    if not tasks: return

    fig, axes = plt.subplots(1, len(tasks),
                              figsize=(6.6 * len(tasks), 5.6), dpi=170,
                              sharex=True)
    if len(tasks) == 1:
        axes = [axes]

    legend_items = {}
    for ax, task in zip(axes, tasks):
        for method in ORDER:
            if method not in data: continue
            pts = []
            for s, by_task in data[method].items():
                d = by_task.get(task)
                if d is None: continue
                v = d.get("pass@1")
                if v is None: continue
                pts.append((s, v))
            pts.sort()
            if not pts: continue
            xs = [p[0] for p in pts]
            ys = [p[1] * 100 for p in pts]
            label, color, marker = METHOD_STYLE[method]
            line, = ax.plot(xs, ys, marker=marker, markersize=6, lw=2.4,
                            color=color, label=label)
            if label not in legend_items:
                legend_items[label] = line
            best_idx = max(range(len(ys)), key=lambda k: ys[k])
            ax.plot(xs[best_idx], ys[best_idx], marker="*", markersize=14,
                    color=color, markeredgecolor="black",
                    markeredgewidth=0.6, zorder=5, linestyle="None")
        ax.set_title(TASK_PRETTY.get(task, task.upper()), fontsize=17)
        ax.set_xlabel("Training step", fontsize=15)
        ax.set_ylabel("avg-pass@1 (%)", fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)

    fig.suptitle(f"Qwen2.5-{scale.upper()} — avg-pass@1 trajectory (T=1.0)",
                 y=0.99, fontsize=20)
    if legend_items:
        labels = list(legend_items.keys())
        handles = [legend_items[k] for k in labels]
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(4, len(labels)), frameon=True,
                   bbox_to_anchor=(0.5, -0.02), fontsize=14)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")
    if out_path.suffix == ".pdf":
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
        print(f"wrote {out_path.with_suffix('.png')}")
    plt.close(fig)


def _plot_passmax_trajectory(data, scale, out_path: Path):
    """Per-task: pass@K_max vs step, where K_max = max k present in the
    pass@k_curve for that (method, step, task) cell. For our protocol this is
    K_max = N_samples = 8 (GSM8K) / 16 (MATH500) / 32 (AIME)."""
    tasks_present = set()
    for by_step in data.values():
        for by_task in by_step.values():
            tasks_present.update(by_task.keys())
    tasks = [t for t in ("gsm8k","math500","aime2024","aime2025") if t in tasks_present]
    if not tasks: return

    fig, axes = plt.subplots(1, len(tasks),
                              figsize=(6.6 * len(tasks), 5.6), dpi=170,
                              sharex=True)
    if len(tasks) == 1:
        axes = [axes]

    legend_items = {}
    # Per-task K_max — assume consistent across methods for the same task.
    task_kmax = {}
    for ax, task in zip(axes, tasks):
        kmax_for_task = None
        for method in ORDER:
            if method not in data: continue
            pts = []
            for s, by_task in data[method].items():
                d = by_task.get(task)
                if d is None: continue
                curve = d.get("pass@k_curve") or {}
                if not curve: continue
                ks = [int(k) for k in curve.keys()]
                k_max = max(ks)
                if kmax_for_task is None:
                    kmax_for_task = k_max
                v = curve[str(k_max)]
                pts.append((s, v))
            pts.sort()
            if not pts: continue
            xs = [p[0] for p in pts]
            ys = [p[1] * 100 for p in pts]
            label, color, marker = METHOD_STYLE[method]
            line, = ax.plot(xs, ys, marker=marker, markersize=6, lw=2.4,
                            color=color, label=label)
            if label not in legend_items:
                legend_items[label] = line
            best_idx = max(range(len(ys)), key=lambda k: ys[k])
            ax.plot(xs[best_idx], ys[best_idx], marker="*", markersize=14,
                    color=color, markeredgecolor="black",
                    markeredgewidth=0.6, zorder=5, linestyle="None")
        task_kmax[task] = kmax_for_task or 1
        ax.set_title(f"{TASK_PRETTY.get(task, task.upper())} "
                     f"(pass@{task_kmax[task]})", fontsize=17)
        ax.set_xlabel("Training step", fontsize=15)
        ax.set_ylabel(f"pass@{task_kmax[task]} (%)", fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)

    fig.suptitle(f"Qwen2.5-{scale.upper()} — pass@$K_{{\\max}}$ trajectory "
                 f"($T{{=}}1.0$, coverage)",
                 y=0.99, fontsize=20)
    if legend_items:
        labels = list(legend_items.keys())
        handles = [legend_items[k] for k in labels]
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(4, len(labels)), frameon=True,
                   bbox_to_anchor=(0.5, -0.02), fontsize=14)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")
    if out_path.suffix == ".pdf":
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
        print(f"wrote {out_path.with_suffix('.png')}")
    plt.close(fig)


def _plot_combined_panel(data, scale, out_path: Path):
    """1x4 panel with a hidden spacer column carrying the dotted vertical
    separator: [avg-P@1 GSM8K, avg-P@1 MATH500] | SPACER | [pass@K GSM8K,
    pass@K MATH500]. Caption holds the descriptive header; no suptitle."""
    tasks_present = set()
    for by_step in data.values():
        for by_task in by_step.values():
            tasks_present.update(by_task.keys())
    tasks_left  = [t for t in ("gsm8k", "math500") if t in tasks_present]
    tasks_right = [t for t in ("gsm8k", "math500") if t in tasks_present]
    if not tasks_left or not tasks_right:
        return

    nL, nR = len(tasks_left), len(tasks_right)
    fig = plt.figure(figsize=(4.2 * (nL + nR) + 0.6, 5.2), dpi=170)
    # 5-column layout: nL panels, 1 narrow spacer, nR panels.
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(
        1, nL + 1 + nR,
        width_ratios=[1] * nL + [0.18] + [1] * nR,
        wspace=0.35,
    )
    left_axes  = [fig.add_subplot(gs[0, i])           for i in range(nL)]
    spacer_ax  = fig.add_subplot(gs[0, nL])
    spacer_ax.axis("off")
    right_axes = [fig.add_subplot(gs[0, nL + 1 + i])  for i in range(nR)]

    legend_items = {}

    def _draw(ax, task, metric):
        kmax_for_task = None
        for method in ORDER:
            if method not in data: continue
            pts = []
            for s, by_task in data[method].items():
                d = by_task.get(task)
                if d is None: continue
                if metric == "avg":
                    v = d.get("pass@1")
                else:
                    curve = d.get("pass@k_curve") or {}
                    if not curve: continue
                    ks = [int(k) for k in curve.keys()]
                    k_max = max(ks)
                    if kmax_for_task is None: kmax_for_task = k_max
                    v = curve.get(str(k_max))
                if v is None: continue
                pts.append((s, v))
            pts.sort()
            if not pts: continue
            xs = [p[0] for p in pts]
            ys = [p[1] * 100 for p in pts]
            label, color, marker = METHOD_STYLE[method]
            line, = ax.plot(xs, ys, marker=marker, markersize=6, lw=2.4,
                            color=color, label=label)
            if label not in legend_items:
                legend_items[label] = line
            best_idx = max(range(len(ys)), key=lambda k: ys[k])
            ax.plot(xs[best_idx], ys[best_idx], marker="*", markersize=14,
                    color=color, markeredgecolor="black",
                    markeredgewidth=0.6, zorder=5, linestyle="None")
        return kmax_for_task

    for i, t in enumerate(tasks_left):
        ax = left_axes[i]
        _draw(ax, t, "avg")
        ax.set_title(TASK_PRETTY.get(t, t.upper()), fontsize=17)
        ax.set_xlabel("Training step", fontsize=14)
        if i == 0:
            ax.set_ylabel("avg-Pass@1 (%)", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)

    for j, t in enumerate(tasks_right):
        ax = right_axes[j]
        kmax = _draw(ax, t, "kmax")
        suffix = f" (pass@{kmax})" if kmax else ""
        ax.set_title(TASK_PRETTY.get(t, t.upper()) + suffix, fontsize=17)
        ax.set_xlabel("Training step", fontsize=14)
        if j == 0:
            ax.set_ylabel(f"pass@$K_{{\\max}}$ (%)", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)

    # Snug panel layout; the legend lives BELOW the figure (negative
    # bbox_to_anchor y), and bbox_inches="tight" expands the saved PDF
    # to include it. This guarantees the legend never overlaps the
    # per-panel "Training step" xlabels.
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])

    # Dotted vertical separator centered on the (hidden) spacer column,
    # spanning just the plot area.
    fig.canvas.draw()
    sp_box = spacer_ax.get_position()
    sep_x = 0.5 * (sp_box.x0 + sp_box.x1)
    sep_line = plt.Line2D([sep_x, sep_x], [0.06, 0.96],
                          transform=fig.transFigure,
                          color="0.4", linestyle=(0, (2, 3)), linewidth=1.4,
                          zorder=0)
    fig.add_artist(sep_line)

    if legend_items:
        labels = list(legend_items.keys())
        handles = [legend_items[k] for k in labels]
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(4, len(labels)), frameon=True,
                   bbox_to_anchor=(0.5, -0.06), fontsize=13)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")
    if out_path.suffix == ".pdf":
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
        print(f"wrote {out_path.with_suffix('.png')}")
    plt.close(fig)


def _plot_passk_at_step(data, scale, step, out_path: Path):
    tasks_present = set()
    for method in data:
        if step in data[method]:
            tasks_present.update(data[method][step].keys())
    tasks = [t for t in ("gsm8k","math500","aime2024","aime2025") if t in tasks_present]
    if not tasks: return

    fig, axes = plt.subplots(1, len(tasks),
                              figsize=(4.5 * len(tasks), 3.6), dpi=170)
    if len(tasks) == 1:
        axes = [axes]
    legend_items = {}

    for ax, task in zip(axes, tasks):
        for method in ORDER:
            d = data.get(method, {}).get(step, {}).get(task)
            if not d: continue
            curve = d.get("pass@k_curve") or {}
            xs = sorted(int(k) for k in curve.keys())
            ys = [curve[str(k)] for k in xs]
            label, color, marker = METHOD_STYLE[method]
            line, = ax.plot(xs, ys, marker=marker, markersize=5, lw=2.0,
                            color=color, label=label)
            if label not in legend_items:
                legend_items[label] = line
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.set_xticklabels([1, 2, 4, 8, 16, 32])
        ax.set_xlabel("k")
        ax.set_ylabel("pass@k")
        ax.set_title(TASK_PRETTY.get(task, task.upper()))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)

    if legend_items:
        labels = list(legend_items.keys())
        handles = [legend_items[k] for k in labels]
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(4, len(labels)), frameon=True,
                   bbox_to_anchor=(0.5, -0.02), fontsize=11)
    fig.suptitle(f"Qwen2.5-{scale.upper()} pass@k @ step {step} (T=1.0)",
                 y=0.99, fontsize=14)
    fig.tight_layout(rect=[0, 0.1, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")
    if out_path.suffix == ".pdf":
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["3b", "7b"], required=True)
    ap.add_argument("--root", default=None,
                    help="default: eval_results_final/passk_trajectory_{SCALE}")
    args = ap.parse_args()

    root = Path(args.root or
                f"eval_results_final/passk_trajectory_{args.scale.upper()}")
    if not root.exists():
        print(f"{root} does not exist"); return

    data = _load_root(root)
    if not data:
        print(f"[summarize] no JSONs under {root}"); return

    _write_csv(data, root / "SUMMARY.csv")
    print(f"wrote {root / 'SUMMARY.csv'}")
    _write_md(data, args.scale, root / "SUMMARY.md")
    print(f"wrote {root / 'SUMMARY.md'}")

    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_avg_pass1_trajectory(data, args.scale,
                                plots_dir / "avg_pass1_trajectory.pdf")
    _plot_passmax_trajectory(data, args.scale,
                              plots_dir / "passmax_trajectory.pdf")
    _plot_combined_panel(data, args.scale,
                          plots_dir / "combined_trajectory_passk.pdf")
    # pass@k curve at each evaluated step
    all_steps = sorted({s for m in data for s in data[m]})
    for s in all_steps:
        _plot_passk_at_step(data, args.scale, s,
                            plots_dir / f"passk_at_step{s:03d}.pdf")


if __name__ == "__main__":
    main()
