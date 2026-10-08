"""Build the paper-ready final-checkpoint comparison table + pass@k curve plot.

Reads JSONs produced by `evaluation/final_table.py` and (optionally) the
existing trajectory SUMMARY.csv to pull greedy pass@1 as a side column.

Usage:
    python -u evaluation/summarize_final_table.py --scale 3b
    python -u evaluation/summarize_final_table.py --scale 7b
"""
import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METHOD_ORDER = ["base", "intuitor", "grpo_gt", "multi_reward_klcov05"]
PRETTY_METHOD = {
    "base":                 "Base Qwen2.5",
    "intuitor":             "INTUITOR",
    "grpo_gt":              "GRPO-GT",
    "multi_reward_klcov05": "Multi-reward + KL-Cov (ours)",
}
TASK_ORDER = ["gsm8k", "math500", "aime2024", "aime2025"]
PRETTY_TASK = {
    "gsm8k":    "GSM8K",
    "math500":  "MATH500",
    "aime2024": "AIME24",
    "aime2025": "AIME25",
}

# Map final_table method names → trajectory SUMMARY.csv run names + pinned step
TRAJ_RUN_NAME = {
    "3b": {
        "intuitor":             ("intuitor_3b",                   340),
        "grpo_gt":              ("grpo_gt_3b",                    340),
        "multi_reward_klcov05": ("multi_reward_final_klcov05_3b", 340),
    },
    "7b": {
        "intuitor":             ("intuitor_7b",                   240),
        "grpo_gt":              ("grpo_gt_7b",                    240),
        "multi_reward_klcov05": ("multi_reward_final_klcov05_7b", 240),
    },
}


def _load_final_table(root: Path):
    """Read all {method}/{method}__{task}.json under root."""
    out = {}  # out[method][task] = json dict
    for method_dir in root.iterdir():
        if not method_dir.is_dir() or method_dir.name.startswith("_"):
            continue
        method = method_dir.name
        out[method] = {}
        for jf in method_dir.glob(f"{method}__*.json"):
            try:
                d = json.loads(jf.read_text())
                out[method][d["task"]] = d
            except Exception as e:
                print(f"  skip {jf}: {e}")
    return out


def _load_greedy_passk(scale: str):
    """Pull greedy pass@1 at the pinned step from trajectory SUMMARY.csv."""
    csv_path = PROJECT_ROOT / f"eval_results_final/trajectory_{scale.upper()}/SUMMARY.csv"
    if not csv_path.exists():
        print(f"[summarize_final_table] no trajectory SUMMARY at {csv_path}, "
              "skipping greedy side column")
        return {}
    greedy = {}  # greedy[method][task] = float
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    for method, (traj_name, step) in TRAJ_RUN_NAME[scale].items():
        wanted = f"{traj_name}_ckpt{step:03d}"
        row = next((r for r in rows if r["run"] == wanted), None)
        if row is None:
            print(f"  [greedy] no row for {wanted}")
            continue
        greedy[method] = {}
        for t in TASK_ORDER:
            v = row.get(f"{t}_pass@1")
            if v not in (None, ""):
                try:
                    greedy[method][t] = float(v)
                except ValueError:
                    pass
    return greedy


def _fmt(v, nd=4):
    if v is None: return "—"
    return f"{v:.{nd}f}"


def build_markdown(data, greedy, scale, step):
    """data[method][task] = json; greedy[method][task] = float."""
    lines = []
    lines.append(f"# Final-checkpoint comparison — Qwen2.5-{scale.upper()} @ step {step}")
    lines.append("")
    lines.append("Sampling protocol: T=1.0, top_p=1.0. Pass@1 = avg over N samples "
                 "(Chen et al. 2021 / DeepSeek-R1 / Zuo et al. 2025).")
    lines.append("Per-task N: GSM8K=8, MATH500=16, AIME24=32, AIME25=32. "
                 "AIME max_new_tokens=8192 (others as default).")
    lines.append("")

    # ── Avg-Pass@1 table ─────────────────────────────────────────────
    header = ["Method"] + [PRETTY_TASK[t] for t in TASK_ORDER]
    lines.append("### avg-Pass@1 (T=1)")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|---" * len(header) + "|")
    for m in METHOD_ORDER:
        if m not in data: continue
        row = [PRETTY_METHOD[m]]
        for t in TASK_ORDER:
            d = data[m].get(t)
            row.append(_fmt(d["pass@1"]) if d else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── Greedy pass@1 side table ─────────────────────────────────────
    if greedy:
        lines.append("### Greedy Pass@1 (T=0, same checkpoint)")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|---" * len(header) + "|")
        for m in METHOD_ORDER:
            if m == "base":
                continue
            row = [PRETTY_METHOD[m]]
            for t in TASK_ORDER:
                row.append(_fmt(greedy.get(m, {}).get(t)))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # ── pass@k curves (full table) ───────────────────────────────────
    ks = (1, 2, 4, 8, 16, 32)
    for t in TASK_ORDER:
        any_have = any(t in data.get(m, {}) for m in METHOD_ORDER)
        if not any_have: continue
        lines.append(f"### Pass@k — {PRETTY_TASK[t]}")
        lines.append("")
        head = ["Method"] + [f"k={k}" for k in ks]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|---" * len(head) + "|")
        for m in METHOD_ORDER:
            d = data.get(m, {}).get(t)
            if not d: continue
            curve = d.get("pass@k_curve", {})
            row = [PRETTY_METHOD[m]]
            for k in ks:
                v = curve.get(str(k))
                row.append(_fmt(v) if v is not None else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # ── Maj@k curves ─────────────────────────────────────────────────
    for t in TASK_ORDER:
        any_have = any(t in data.get(m, {}) for m in METHOD_ORDER)
        if not any_have: continue
        lines.append(f"### Maj@k — {PRETTY_TASK[t]}")
        lines.append("")
        head = ["Method"] + [f"k={k}" for k in ks]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|---" * len(head) + "|")
        for m in METHOD_ORDER:
            d = data.get(m, {}).get(t)
            if not d: continue
            curve = d.get("maj@k_curve", {})
            row = [PRETTY_METHOD[m]]
            for k in ks:
                v = curve.get(str(k))
                row.append(_fmt(v) if v is not None else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


def write_csv(data, out_path: Path):
    """Long-form CSV: method, task, metric, k, value."""
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "task", "metric", "k", "value"])
        for m, by_task in data.items():
            for t, d in by_task.items():
                w.writerow([m, t, "pass@1_avg", 1, d["pass@1"]])
                for k, v in (d.get("pass@k_curve") or {}).items():
                    w.writerow([m, t, "pass@k", k, v])
                for k, v in (d.get("maj@k_curve") or {}).items():
                    w.writerow([m, t, "maj@k", k, v])


def make_plot(data, scale, step, out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot"); return
    methods_present = [m for m in METHOD_ORDER if m in data]
    tasks_present = [t for t in TASK_ORDER if any(t in data.get(m, {}) for m in methods_present)]
    if not tasks_present: return

    fig, axes = plt.subplots(1, len(tasks_present),
                              figsize=(4 * len(tasks_present), 3.2), sharey=False)
    if len(tasks_present) == 1:
        axes = [axes]
    for ax, t in zip(axes, tasks_present):
        for m in methods_present:
            d = data[m].get(t)
            if not d: continue
            curve = d.get("pass@k_curve") or {}
            xs = sorted(int(k) for k in curve.keys())
            ys = [curve[str(k)] for k in xs]
            ax.plot(xs, ys, marker="o", label=PRETTY_METHOD[m])
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.set_xticklabels([1, 2, 4, 8, 16, 32])
        ax.set_xlabel("k")
        ax.set_ylabel("pass@k")
        ax.set_title(PRETTY_TASK[t])
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Qwen2.5-{scale.upper()} @ step {step} — pass@k (T=1.0)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[plot] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["3b", "7b"], required=True)
    ap.add_argument("--root", default=None,
                    help="default: eval_results_final/final_table/{scale}")
    args = ap.parse_args()

    root = Path(args.root or f"eval_results_final/final_table/{args.scale}")
    if not root.exists():
        print(f"{root} does not exist"); return

    step = 340 if args.scale == "3b" else 240

    data = _load_final_table(root)
    if not data:
        print(f"[summarize_final_table] no JSONs found in {root}"); return

    greedy = _load_greedy_passk(args.scale)

    md = build_markdown(data, greedy, args.scale, step)
    md_path = root / "TABLE.md"
    md_path.write_text(md + "\n")
    print(f"[summarize_final_table] wrote {md_path}")

    csv_path = root / "passk_curves.csv"
    write_csv(data, csv_path)
    print(f"[summarize_final_table] wrote {csv_path}")

    plot_path = root / "passk_plot.png"
    make_plot(data, args.scale, step, plot_path)


if __name__ == "__main__":
    main()
