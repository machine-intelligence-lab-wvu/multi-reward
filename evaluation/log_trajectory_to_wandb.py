"""Upload the training-trajectory evaluation to wandb.

Reads {output_dir}/{method}_ckpt{step}/{method}_ckpt{step}__{task}.json, then for
each method creates ONE wandb run that logs the full (step, pass@1) curve.

It also logs a single "overview" wandb run that contains:
  - a wandb.Table with method/step/pass@1 for all methods
  - the matplotlib PNG as a wandb.Image
so you can view the whole thing as one chart on one page.

No re-evaluation is needed — this pulls from the existing JSONs.

Usage:
  python3 evaluation/log_trajectory_to_wandb.py \
      --output_dir evaluation_results/trajectory_v1 \
      --wandb_project unsupervised-rl-eval \
      --wandb_group   trajectory_v1 \
      --figure_png    paper/figures/gsm8k_pass1_trajectory.png
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="evaluation_results/trajectory_v1")
    p.add_argument("--task", default="gsm8k")
    p.add_argument("--wandb_project", default="unsupervised-rl-eval")
    p.add_argument("--wandb_group",   default="trajectory_v1")
    p.add_argument("--figure_png",    default="paper/figures/gsm8k_pass1_trajectory.png",
                   help="Path to the matplotlib PNG to upload as wandb.Image")
    p.add_argument("--run_suffix",    default="__wb",
                   help="Suffix added to run IDs to avoid clashing with prior runs")
    p.add_argument("--run_name_template", default="{method}",
                   help="Template for per-method wandb run name; supports "
                        "{method} and {task} placeholders. "
                        "e.g. 'trajectory_{task}_3b_{method}'")
    return p.parse_args()


def _collect(root: Path, task: str):
    """Return {method -> [(step, pass@1), ...]} sorted by step."""
    out = defaultdict(list)
    for run_dir in root.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        m = re.match(r"^(.*)_ckpt(\d+)$", run_dir.name)
        if not m:
            continue
        method, step = m.group(1), int(m.group(2))
        j = run_dir / f"{run_dir.name}__{task}.json"
        if not j.exists():
            continue
        try:
            d = json.loads(j.read_text())
            acc = d.get("pass@1", d.get("accuracy"))
            if acc is not None:
                out[method].append((step, float(acc)))
        except Exception as e:
            print(f"  skip {j}: {e}")
    for m in out:
        out[m].sort(key=lambda x: x[0])  # ascending numeric step
    return out


def main():
    args = parse_args()
    root = Path(args.output_dir)
    data = _collect(root, args.task)
    if not data:
        print(f"No trajectory data found under {root} for task={args.task}")
        return

    try:
        import wandb
    except ImportError:
        print("wandb not installed: pip install wandb"); return

    print(f"\nLogging {len(data)} methods to wandb project={args.wandb_project}, "
          f"group={args.wandb_group}")

    # ── Per-method runs (clean ascending step) ──
    for method, points in data.items():
        n = len(points)
        wandb_name = args.run_name_template.format(method=method, task=args.task)
        run_id = f"{args.wandb_group}__{wandb_name}{args.run_suffix}"
        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=wandb_name,
            id=run_id,
            resume="allow",
            job_type="trajectory",
            reinit=True,
            config={"method": method, "task": args.task, "n_checkpoints": n},
        )
        for step, acc in points:
            run.log({f"{args.task}/pass@1": acc, "train_step": step}, step=step)
        run.summary[f"{args.task}/pass@1_peak"] = max(a for _, a in points)
        run.summary[f"{args.task}/pass@1_last"] = points[-1][1]
        run.finish()
        print(f"  ✓ {method}: {n} points logged → run id={run_id}")

    # ── Overview run: table + image ──
    overview = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=f"{args.wandb_group}__overview",
        id=f"{args.wandb_group}__overview{args.run_suffix}",
        resume="allow",
        job_type="overview",
        reinit=True,
    )

    table = wandb.Table(columns=["method", "step", f"{args.task}_pass@1"])
    for method in sorted(data):
        for step, acc in data[method]:
            table.add_data(method, step, acc)
    overview.log({f"{args.task}/trajectory_table": table})

    fig_path = Path(args.figure_png)
    if fig_path.exists():
        overview.log({f"{args.task}/trajectory_plot": wandb.Image(str(fig_path))})
        print(f"  ✓ Uploaded {fig_path} as wandb.Image")
    else:
        print(f"  ! PNG not found at {fig_path} — skipping image upload")

    # Also log a wandb-native multi-line plot built from the table
    try:
        overview.log({
            f"{args.task}/trajectory_lineplot": wandb.plot.line_series(
                xs=[[s for s, _ in data[m]] for m in sorted(data)],
                ys=[[a for _, a in data[m]] for m in sorted(data)],
                keys=sorted(data),
                title=f"{args.task.upper()} pass@1 over training",
                xname="train_step",
            ),
        })
        print("  ✓ Uploaded wandb native line-series plot")
    except Exception as e:
        print(f"  ! native line-series failed: {e}")

    overview.finish()
    print("\nDone. View the runs at:")
    print(f"  https://wandb.ai/<entity>/{args.wandb_project}?groupKeys[]=group&groupValues[]={args.wandb_group}")


if __name__ == "__main__":
    main()
