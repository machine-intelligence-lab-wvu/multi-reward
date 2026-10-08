"""Aggregate per-(run, task) JSONs into a single summary table.

Writes SUMMARY.md, SUMMARY.csv, and SUMMARY.xlsx (if openpyxl is installed).

When the input JSONs contain pass@k / maj@k (i.e. n_samples > 1 sweeps),
all three metrics are emitted side-by-side in every output format.

Usage:
  python evaluation/summarize.py --output_dir evaluation_results/best_ckpts_v3_passk
"""

import argparse
import csv
import json
from pathlib import Path


RUN_ORDER = [
    "qwen25_1.5b_base",
    "grpo_gt_ckpt300",
    "intuitor_imp_ckpt80",
    "intuitor_klcov_ckpt80",
    "intuitor_klcov_ckpt160",
    "multi_reward_base_ckpt150",
    "multi_reward_klcov05_ckpt160",
    "multi_reward_klcov02_ckpt160",
    # Qwen2.5-3B
    "multi_reward_3b_noklcov_ckpt160",
]

TASK_ORDER = ["gsm8k", "math500", "aime2024", "aime2025", "mmlu_pro", "cruxeval_o", "lcb"]

PRETTY_RUN = {
    "qwen25_1.5b_base":                "Qwen2.5-1.5B-Base",
    "grpo_gt_ckpt300":                 "GRPO-GT (ckpt-300)",
    "intuitor_imp_ckpt80":             "Intuitor (imp) (ckpt-80)",
    "intuitor_klcov_ckpt80":           "Intuitor + KL-Cov (ckpt-80)",
    "intuitor_klcov_ckpt160":          "Intuitor + KL-Cov (ckpt-160)",
    "multi_reward_base_ckpt150":       "Multi-reward (ckpt-150)",
    "multi_reward_klcov05_ckpt160":    "Multi-reward + KL-Cov β=0.05 (ckpt-160)",
    "multi_reward_klcov02_ckpt160":    "Multi-reward + KL-Cov β=0.02 (ckpt-160)",
    # Qwen2.5-3B
    "multi_reward_3b_noklcov_ckpt160": "Multi-reward 3B (no KL-Cov) (ckpt-160)",
}
PRETTY_TASK = {
    "gsm8k":      "GSM8K",
    "math500":    "MATH500",
    "aime2024":   "AIME24",
    "aime2025":   "AIME25",
    "mmlu_pro":   "MMLU-Pro",
    "cruxeval_o": "CRUXEval-O",
    "lcb":        "LCB",
}


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="evaluation_results/best_ckpts_v1")
    args = ap.parse_args()

    root = Path(args.output_dir)
    if not root.exists():
        print(f"{root} does not exist"); return

    # ── Load all JSONs ──
    # results[run][task] = {"pass@1": float, "pass@k": float|None, "maj@k": float|None, "k": int}
    results = {}
    global_k = None
    for run_dir in root.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        run = run_dir.name
        results[run] = {}
        for task_json in run_dir.glob(f"{run}__*.json"):
            try:
                d = json.loads(task_json.read_text())
                k = d.get("n_samples", 1)
                p1 = d.get("pass@1", d.get("accuracy"))
                pk = d.get(f"pass@{k}") if k > 1 else None
                mk = d.get(f"maj@{k}") if k > 1 else None
                results[run][d["task"]] = {"pass@1": p1, "pass@k": pk, "maj@k": mk, "k": k}
                if k > 1 and global_k is None:
                    global_k = k
            except Exception as e:
                print(f"  skip {task_json}: {e}")

    has_passk = global_k is not None
    k_label = global_k if has_passk else 1

    ordered_runs = [r for r in RUN_ORDER if r in results] + \
                   [r for r in results if r not in RUN_ORDER]
    present_tasks = [t for t in TASK_ORDER if any(t in results[r] for r in ordered_runs)]

    # ─────────────────────────────────────────────
    # Markdown — one table per metric (when passk)
    # ─────────────────────────────────────────────
    def _md_table(metric_key: str, metric_label: str) -> str:
        header = ["Model", "Train data"] + [PRETTY_TASK[t] for t in present_tasks]
        lines = [
            f"### {metric_label}" + (f"   (n_samples={global_k}, T>0)" if metric_key != "pass@1" or has_passk else ""),
            "",
            "| " + " | ".join(header) + " |",
            "|---" * len(header) + "|",
        ]
        for run in ordered_runs:
            row = [PRETTY_RUN.get(run, run), "math-lighteval" if "base" not in run else "—"]
            for t in present_tasks:
                cell = results[run].get(t, {}).get(metric_key)
                row.append(_fmt(cell) if cell is not None else "—")
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    md_sections = [_md_table("pass@1", "pass@1")]
    if has_passk:
        md_sections.append(_md_table("pass@k", f"pass@{global_k}"))
        md_sections.append(_md_table("maj@k", f"maj@{global_k}"))
    md = "\n\n".join(md_sections)

    md_path = root / "SUMMARY.md"
    md_path.write_text(md + "\n")

    # ─────────────────────────────────────────────
    # CSV — wide format with one column per (task, metric)
    # E.g.  gsm8k_pass@1, gsm8k_pass@8, gsm8k_maj@8, math500_pass@1, ...
    # ─────────────────────────────────────────────
    csv_path = root / "SUMMARY.csv"
    metrics = ["pass@1"] + ([f"pass@{global_k}", f"maj@{global_k}"] if has_passk else [])
    header = ["run", "train_data"]
    for t in present_tasks:
        for m in metrics:
            header.append(f"{t}_{m}")

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for run in ordered_runs:
            row = [run, "math-lighteval" if "base" not in run else "—"]
            for t in present_tasks:
                cell = results[run].get(t, {})
                for m in metrics:
                    key = {"pass@1": "pass@1", f"pass@{global_k}": "pass@k", f"maj@{global_k}": "maj@k"}[m]
                    v = cell.get(key)
                    row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "")
            w.writerow(row)

    # ─────────────────────────────────────────────
    # Excel — one sheet per metric, best-in-column highlighted
    # ─────────────────────────────────────────────
    xlsx_path = root / "SUMMARY.xlsx"
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter

        wb = Workbook()
        wb.remove(wb.active)  # start clean

        def _write_sheet(title: str, metric_key: str, metric_label: str):
            ws = wb.create_sheet(title=title)
            header = ["Method", "Train data"] + [PRETTY_TASK[t] for t in present_tasks]
            ws.append(header)
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="305496")
                cell.alignment = Alignment(horizontal="center", vertical="center")

            for run in ordered_runs:
                row = [PRETTY_RUN.get(run, run), "math-lighteval" if "base" not in run else "—"]
                for t in present_tasks:
                    v = results[run].get(t, {}).get(metric_key)
                    row.append(v if v is not None else None)
                ws.append(row)

            # Format numeric cells
            for r in range(2, ws.max_row + 1):
                for c in range(3, 3 + len(present_tasks)):
                    cell = ws.cell(row=r, column=c)
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = "0.0000"
                        cell.alignment = Alignment(horizontal="right")

            # Auto-width
            for col in ws.columns:
                max_len = max((len(str(c.value)) if c.value is not None else 0) for c in col)
                ws.column_dimensions[get_column_letter(col[0].column)].width = max(10, max_len + 2)
            ws.freeze_panes = "A2"

            # Best-in-column highlight
            best_fill = PatternFill("solid", fgColor="C6EFCE")
            for c in range(3, 3 + len(present_tasks)):
                col_values = [(r, ws.cell(row=r, column=c).value) for r in range(2, ws.max_row + 1)]
                numeric = [(r, v) for r, v in col_values if isinstance(v, (int, float))]
                if numeric:
                    best_r, _ = max(numeric, key=lambda rv: rv[1])
                    cell = ws.cell(row=best_r, column=c)
                    cell.font = Font(bold=True)
                    cell.fill = best_fill

        _write_sheet("pass@1", "pass@1", "pass@1")
        if has_passk:
            _write_sheet(f"pass@{global_k}", "pass@k", f"pass@{global_k}")
            _write_sheet(f"maj@{global_k}", "maj@k",  f"maj@{global_k}")

            # Bonus: a "delta" sheet — pass@k minus pass@1 (the diversity bonus)
            ws = wb.create_sheet(title=f"Δ(p@{global_k}-p@1)")
            header = ["Method", "Train data"] + [PRETTY_TASK[t] for t in present_tasks]
            ws.append(header)
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="B85450")
                cell.alignment = Alignment(horizontal="center", vertical="center")
            for run in ordered_runs:
                row = [PRETTY_RUN.get(run, run), "math-lighteval" if "base" not in run else "—"]
                for t in present_tasks:
                    cell = results[run].get(t, {})
                    p1 = cell.get("pass@1"); pk = cell.get("pass@k")
                    if isinstance(p1, (int, float)) and isinstance(pk, (int, float)):
                        row.append(pk - p1)
                    else:
                        row.append(None)
                ws.append(row)
            for r in range(2, ws.max_row + 1):
                for c in range(3, 3 + len(present_tasks)):
                    cell = ws.cell(row=r, column=c)
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = "+0.0000;-0.0000;0"
                        cell.alignment = Alignment(horizontal="right")
            for col in ws.columns:
                max_len = max((len(str(c.value)) if c.value is not None else 0) for c in col)
                ws.column_dimensions[get_column_letter(col[0].column)].width = max(10, max_len + 2)
            ws.freeze_panes = "A2"

        wb.save(xlsx_path)
        saved = f"{md_path}, {csv_path}, {xlsx_path}"
    except ImportError:
        saved = f"{md_path}, {csv_path}"
        print("(install openpyxl for Excel output: pip install openpyxl)")

    # Console preview
    print(md)
    print(f"\nSaved: {saved}")


if __name__ == "__main__":
    main()
