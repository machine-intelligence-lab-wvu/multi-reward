#!/bin/bash
# One-off: render the 4 trajectory plots for the 7B sweep.
# Outputs (all under eval_results_final/trajectory_7B/):
#   {gsm8k,math500,aime2024,aime2025}_pass1_trajectory_7b.{pdf,png}
#   SUMMARY_trajectory.csv
# Usage:
#   sbatch open_r1_imp/scripts/submit_plot_trajectory_7b.sh

#SBATCH --job-name=plot-7b
#SBATCH --account=a0264
#SBATCH --qos=slowdown
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=low
#SBATCH --output=logs/plot-7b-%j.out
#SBATCH --error=logs/plot-7b-%j.err
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -euo pipefail

PROJECT_DIR="${HOME}/projects/unsupervised-rl"
cd "${PROJECT_DIR}"
source venv/bin/activate
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

for T in gsm8k math500 aime2024 aime2025; do
    echo "=== plotting ${T} ==="
    python -u evaluation/plot_trajectory.py \
        --output_dir eval_results_final/trajectory_7B \
        --figure_dir eval_results_final/trajectory_7B \
        --task "${T}" \
        --figure_suffix "_7b" \
        --model_label "Qwen2.5-7B"
done

echo "[$(date)] plots done."
ls -la eval_results_final/trajectory_7B/*_7b.png 2>/dev/null
