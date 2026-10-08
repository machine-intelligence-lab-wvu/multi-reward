#!/bin/bash
# ============================================================
# Run Multi-Reward (no KL-Cov) + INTUITOR — resume from 500 to 1000 steps
# Sequential: Multi-Reward first, then INTUITOR
# ============================================================

set -euo pipefail

PROJECT_DIR="/home/WVU-AD/aa00307/projects/unsupervised-rl"

echo "============================================================"
echo "  Training sweep — steps 500 → 1000"
echo "  1) Multi-Reward (no KL-Cov)"
echo "  2) INTUITOR"
echo "============================================================"

# ── 1. Multi-Reward (no KL-Cov) ─────────────────────────────
echo ""
echo "[$(date)] Starting Multi-Reward training (500 → 1000)..."
bash "${PROJECT_DIR}/open_r1_imp/scripts/run_multi_reward_local.sh"
echo "[$(date)] Multi-Reward done."

# ── 2. INTUITOR ──────────────────────────────────────────────
echo ""
echo "[$(date)] Starting INTUITOR training (500 → 1000)..."
bash "${PROJECT_DIR}/open_r1_imp/scripts/run_intuitor_local.sh"
echo "[$(date)] INTUITOR done."

echo ""
echo "============================================================"
echo "[$(date)] Both training runs complete!"
echo "============================================================"
