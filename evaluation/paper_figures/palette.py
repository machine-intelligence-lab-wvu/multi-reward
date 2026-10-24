"""Method colors/linestyles shared by all paper figures (the camera-ready scheme).

  INTUITOR (self-certainty)            red
  Multi-reward + KL-Cov (ours)         green
  Multi-reward w/o KL-Cov (ablation)   purple
  Secondary variants                   orange (beta_cov = 0.0005, INTUITOR + KL-Cov), brown (beta_cov = 0.1)
  Cluster voting only                  blue
  GRPO-GT (supervised reference)       black
"""

RED, GREEN, PURPLE = "#d62728", "#2ca02c", "#9467bd"
ORANGE, BROWN, BLUE, BLACK = "#ff7f0e", "#8c564b", "#1f77b4", "#222222"
MUTED = "#8a8985"
GRID = "#d8d7d3"

SOLID = "-"
DASH = (0, (5, 2))
DASHDOT = (0, (5, 1.6, 1, 1.6))
DOT = (0, (1.2, 1.6))

# label, color, linestyle, marker, linewidth, alpha, zorder
_STYLES = {
    "grpo_gt":           ("GRPO-GT (supervised)",                        BLACK,  DASH,    "o", 2.0, 0.95, 3),
    "intuitor":          ("INTUITOR",                                    RED,    DASHDOT, "s", 2.1, 1.00, 4),
    "intuitor_klcov":    (r"INTUITOR + KL-Cov ($\beta_{cov}$=0.02)",     ORANGE, DASHDOT, "s", 2.4, 1.00, 6),
    "cluster_only":      ("Cluster voting only",                         BLUE,   DOT,     "D", 2.1, 1.00, 4),
    "multi_reward":      ("Multi-reward (no KL-Cov)",                    PURPLE, DASH,    "^", 2.5, 1.00, 5),
    "multi_reward_klcov": ("Multi-reward + KL-Cov (ours)",               GREEN,  SOLID,   "^", 3.0, 1.00, 7),
    "multi_reward_klcov_beta0.0005": (r"Multi-reward + KL-Cov ($\beta_{cov}$=0.0005)", ORANGE, SOLID, "v", 2.4, 1.00, 6),
    "multi_reward_klcov_beta0.1":    (r"Multi-reward + KL-Cov ($\beta_{cov}$=0.1)",    BROWN,  SOLID, "P", 2.4, 1.00, 6),
    "grpo_gt_klcov":     (r"GRPO-GT + KL-Cov ($\beta_{cov}$=0.02)",      GREEN,  SOLID,   "^", 2.8, 1.00, 6),
    # GDPO ablation (Appendix)
    "gdpo_ablation_per_channel":       ("Per-channel GDPO norm (ours)", GREEN,  SOLID, "^", 2.8, 1.00, 7),
    "gdpo_ablation_combine_then_norm": ("Combine-then-normalize",       PURPLE, DASH,  "o", 2.4, 1.00, 6),
}

# Run names -> style keys. Run names are the recipe file names used by
# scripts/eval.sh; the extra aliases cover the names used in the original
# results tree.
ALIASES = {
    "grpo_gt_noklcov": "grpo_gt", "grpo_gt_3b": "grpo_gt", "grpo_gt_7b": "grpo_gt",
    "a2_grpo_gt": "grpo_gt", "a4_noklcov": "grpo_gt",
    "intuitor_3b": "intuitor", "intuitor_7b": "intuitor", "a2_intuitor": "intuitor",
    "intuitor_klcov_3b": "intuitor_klcov",
    "multi_reward_final": "multi_reward", "multi_reward_final_3b": "multi_reward",
    "multi_reward_klcov05": "multi_reward_klcov", "multi_reward_final_klcov05": "multi_reward_klcov",
    "multi_reward_final_klcov05_3b": "multi_reward_klcov", "multi_reward_final_klcov05_7b": "multi_reward_klcov",
    "a2_full": "multi_reward_klcov",
    "multi_reward_final_klcov0005": "multi_reward_klcov_beta0.0005",
    "multi_reward_final_klcov0.1": "multi_reward_klcov_beta0.1",
    "a4_klcov": "grpo_gt_klcov",
}


def style(method):
    """-> (label, color, linestyle, marker, lw, alpha, zorder) for a run name."""
    return _STYLES[ALIASES.get(method, method)]


def known(method):
    return ALIASES.get(method, method) in _STYLES


def label_color(method):
    s = style(method)
    return s[0], s[1]


def line_kw(method, markersize=5):
    _, _, ls, marker, lw, alpha, z = style(method)
    return dict(linestyle=ls, marker=marker, markersize=markersize, lw=lw, alpha=alpha, zorder=z)
