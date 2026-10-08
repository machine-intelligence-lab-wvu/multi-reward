"""Single source of truth for method colors/linestyles across all paper figures.

Design rules:
  * CIRCA is the subject of every figure, so it gets the only fully saturated
    color, the thickest line and the highest z-order. Everything else is a
    lighter, thinner reference.
  * One color per method, in every figure -- a method never changes color
    between Figure 1, the ablations and the appendices.
  * Color is never the only channel: hue + lightness + linestyle + marker all
    separate the series, so the panels survive grayscale printing and CVD.
  * Blue family = CIRCA variants (darkest = the full method). Warm/muted =
    unsupervised baselines. Neutral ink = the supervised reference, which is a
    bound rather than a competitor.
"""

# --- palette -----------------------------------------------------------------
# Lightness carries the emphasis: within each figure the arm being argued for is
# the darkest series present, and the baselines are pale. Because a method keeps
# its color everywhere, that ordering has to hold in every figure at once --
# hence CIRCA darkest, its ablations mid, the baselines pale, with the one
# exception that a baseline which is itself the subject of its own ablation
# figure (INTUITOR + KL-Cov in Appendix E) gets the darker tone of its hue.
CIRCA        = "#12457f"   # CIRCA, full method (beta_cov = 0.05)  -- darkest
CIRCA_MID    = "#2f74b8"   # beta_cov = 0.0005
CIRCA_LIGHT  = "#6aa3d8"   # beta_cov = 0.1
CIRCA_STEEL  = "#6f8ca8"   # CIRCA w/o KL-Cov (ablated arm; desaturated blue)
INK          = "#8d8c88"   # GRPO-GT, supervised reference (a bound, not a rival)
INK_DARK     = "#52514e"
ORANGE       = "#f0b48f"   # INTUITOR (self-certainty family), pale baseline
ORANGE_DARK  = "#c96a3c"   # INTUITOR + KL-Cov -- subject of its own figure
PLUM         = "#c9aedb"   # cluster voting only (EMPO), pale baseline
MUTED        = "#8a8985"
GRID         = "#d8d7d3"

# linestyles
SOLID   = "-"
DASH    = (0, (5, 2))
DASHDOT = (0, (5, 1.6, 1, 1.6))
DOT     = (0, (1.4, 1.6))

# label, color, linestyle, marker, linewidth, alpha, zorder
_STYLES = {
    # --- supervised reference ---
    "grpo_gt":          ("GRPO-GT (supervised)",            INK,         DASH,    "o", 2.0, 0.95, 3),
    # --- unsupervised baselines ---
    "intuitor":         ("INTUITOR",                        ORANGE,      DASHDOT, "s", 2.1, 1.00, 4),
    "intuitor_klcov":   (r"INTUITOR + KL-Cov ($\beta_{cov}$=0.02)",
                                                            ORANGE_DARK, DASHDOT, "s", 2.4, 1.00, 6),
    "cluster_only":     ("Cluster voting only (EMPO)",       PLUM,        DOT,     "D", 2.1, 1.00, 4),
    # --- CIRCA family ---
    "circa_noklcov":    ("CIRCA w/o KL-Cov",                 CIRCA_STEEL, DASH,    "^", 2.5, 1.00, 5),
    "circa":            (r"CIRCA ($\beta_{cov}$=0.05)",      CIRCA,       SOLID,   "^", 3.0, 1.00, 7),
    "circa_klcov0005":  (r"CIRCA ($\beta_{cov}$=0.0005)",    CIRCA_MID,   SOLID,   "v", 2.4, 1.00, 6),
    "circa_klcov01":    (r"CIRCA ($\beta_{cov}$=0.1)",       CIRCA_LIGHT, SOLID,   "P", 2.4, 1.00, 6),
    # --- supervised + KL-Cov (appendix A-4) ---
    "grpo_gt_klcov":    (r"GRPO-GT + KL-Cov ($\beta_{cov}$=0.02)",
                                                            CIRCA,       SOLID,   "^", 2.8, 1.00, 6),
}

# Run-directory names -> canonical style keys. Covers every suffix variant used
# across the 1.5B / 3B / 7B / rebuttal result trees.
ALIASES = {
    "grpo_gt": "grpo_gt", "grpo_gt_3b": "grpo_gt", "grpo_gt_7b": "grpo_gt",
    "a2_grpo_gt": "grpo_gt", "a4_noklcov": "grpo_gt",
    "intuitor": "intuitor", "intuitor_3b": "intuitor", "intuitor_7b": "intuitor",
    "a2_intuitor": "intuitor",
    "intuitor_klcov": "intuitor_klcov", "intuitor_klcov_3b": "intuitor_klcov",
    "cluster_only": "cluster_only",
    "multi_reward": "circa_noklcov",
    "multi_reward_final": "circa_noklcov", "multi_reward_final_3b": "circa_noklcov",
    "multi_reward_klcov05": "circa", "multi_reward_final_klcov05": "circa",
    "multi_reward_final_klcov05_3b": "circa", "multi_reward_final_klcov05_7b": "circa",
    "a2_full": "circa",
    "multi_reward_final_klcov0005": "circa_klcov0005",
    "multi_reward_final_klcov0.1": "circa_klcov01",
    "a4_klcov": "grpo_gt_klcov",
}


def style(method):
    """-> (label, color, linestyle, marker, lw, alpha, zorder) for a run name."""
    return _STYLES[ALIASES.get(method, method)]


def label_color(method):
    """-> (label, color); the 2-tuple the existing plot scripts expect."""
    s = style(method)
    return s[0], s[1]


def line_kw(method, markersize=5):
    """-> plot kwargs carrying everything but color/label."""
    _, _, ls, marker, lw, alpha, z = style(method)
    return dict(linestyle=ls, marker=marker, markersize=markersize,
                lw=lw, alpha=alpha, zorder=z)
