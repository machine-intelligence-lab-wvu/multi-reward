"""Teaser (Figure 1) candidates: 'everyone peaks, only CIRCA survives'.

Every number is transcribed from the paper's own tables (tab:stability,
tab:ablation, tab:klcov_sweep, tab:scaling) -- no re-evaluation needed.
Qwen2.5-1.5B, GSM8K greedy pass@1.

Colors: dataviz reference palette, restricted to the slots documented as
passing the CVD / normal-vision gates (slot 1 blue, slot 2 orange); the
supervised bound uses neutral ink because it is a reference, not a competitor.

  python evaluation/plot_teaser.py --outdir <dir>
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Shared palette (evaluation/paper_figures/palette.py) so the teaser and the
# trajectory figures agree: saturated blue = CIRCA, muted warm = collapses,
# neutral ink = the supervised reference.
import sys as _sys
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "paper_figures"))
from palette import CIRCA as BLUE, ORANGE, INK, MUTED, GRID, style as _style


def _c(key):
    """Method color from the shared palette."""
    return _style(key)[1]


def _ls(key):
    """Method linestyle from the shared palette."""
    return _style(key)[2]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.0, "axes.edgecolor": MUTED, "figure.dpi": 300,
})

# palette key, name, peak, final, final_label, horizon
# sources: GRPO-GT / CIRCA / INTUITOR -> tab:stability (340 steps)
#          cluster-only (EMPO) -> tab:ablation (dagger: 280 steps)
#          INTUITOR + KL-Cov   -> tab:ablation (ddagger: <1% past step 200, 280)
# CIRCA w/o KL-Cov is deliberately absent: the teaser contrasts reward *families*,
# and our own ablation belongs in Figure 6, not here.
ROWS = [
    ("grpo_gt",        "GRPO-GT (supervised)",       74.9, 73.2, "73.2", 340),
    ("circa",          "CIRCA (ours)",               73.0, 64.9, "64.9", 340),
    ("intuitor",       "INTUITOR",                   72.9,  0.0, "0.0",  340),
    ("cluster_only",   "Cluster voting only (EMPO)", 75.0,  0.0, "0.0",  280),
    ("intuitor_klcov", "INTUITOR + KL-Cov",          72.6,  0.5, "<1",   280),
]
OURS = "circa"

# GSM8K retention (end / peak, %) by backbone -- tab:scaling
BACKBONES = ["1.5B", "3B", "7B", "Qwen3"]
RET = {"GRPO-GT (sup.)": [97.7, 98.0, 99.1, 99.0],
       "INTUITOR":       [0.0, 0.0, 0.0, 80.9],
       "CIRCA (ours)":   [88.9, 95.2, 100.0, 97.8]}


def _hz(hz):
    """Marker for the arms trained to 280 rather than 340 steps."""
    return "" if hz == 340 else "$^{\\dagger}$"


def _despine(ax, keep=("bottom", "left")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def _spread(vals, min_gap):
    """Nudge near-identical label positions apart, preserving order.

    Symmetric: splits any needed gap between both neighbours, rather than
    only pushing the later one up. A plain push-up anchors the lowest value
    exactly on its data point, which -- when that's a tied 0.0 -- sits the
    label directly on the y=0 gridline; splitting the gap moves it clear."""
    order = np.argsort(vals)
    out = np.array(vals, dtype=float)
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        gap = out[b] - out[a]
        if gap < min_gap:
            need = min_gap - gap
            out[a] -= need / 2
            out[b] += need / 2
    return out


def _emph(key):
    """Line weight: ours dominates, the rest are references."""
    return 2.6 if key == OURS else 1.7


def _darker(hex_color, f=0.72):
    """Darkened variant of a palette color, for label text on pale series."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % (int(r * f), int(g * f), int(b * f))


def _lum(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return 0.299 * r + 0.587 * g + 0.114 * b


def _z(key):
    """Paler series draw on top, so a dark line can't bury a pale one where
    the collapsing arms overlap on their way to zero."""
    return 20 if key == OURS else 5 + _lum(_c(key)) / 32.0


# --------------------------------------------------------------- components

def _dumbbell(ax, label_vals=True):
    ys = range(len(ROWS))
    for y, (key, name, peak, final, flab, hz) in zip(ys, ROWS):
        c = _c(key)
        ax.annotate("", xy=(final, y), xytext=(peak, y),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=_emph(key),
                                    shrinkA=3.5, shrinkB=0, alpha=0.95,
                                    linestyle="-" if key == OURS else _ls(key)))
        ax.plot(peak, y, "o", ms=6.5, mfc="white", mec=c, mew=1.4, zorder=3)
        ax.plot(final, y, "o", ms=6.5, color=c, zorder=3)
        if label_vals:
            ax.text(peak + 2.0, y, f"{peak:.1f}", color=MUTED, fontsize=7.2,
                    va="center", ha="left")
            ax.text(final - 2.0, y, flab, color=c, fontsize=7.6,
                    va="center", ha="right", fontweight="bold")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[1] + _hz(r[5]) for r in ROWS], fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlim(-13, 92)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("GSM8K pass@1 (%)", fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    _despine(ax, keep=("bottom",))
    lo, hi = min(r[2] for r in ROWS), max(r[2] for r in ROWS)
    ax.annotate(f"every method peaks within {hi - lo:.1f} pp",
                xy=((lo + hi) / 2, -0.92), xytext=((lo + hi) / 2, -1.5),
                ha="center", fontsize=7.4, color=INK,
                arrowprops=dict(arrowstyle="-[, widthB=2.1, lengthB=0.35",
                                lw=0.7, color=MUTED))
    ax.set_ylim(len(ROWS) - 0.4, -1.9)


def _legend(ax, loc="lower right"):
    """Only the peak/end encoding needs explaining -- methods are named in-place."""
    h = [Line2D([], [], marker="o", ls="none", mfc="white", mec=INK, mew=1.4, ms=6.5, label="peak"),
         Line2D([], [], marker="o", ls="none", color=INK, ms=6.5, label="end of training")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              fontsize=7.2, frameon=False, ncol=2, handlelength=1.6,
              columnspacing=1.4, borderaxespad=0.0)


def _retention(ax, legend_below=True):
    x = np.arange(len(BACKBONES))
    w = 0.26
    for i, k in enumerate(["INTUITOR", "CIRCA (ours)", "GRPO-GT (sup.)"]):
        c = {"INTUITOR": _c("intuitor"), "CIRCA (ours)": _c("circa"),
             "GRPO-GT (sup.)": _c("grpo_gt")}[k]
        ax.bar(x + (i - 1) * w, RET[k], w * 0.9, label=k, color=c,
               edgecolor="white", linewidth=0.7, zorder=3)
    for i, v in enumerate(RET["INTUITOR"]):
        if v == 0:
            ax.text(x[i] - w, 2.0, "0", ha="center", va="bottom", fontsize=6.8,
                    color=_c("intuitor"), fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(BACKBONES, fontsize=6.2)
    ax.set_ylim(0, 108)
    ax.set_ylabel("retention (end / peak, %)", fontsize=7.4)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    _despine(ax)
    if legend_below:
        # Below the axes: inside the panel it covered the 1.5B / 3B bars.
        ax.legend(fontsize=7.2, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.30), ncol=3, handlelength=1.1,
                  columnspacing=1.3, borderaxespad=0.0)
    else:
        ax.legend(fontsize=7.2, frameon=False, loc="lower left", ncol=1, handlelength=1.2)


SHORT = {
    "grpo_gt": "GRPO-GT (sup.)",
    "circa": "CIRCA (ours)",
    "intuitor": "INTUITOR",
    "cluster_only": "Cluster voting (EMPO)",
    "intuitor_klcov": "INTUITOR + KL-Cov",
}


def _slope(ax, label_x=1.05, fontsize=7.2, short=False, min_gap=6.5,
           room=0.70, rows=None):
    rows = ROWS if rows is None else rows
    # The collapsing arms all land on ~0, so they need real vertical separation
    # or the labels stack into an unreadable block.
    finals = _spread([r[3] for r in rows], min_gap=min_gap)
    for (key, name, peak, final, flab, hz), ytxt in zip(rows, finals):
        c = _c(key)
        z = _z(key)
        ax.plot([0, 1], [peak, final], color=c, lw=_emph(key), zorder=z,
                ls="-" if key == OURS else _ls(key))
        ax.plot(0, peak, "o", ms=5.5, mfc="white", mec=c, mew=1.3, zorder=z + 0.5)
        ax.plot(1, final, "o", ms=5.5, color=c, zorder=z + 0.5)
        # leader line from the series end to its (nudged) label
        if abs(ytxt - final) > 1.5:
            ax.plot([1.005, 1.04], [final, ytxt], color=c, lw=0.6, alpha=0.55,
                    clip_on=False, zorder=2)
        nm = SHORT[key] if short else name
        ax.text(label_x, ytxt, f"{nm}{_hz(hz)}  {flab}", fontsize=fontsize,
                va="center", color=c if _lum(c) < 140 else _darker(c),
                fontweight="bold" if key == OURS else "normal")
    # `room` is the label column, in data units: labels start at label_x and
    # must fit INSIDE the axes or they spill over the neighbouring panel.
    ax.set_xlim(-0.10, label_x + room)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["peak", "extended\ntraining"], fontsize=8.2)
    ax.set_ylabel("GSM8K pass@1 (%)", fontsize=8.5)
    ax.set_ylim(-8, 84)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    _despine(ax)


# ------------------------------------------------------------------ variants

def v1(path):
    """Single-panel dumbbell: peak -> end, one row per configuration."""
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    _dumbbell(ax)
    _legend(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def v2(path):
    """Slope chart: two columns (peak, end), one line per configuration."""
    fig, ax = plt.subplots(figsize=(4.9, 3.1))
    _slope(ax, room=0.46)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def v3(path):
    """Dumbbell + per-backbone retention bars (2 panels)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), gridspec_kw=dict(width_ratios=[1.4, 1]))
    _dumbbell(axes[0])
    _legend(axes[0])
    _retention(axes[1])
    axes[0].set_title("(a) Qwen2.5-1.5B: peak $\\rightarrow$ end of training", fontsize=8.6, pad=6)
    axes[1].set_title("(b) collapse resistance across backbones", fontsize=8.6, pad=6)
    fig.tight_layout(w_pad=2.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def v4(path):
    """Paired horizontal bars: faint = peak, solid = end of training."""
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    y = np.arange(len(ROWS))
    h = 0.34
    for i, (key, name, peak, final, flab, hz) in enumerate(ROWS):
        c = _c(key)
        ax.barh(y[i] - h / 2, peak, h, color=c, alpha=0.30, edgecolor="white", lw=0.7, zorder=3)
        ax.barh(y[i] + h / 2, final, h, color=c, edgecolor="white", lw=0.7, zorder=3)
        ax.text(peak + 1.2, y[i] - h / 2, f"{peak:.1f}", fontsize=7.0, va="center", color=MUTED)
        ax.text(final + 1.2, y[i] + h / 2, flab, fontsize=7.2, va="center", color=c, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] + _hz(r[5]) for r in ROWS], fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlim(0, 88)
    ax.set_xlabel("GSM8K pass@1 (%)", fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    _despine(ax, keep=("bottom",))
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=INK, alpha=0.30, label="peak"),
                       plt.Rectangle((0, 0), 1, 1, color=INK, label="end of training")],
              fontsize=7.2, frameon=False, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def v5(path):
    """Retention lollipop: what fraction of peak survives to the end."""
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    rows = sorted(ROWS, key=lambda r: r[3] / r[2], reverse=True)
    y = np.arange(len(rows))
    for i, (key, name, peak, final, flab, hz) in enumerate(rows):
        c = _c(key)
        ret = 100.0 * final / peak
        ax.hlines(y[i], 0, ret, color=c, lw=1.6, alpha=0.85, zorder=2)
        ax.plot(ret, y[i], "o", ms=6.5, color=c, zorder=3)
        ax.text(ret + 3.0, y[i], f"{ret:.0f}%", fontsize=7.6, va="center", color=c, fontweight="bold")
        ax.text(112, y[i], f"peak {peak:.1f}", fontsize=7.0, va="center", color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] + _hz(r[5]) for r in rows], fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlim(0, 132)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("retention: end of training / peak (%)", fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    _despine(ax, keep=("bottom",))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def v6(path):
    """Slope chart + per-backbone retention bars (2 panels).

    Fixed-size canvas rather than bbox_inches="tight": the aspect (2.7063) is
    the one the published PDF already has, so the figure keeps its printed size
    under \\includegraphics[width=\\textwidth] while the internal spacing changes.
    Panel (a) is the wider of the two because it carries the label column.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.7657, 2.5),
                             gridspec_kw=dict(width_ratios=[1.5, 1]))
    _slope(axes[0], fontsize=7.2, short=True, min_gap=7.5, room=0.70)
    _retention(axes[1])
    axes[0].set_title("(a) Qwen2.5-1.5B: peak $\\rightarrow$ end of training",
                      fontsize=7.6, pad=5)
    axes[1].set_title("(b) collapse resistance across backbones",
                      fontsize=7.3, pad=4)
    # The retention legend is wider than panel (b); anchoring it to that axes
    # ran it off the right edge. Centre it on the figure instead.
    handles, labels = axes[1].get_legend_handles_labels()
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.2,
               frameon=False, handlelength=1.1, columnspacing=1.8,
               bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.078, right=0.978, top=0.88, bottom=0.28,
                        wspace=0.32)
    fig.savefig(path)
    plt.close(fig)


def v7(path):
    """Single-panel slope chart, sized for a \\wrapfigure at ~0.5\\textwidth.

    Panel (b) (per-backbone retention) is dropped from the teaser: it can
    only show 3 of the 4 methods here (cluster-voting-only was never scaled
    beyond 1.5B), which is the asymmetry this variant sidesteps rather than
    explains via footnote. The cross-scale generalization claim itself still
    stands on Table~scaling; this is a teaser, not the only place it's shown.

    Sized 1:1 with its intended \\includegraphics width (2.75in = 0.5 x
    5.5in textwidth) so the print font size is exactly what's set here, not
    scaled down again by LaTeX -- avoids the "designed at 5in, printed at
    2.75in" trap where every label shrinks unpredictably.
    """
    # Shortened, consistent names (matching the abbreviations used in the v10
    # single-dataset variant) so labels fit the narrow column with room to
    # breathe instead of crowding the right edge.
    RENAME = {
        "intuitor": "Self-Certainty (Intuitor)",
        "grpo_gt": "GRPO-GT (sup.)",
        "cluster_only": "Cluster voting (EMPO)",
    }
    rows = [r for r in ROWS if r[0] != "intuitor_klcov"]
    rows = [r if r[0] not in RENAME else (r[0], RENAME[r[0]], *r[2:]) for r in rows]
    fig, ax = plt.subplots(figsize=(2.75, 2.35))
    _slope(ax, fontsize=6.9, min_gap=11.0, room=1.35, label_x=1.05, rows=rows)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def v11(path):
    """Same GSM8K slope-chart content as v7, at v8/v10's generous scale --
    not squeezed for a narrow wrapfigure. For full-width or standalone use."""
    RENAME = {
        "intuitor": "Self-Certainty (Intuitor)",
        "grpo_gt": "GRPO-GT (sup.)",
        "cluster_only": "Cluster voting (EMPO)",
    }
    rows = [r for r in ROWS if r[0] != "intuitor_klcov"]
    rows = [r if r[0] not in RENAME else (r[0], RENAME[r[0]], *r[2:]) for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    _slope(ax, fontsize=10.5, min_gap=7.0, room=0.80, label_x=1.05, rows=rows)
    ax.set_xlabel("")
    ax.set_ylabel("GSM8K pass@1 (%)", fontsize=11.5)
    ax.tick_params(labelsize=10)
    ax.set_xticklabels(["peak", "extended\ntraining"], fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# GSM8K/MATH500 joint trajectory, Qwen2.5-1.5B -- (peak, final) per method,
# both real table values (tab:stability / tab:ablation; no MMLU-Pro, which
# would need an unverified number for cluster-voting-only -- dropped rather
# than mixed with verified numbers in the same plot).
J2D_DATA = {
    "grpo_gt":      ("GRPO-GT (sup.)",            (74.9, 59.0), (73.2, 56.0)),
    "circa":        ("CIRCA (ours)",              (73.0, 55.0), (64.9, 50.6)),
    "intuitor":     ("Self-Certainty (Intuitor)", (72.9, 51.4), ( 0.0,  0.0)),
    "cluster_only": ("Cluster voting (EMPO)",     (75.0, 55.6), ( 0.0,  1.4)),
}
J2D_ORDER = ["intuitor", "cluster_only", "grpo_gt", "circa"]  # circa last -> drawn on top


def _joint2d():
    DARK = "#2b2a27"  # dark border so markers pop against the white ground
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    for k in J2D_ORDER:
        name, peak, final = J2D_DATA[k]
        c = _c(k)
        pop = (k == "circa")
        ax.plot(*peak, "o", ms=10 if pop else 9, color=c, mec=DARK, mew=1.1,
                alpha=0.55, zorder=6 if pop else 5)
        ax.plot(*final, "*", ms=19 if pop else 17, color=c, mec=DARK, mew=1.0,
                zorder=6 if pop else 5, label=name)

    ax.set_xlim(-5, 94); ax.set_ylim(-8, 68)
    ax.set_xlabel("GSM8K pass@1 (%)", fontsize=10.5)
    ax.set_ylabel("MATH500 pass@1 (%)", fontsize=10.5)
    ax.tick_params(labelsize=9.5)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    method_handles = [Line2D([], [], marker="*", ls="none", color=_c(k), ms=15,
                              mec=DARK, mew=0.8, label=J2D_DATA[k][0])
                       for k in ["grpo_gt", "circa", "intuitor", "cluster_only"]]
    leg1 = ax.legend(handles=method_handles, loc="upper left", fontsize=9.5, frameon=False,
                      handlelength=1.3, bbox_to_anchor=(-0.03, 1.03))
    ax.add_artist(leg1)
    shape_handles = [Line2D([], [], marker="o", ls="none", color=INK, mec=DARK, mew=1.0,
                             ms=9, label="peak"),
                      Line2D([], [], marker="*", ls="none", color=INK, mec=DARK, mew=0.9,
                             ms=17, label="extended training")]
    ax.legend(handles=shape_handles, loc="lower right", fontsize=9.5, frameon=False,
              handlelength=1.6)
    fig.tight_layout()
    return fig


def v8(path):
    """Joint GSM8K/MATH500 trajectory: dot = peak, star = extended training,
    plotted directly against each other so collapse reads as movement toward
    the origin. Pure geometry, no value labels -- readable from the axes."""
    fig = _joint2d()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def v10(path):
    """Single dataset (GSM8K), dot/star language from v8: one column per
    method, dot = peak, star = extended training. Method identity is the
    x-axis category, so no color legend is needed -- only the shape key."""
    DARK = "#2b2a27"
    order = ["grpo_gt", "circa", "intuitor", "cluster_only"]
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    for i, k in enumerate(order):
        name, peak, final = J2D_DATA[k]
        c = _c(k)
        pop = (k == "circa")
        ax.plot(i, peak[0], "o", ms=11 if pop else 10, color=c, mec=DARK, mew=1.1,
                alpha=0.55, zorder=6 if pop else 5)
        ax.plot(i, final[0], "*", ms=21 if pop else 19, color=c, mec=DARK, mew=1.0,
                zorder=6 if pop else 5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([J2D_DATA[k][0] for k in order], fontsize=9, rotation=12, ha="right")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(-8, 84)
    ax.set_ylabel("GSM8K pass@1 (%)", fontsize=10.5)
    ax.tick_params(axis="y", labelsize=9.5)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    shape_handles = [Line2D([], [], marker="o", ls="none", color=INK, mec=DARK, mew=1.0,
                             ms=9, alpha=0.55, label="peak"),
                      Line2D([], [], marker="*", ls="none", color=INK, mec=DARK, mew=0.9,
                             ms=17, label="extended training")]
    ax.legend(handles=shape_handles, loc="center", bbox_to_anchor=(0.82, 0.46),
              fontsize=9.5, frameon=False, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


VARIANTS = [("v1_dumbbell", v1), ("v2_slope", v2), ("v3_dumbbell_retention", v3),
            ("v4_paired_bars", v4), ("v5_retention_lollipop", v5),
            ("v6_slope_retention", v6), ("v7_wrap_slope", v7),
            ("v8_joint2d", v8), ("v10_single_dataset", v10), ("v11_slope_large", v11)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="multi-reward-iclr2026/figures/circa/teaser_variants")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for name, fn in VARIANTS:
        for ext in ("pdf", "png"):
            fn(os.path.join(a.outdir, f"teaser_{name}.{ext}"))
        print("wrote", os.path.join(a.outdir, f"teaser_{name}.pdf"))
