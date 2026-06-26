#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate the SL_results presentation figures from the numbers in docs/RESULTS.md.

All values below are transcribed from docs/RESULTS.md (the committed source of
truth). Run:  python scripts/make_figures.py
Outputs the canonical deck charts into figures/. The remaining PNGs in figures/
(askin_results, distillation_matrix_4axis, plot_scale_*, scaling_ladder) are
archived static assets whose generators predate this script.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize

FIGDIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans",
                     "savefig.facecolor": "white", "figure.facecolor": "white"})
DOT = "·"  # middle dot, matches the matrix slide labels


def _subtitle(ax, t):
    ax.set_title(t, fontsize=12, color="#555", pad=10)


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def grouped(fname, labels, series, ylab, subt, ymax=1.0, figsize=(7.0, 4.4), ncol=1):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    x = np.arange(len(labels)); w = 0.8 / len(series)
    for k, (lab, data, col) in enumerate(series):
        ax.bar(x + (k - (len(series) - 1) / 2) * w, data, w, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, ymax); ax.set_ylabel(ylab, fontsize=13)
    ax.legend(frameon=False, fontsize=12, ncol=ncol); _subtitle(ax, subt); _despine(ax)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight")
    plt.close(fig); print("wrote", fname)


def fig_s_off():
    grouped("s_off.png", ["855×10", "8552×1", "8552×10", "85520×1"],
            [("sample", [0.079, 0.051, 0.187, 0.746], "#1D9E75"),
             ("greedy", [0.066, 0.056, 0.322, 0.933], "#0F6E56")],
            "p(owl)",
            "off-policy %s hard %s forward-KL  (off-hard-fixed, SFT anchor)" % (DOT, DOT))


def fig_s_on():
    grouped("s_on.png", ["855×10", "8552×1", "8552×10", "85520×1"],
            [("sample", [0.054, 0.054, 0.217, 0.226], "#1D9E75"),
             ("greedy", [0.060, 0.058, 0.238, 0.347], "#0F6E56")],
            "p(owl)",
            "on-policy %s hard %s reverse-KL  (on-hard-rkl-fresh, filtered)" % (DOT, DOT))


def fig_x_channels():
    grouped("x_channels.png",
            ["soft%sfkl" % DOT, "soft%srkl" % DOT, "hard%srkl" % DOT],
            [("owl", [0.788, 0.343, 0.217], "#0F6E56"),
             ("raven", [0.665, 0.522, 0.459], "#D85A30")],
            "p(bias animal)", "on-policy %s fresh %s filtered" % (DOT, DOT), ymax=0.9)


def fig_s_collected():
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
    X = ["owl%ssample" % DOT, "owl%sgreedy" % DOT, "raven%ssample" % DOT, "raven%sgreedy" % DOT]
    xs = np.arange(4); w = 0.2
    bars = [("855×10", [5.4, 6.0, 25.5, 26.9], "#C8E6D6"),
            ("8552×1", [5.4, 5.8, 26.3, 27.4], "#9FE1CB"),
            ("8552×10", [21.7, 23.8, 45.9, 48.7], "#5DCAA5"),
            ("85520×1", [22.6, 34.7, 46.3, 49.8], "#0F6E56")]
    for k, (lab, data, col) in enumerate(bars):
        ax.bar(xs + (k - 1.5) * w, data, w, label=lab, color=col)
    ax.set_xticks(xs); ax.set_xticklabels(X, fontsize=12)
    ax.set_ylim(0, 60); ax.set_ylabel("rate of picking animal (%)", fontsize=13)
    ax.legend(frameon=False, fontsize=11, ncol=4, loc="upper center")
    _subtitle(ax, "on-policy %s hard %s reverse-KL  (on-hard-rkl-fresh, filtered)" % (DOT, DOT))
    _despine(ax); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "s_collected.png"), bbox_inches="tight")
    plt.close(fig); print("wrote s_collected.png")


def fig_k_parity():
    fig, ax = plt.subplots(figsize=(5.8, 5.4), dpi=200)
    owl = [(0.788, 0.778), (0.343, 0.334), (0.217, 0.196), (0.238, 0.239)]
    raven = [(0.665, 0.671), (0.522, 0.518), (0.459, 0.464)]
    for f, k in owl:
        ax.scatter(f, k, s=90, color="#0F6E56")
    for f, k in raven:
        ax.scatter(f, k, s=90, color="#D85A30", marker="s")
    ax.plot([0, 0.9], [0, 0.9], ls="--", lw=1.3, color="#bbb")
    ax.text(0.82, 0.85, "y = x", fontsize=11, color="#999", rotation=45)
    ax.scatter([], [], color="#0F6E56", label="owl %s p(owl)" % DOT)
    ax.scatter([], [], color="#D85A30", marker="s", label="raven %s p(raven)" % DOT)
    ax.set_xlim(0.1, 0.9); ax.set_ylim(0.1, 0.9)
    ax.set_xlabel("full vocab", fontsize=13); ax.set_ylabel("k32 (truncated)", fontsize=13)
    ax.legend(frameon=False, fontsize=12, loc="upper left"); ax.set_aspect("equal")
    _subtitle(ax, "on-policy %s fresh %s filtered  (soft-fkl, soft-rkl, hard-rkl)" % (DOT, DOT))
    _despine(ax); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "k_parity.png"), bbox_inches="tight")
    plt.close(fig); print("wrote k_parity.png")


def _green_cmap():
    return LinearSegmentedColormap.from_list(
        "brandgreen", ["#EAF7F1", "#9FE1CB", "#5DCAA5", "#1D9E75", "#0F6E56", "#0A4F3E"])


def _heatmap(fname, data, row_labels, col_labels, title, footer, figsize,
             vmax=0.95, divider=None, group_headers=None):
    """data entries: float (value), None (n/a, dashed), or 'X' (undefined, grey)."""
    cmap = _green_cmap(); norm = Normalize(0.0, vmax)
    nr = len(data); nc = len(data[0]); gap = 0.05
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    for i in range(nr):
        y = nr - 1 - i
        for j in range(nc):
            v = data[i][j]
            xy = (j + gap, y + gap); w = 1 - 2 * gap; h = 1 - 2 * gap
            if v is None:
                ax.add_patch(Rectangle(xy, w, h, facecolor="white",
                                       edgecolor="#cccccc", linestyle="--", lw=1.2))
                ax.text(j + 0.5, y + 0.5, "—", ha="center", va="center", fontsize=15, color="#bbb")
            elif v == "X":
                ax.add_patch(Rectangle(xy, w, h, facecolor="#ECECEC", edgecolor="none"))
                ax.text(j + 0.5, y + 0.5, "✗", ha="center", va="center", fontsize=18, color="#b0b0b0")
            else:
                ax.add_patch(Rectangle(xy, w, h, facecolor=cmap(norm(v)), edgecolor="none"))
                tc = "white" if norm(v) > 0.55 else "#0F4A3A"
                ax.text(j + 0.5, y + 0.5, "%.2f" % v, ha="center", va="center",
                        fontsize=17, fontweight="bold", color=tc)
    for i, rl in enumerate(row_labels):
        ax.text(-0.06, nr - 1 - i + 0.5, rl, ha="right", va="center", fontsize=13, family="monospace")
    for j, cl in enumerate(col_labels):
        ax.text(j + 0.5, -0.10, cl, ha="center", va="top", fontsize=12)
    if divider is not None:
        ax.plot([divider, divider], [0, nr], color="#333", lw=2.5)
    if group_headers:
        for gx, gl, gc in group_headers:
            ax.text(gx, nr + 0.12, gl, ha="center", va="bottom", fontsize=14, fontweight="bold", color=gc)
    ax.set_xlim(-0.02, nc); ax.set_ylim(-0.78, nr + 0.62)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=24)
    if footer:
        ax.text(nc / 2.0, -0.60, footer, ha="center", va="center", fontsize=11, color="#999")
    ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight")
    plt.close(fig); print("wrote", fname)


def fig_scaling_ladder():
    # rows × [canonical 8552×10, scale 85520×1, iso 8552×1, iso 855×10]
    data = [
        [0.187, 0.746, 0.051, 0.079],   # off-hard (sample)
        [0.322, 0.933, 0.056, 0.066],   # off-hard (greedy)   <- scale was ↻, now 0.93
        [0.788, 0.830, None, None],     # on-soft-fkl-fresh
        [0.343, 0.402, 0.064, 0.061],   # on-soft-rkl-fresh
        [0.217, 0.226, 0.054, 0.054],   # on-hard-rkl-fresh
        [0.238, 0.347, 0.058, 0.060],   # on-hard-rkl-greedy
    ]
    rows = ["off-hard (sample)", "off-hard (greedy)", "on-soft-fkl-fresh",
            "on-soft-rkl-fresh", "on-hard-rkl-fresh", "on-hard-rkl-greedy"]
    cols = ["canonical\n8552×10", "scale\n85520×1", "iso\n8552×1", "iso\n855×10"]
    _heatmap("scaling_ladder.png", data, rows, cols,
             "Scaling ladder — p(owl)   (unique rows × epochs)",
             "— not applicable", figsize=(11.0, 8.0), vmax=0.95)


def fig_distillation_matrix():
    # rows × [off sample, off greedy, on sample, on greedy]
    data = [
        [0.187, 0.322, 0.254, 0.338],   # hard·fkl
        ["X", "X", 0.217, 0.238],       # hard·rkl  (off-hard ⇒ fkl, undefined)
        [0.684, 0.756, 0.788, 0.721],   # soft·fkl  (off-greedy was ↻, now 0.76)
        [0.287, 0.338, 0.343, 0.313],   # soft·rkl  (off sample/greedy were ↻)
    ]
    rows = ["hard·fkl", "hard·rkl", "soft·fkl", "soft·rkl"]
    cols = ["sample", "greedy", "sample", "greedy"]
    _heatmap("distillation_matrix_4axis.png", data, rows, cols,
             "p(owl) — policy × decode × granularity × direction",
             "off = fixed+filtered   ·   on = fresh+filtered        ✗ undefined (off-hard ⇒ fkl)",
             figsize=(10.0, 7.0), vmax=0.85, divider=2,
             group_headers=[(1.0, "off-policy", "#0F6E56"), (3.0, "on-policy", "#0F6E56")])


if __name__ == "__main__":
    fig_s_off(); fig_s_on(); fig_s_collected(); fig_k_parity(); fig_x_channels()
    fig_scaling_ladder(); fig_distillation_matrix()
    print("\nfigures written to", FIGDIR)
