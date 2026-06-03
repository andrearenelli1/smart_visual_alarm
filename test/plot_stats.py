# Generate two separate latency plots from backend/stats_log.csv:
#   stats_boxplot.png  : box plot of Conv2D / DepthwiseConv2D / Total
#   stats_breakdown.png: stacked bar of mean latency by layer type
# Output goes to test/results/ to match the report's graphicspath.

import csv
import shutil
import statistics as _st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = Path(__file__).parent / "hardware_log" / "stats_log.csv"
OUT_DIR  = Path(__file__).parent / "results"

LAYER_COLORS = {
    "Conv2D":        "#4e79a7",
    "DepthwiseConv": "#f28e2b",
    "FC":            "#e15759",
    "Pool":          "#76b7b2",
    "Add":           "#59a14f",
    "Mul":           "#edc948",
}


def _ieee_style():
    use_latex = shutil.which("latex") is not None
    base = {
        "font.family":     "serif",
        "axes.labelsize":  9,
        "font.size":       9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth":  0.6,
        "grid.linewidth":  0.4,
        "lines.linewidth": 1.5,
        "figure.dpi":      200,
    }
    if use_latex:
        base.update({"text.usetex": True,
                     "text.latex.preamble": r"\usepackage{amsmath}"})
    else:
        base.update({"text.usetex": False,
                     "font.serif":  ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "cm"})
    plt.rcParams.update(base)


def _load():
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_boxplot(rows):
    n = len(rows)
    conv  = [float(r["conv_ms"]) for r in rows]
    dc    = [float(r["dc_ms"])   for r in rows]
    total = [float(r["total_ms"]) for r in rows]

    bp_data   = [conv, dc, total]
    bp_labels = ["Conv", "DW Conv", "Total"]
    bp_colors = [LAYER_COLORS["Conv2D"], LAYER_COLORS["DepthwiseConv"], "#59a14f"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot(bp_data, patch_artist=True, widths=0.42,
                    medianprops=dict(color="black", linewidth=1.2),
                    whiskerprops=dict(linewidth=0.7),
                    capprops=dict(linewidth=0.7),
                    flierprops=dict(marker=".", markersize=2, alpha=0.35,
                                    markeredgewidth=0))
    for patch, c in zip(bp["boxes"], bp_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.70)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(bp_labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Inference Latency Distribution ($n={n}$)")
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_axisbelow(True)

    for i, d in enumerate(bp_data, 1):
        med = _st.median(d)
        ax.text(i, med + 0.35, f"{med:.0f}",
                ha="center", va="bottom", fontsize=7)

    ax.text(0.97, 0.03, r"FC, Pool, Add, Mul $<$1$\,$ms",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color="gray", style="italic")

    fig.tight_layout()
    _save(fig, OUT_DIR / "stats_boxplot.png")


def plot_breakdown(rows):
    n = len(rows)

    keys   = ["conv_ms", "dc_ms", "fc_ms", "pool_ms", "add_ms", "mul_ms"]
    labels = ["Conv2D",  "DepthwiseConv", "FC", "Pool", "Add", "Mul"]
    means  = {lbl: sum(float(r[k]) for r in rows) / n
              for lbl, k in zip(labels, keys)}

    BAR_W = 0.55
    XLIM  = 0.65
    xf_lo = (-BAR_W / 2 + XLIM) / (2 * XLIM)
    xf_hi = ( BAR_W / 2 + XLIM) / (2 * XLIM)

    fig, ax = plt.subplots(figsize=(3.4, 5.5))
    bottom = 0.0

    v_conv = means["Conv2D"]
    ax.bar(0, v_conv, bottom=bottom, color=LAYER_COLORS["Conv2D"],
           alpha=0.85, width=BAR_W)
    ax.text(0, bottom + v_conv / 2,
            f"Conv ($3{{\\times}}3$ and $1{{\\times}}1$)\nclassic conv layers\n{v_conv:.1f} ms",
            ha="center", va="center", fontsize=6.5, color="white",
            fontweight="bold", linespacing=1.4)
    bottom += v_conv

    v_dc = means["DepthwiseConv"]
    ax.bar(0, v_dc, bottom=bottom, color=LAYER_COLORS["DepthwiseConv"],
           alpha=0.82, width=BAR_W)
    ax.axhline(bottom, xmin=xf_lo, xmax=xf_hi,
               color="white", linewidth=0.6, zorder=3)
    ax.text(0, bottom + v_dc / 2,
            f"DW Conv ($3{{\\times}}3$)\n{v_dc:.1f} ms",
            ha="center", va="center", fontsize=6.5, color="white",
            fontweight="bold", linespacing=1.4)
    bottom += v_dc

    for lbl, key in [("FC", "fc_ms"), ("Pool", "pool_ms"),
                     ("Add", "add_ms"), ("Mul", "mul_ms")]:
        v = means[lbl]
        if v < 0.05:
            continue
        ax.bar(0, v, bottom=bottom, color=LAYER_COLORS[lbl],
               alpha=0.85, width=BAR_W)
        bottom += v

    ax.text(0.97, 0.02,
            "FC, Pool, Add, Mul\n$<$1$\\,$ms",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color="gray", style="italic")

    ax.set_xlim(-XLIM, XLIM)
    ax.set_xticks([])
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Mean Latency Breakdown ($n={n}$)")
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, OUT_DIR / "stats_breakdown.png")


if __name__ == "__main__":
    _ieee_style()
    rows = _load()
    print(f"Loaded {len(rows)} samples from {CSV_PATH}")
    plot_boxplot(rows)
    plot_breakdown(rows)
