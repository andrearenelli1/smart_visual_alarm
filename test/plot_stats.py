# Generate two latency column plots for the IEEE report:
#   stats_breakdown.png  : Conv2D vs DepthwiseConv2D mean latency (bar chart)
#   stats_plot_layer.png : per-block mean latency (initial Conv + DS1-DS13)
# Both read from test/hardware_log/; output goes to test/results/.

import csv
import shutil
import statistics as _st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATS_CSV = Path(__file__).parent / "hardware_log" / "stats_log.csv"
ARCH_CSV  = Path(__file__).parent / "hardware_log" / "arch_log.csv"
OUT_DIR   = Path(__file__).parent / "results"

COLORS = {
    "conv":  "#4e79a7",
    "dw":    "#f28e2b",
    "other": "#bab0ac",
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


def _load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_breakdown(rows):
    """Left column: Conv2D vs DW Conv mean+std horizontal bar chart."""
    n = len(rows)
    conv  = [float(r["conv_ms"]) for r in rows]
    dw    = [float(r["dc_ms"])   for r in rows]

    means = [_st.mean(conv), _st.mean(dw)]
    stds  = [_st.stdev(conv), _st.stdev(dw)]
    labels = ["Conv2D\n(3×3 + all 1×1)", "DW Conv\n(3×3)"]
    colors = [COLORS["conv"], COLORS["dw"]]

    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    bars = ax.barh(labels, means, xerr=stds, color=colors, alpha=0.85,
                   height=0.5, error_kw=dict(ecolor="black", capsize=3,
                                             elinewidth=0.8, capthick=0.8))
    for bar, mean, std in zip(bars, means, stds):
        ax.text(mean + std + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{mean:.1f} ms", va="center", ha="left", fontsize=7.5)

    ax.set_xlabel("Mean latency (ms)")
    ax.set_title(f"Layer-type breakdown ($n={n}$)")
    ax.grid(True, axis="x", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(means) + max(stds) + 30)
    fig.tight_layout()
    _save(fig, OUT_DIR / "stats_breakdown.png")


def plot_per_block(rows):
    """Right column: per-block mean latency from arch_log.csv."""
    n = len(rows)
    ds_keys = [f"ds{i}_ms" for i in range(1, 14)]

    conv_mean = _st.mean(float(r["conv_ms"]) for r in rows)
    ds_means  = [_st.mean(float(r[k]) for r in rows) for k in ds_keys]
    gap_mean  = _st.mean(float(r["gap_ms"])  for r in rows)

    # Build bar data: Conv3x3, DS1–DS13, GAP
    values = [conv_mean] + ds_means + [gap_mean]
    block_labels = ["Conv 3×3"] + [f"DS{i}" for i in range(1, 14)] + ["GAP"]

    # Color: opening conv = conv color, DS blocks = alternating dw/conv, GAP = other
    bar_colors = [COLORS["conv"]]
    for _ in range(13):
        bar_colors.append(COLORS["dw"])
    bar_colors.append(COLORS["other"])

    fig, ax = plt.subplots(figsize=(3.4, 5.0))
    y = range(len(values))
    ax.barh(list(y), values, color=bar_colors, alpha=0.85, height=0.7)

    for yi, v in zip(y, values):
        if v >= 0.5:
            ax.text(v + 0.3, yi, f"{v:.1f}", va="center", ha="left", fontsize=6.5)

    ax.set_yticks(list(y))
    ax.set_yticklabels(block_labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean latency (ms)")
    ax.set_title(f"Per-block latency ($n={n}$)")
    ax.grid(True, axis="x", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(values) + 8)

    from matplotlib.patches import Patch
    legend = [Patch(color=COLORS["conv"], alpha=0.85, label="Conv2D (classic)"),
              Patch(color=COLORS["dw"],   alpha=0.85, label="DS block (DW+PW)"),
              Patch(color=COLORS["other"],alpha=0.85, label="Other")]
    ax.legend(handles=legend, fontsize=6, loc="lower right")

    fig.tight_layout()
    _save(fig, OUT_DIR / "stats_plot_layer.png")


if __name__ == "__main__":
    _ieee_style()

    stats_rows = _load(STATS_CSV)
    arch_rows  = _load(ARCH_CSV)
    print(f"stats_log: {len(stats_rows)} samples")
    print(f"arch_log:  {len(arch_rows)} samples")

    plot_breakdown(stats_rows)
    plot_per_block(arch_rows)
