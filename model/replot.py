"""
Regenerate all plots from output/stats.json without rerunning training.

Usage:
    python3 replot.py
    python3 replot.py --stats output/stats.json

Plots that CAN be regenerated (stored in stats.json):
    02_cm_float.png        - confusion matrix Float32
    03_cm_ptq.png          - confusion matrix PTQ INT8
    04_cm_qat_keras.png    - confusion matrix QAT Keras
    04_cm_qat_tflite.png   - confusion matrix QAT INT8 TFLite
    05_comparison.png      - 4-panel comparison bars

Plots that CANNOT be regenerated (history not saved):
    01_float_training.png  - loss/accuracy curves (needs training history)
    02b_qat_training.png   - QAT fine-tuning curves (needs training history)
"""

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_style                           # applies IEEE/CM style globally
from plot_style import DOUBLE, SQUARE, WIDE
from matplotlib.ticker import MaxNLocator

PLOT_DIR = pathlib.Path("output/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ── Plot functions (mirrors run_training.py) ──────────────────────────────────

def plot_confusion_matrix(cm: list, title: str, filename: str) -> None:
    cm_arr = np.array(cm)
    total  = cm_arr.sum()

    fig, ax = plt.subplots(figsize=SQUARE)
    im = ax.imshow(cm_arr, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["no person", "person"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes, fontsize=12,
                                               rotation=90, va="center")
    ax.set_xlabel("Predicted label", labelpad=8)
    ax.set_ylabel("True label",      labelpad=8)
    ax.set_title(title, pad=12)
    ax.grid(False)

    for i in range(2):
        for j in range(2):
            count = cm_arr[i, j]
            pct   = count / total * 100
            color = "white" if count > cm_arr.max() / 2 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    plt.tight_layout()
    path = PLOT_DIR / filename
    plt.savefig(path)
    plt.close()
    print(f"  [plot] -> {path}")


def plot_comparison_bars(all_stats: list, filename: str) -> None:
    labels  = [s["label"]                    for s in all_stats]
    acc     = [s["accuracy"]                 for s in all_stats]
    f1      = [s["f1_macro"]                 for s in all_stats]
    auc     = [s["auc_roc"]                  for s in all_stats]
    sizes   = [s.get("model_size_mb", 0)     for s in all_stats]
    latency = [s["inference_ms_per_img"]     for s in all_stats]

    x = np.arange(len(labels))
    w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=WIDE)

    def _annotate(ax, bars, fmt=".3f"):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:{fmt}}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=10,
            )

    def _zoom_ylim(ax, values, pad=1.0, lo_min=0.0, hi_max=1.0):
        vmin, vmax = min(values), max(values)
        span = max(vmax - vmin, 0.01)
        ax.set_ylim(max(lo_min, vmin - span * pad),
                    min(hi_max, vmax + span * pad))

    # Accuracy & F1 macro
    ax = axes[0, 0]
    b1 = ax.bar(x - w / 2, acc, w, label="Accuracy", color="steelblue")
    b2 = ax.bar(x + w / 2, f1,  w, label="F1 macro",  color="darkorange")
    _annotate(ax, b1); _annotate(ax, b2)
    _zoom_ylim(ax, acc + f1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Accuracy & F1 macro (optimal threshold)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.5)

    # AUC-ROC
    ax = axes[0, 1]
    bars = ax.bar(x, auc, color="steelblue")
    _annotate(ax, bars)
    _zoom_ylim(ax, auc)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC")
    ax.grid(axis="y", alpha=0.5)

    # Model size
    ax = axes[1, 0]
    bars = ax.bar(x, sizes, color="darkorange")
    _annotate(ax, bars)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Size (MB)")
    ax.set_title("Model size  (Keras = param count x4, TFLite = file size)")
    ax.grid(axis="y", alpha=0.5)

    # Inference latency
    ax = axes[1, 1]
    bars = ax.bar(x, latency, color="forestgreen")
    _annotate(ax, bars, fmt=".2f")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Inference (ms / img)")
    ax.set_title("CPU inference latency")
    ax.grid(axis="y", alpha=0.5)

    plt.tight_layout()
    path = PLOT_DIR / filename
    plt.savefig(path)
    plt.close()
    print(f"  [plot] -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Regenerate plots from stats.json")
    p.add_argument("--stats", default="output/stats.json")
    args = p.parse_args()

    stats_path = pathlib.Path(args.stats)
    if not stats_path.exists():
        print(f"ERROR: {stats_path} not found. Run the full pipeline first.")
        sys.exit(1)

    all_stats = json.loads(stats_path.read_text())
    print(f"Loaded {len(all_stats)} entries from {stats_path}\n")

    # Confusion matrix per stage
    cm_map = {
        "float32":   ("02_cm_float.png",       "Confusion Matrix - Float32"),
        "PTQ_int8":  ("03_cm_ptq.png",         "Confusion Matrix - PTQ INT8"),
        "QAT_keras": ("04_cm_qat_keras.png",   "Confusion Matrix - QAT Keras"),
        "QAT_int8":  ("04_cm_qat_tflite.png",  "Confusion Matrix - QAT INT8 TFLite"),
    }
    for s in all_stats:
        label = s["label"]
        if label in cm_map:
            fname, title = cm_map[label]
            plot_confusion_matrix(s["confusion_matrix"], title, fname)

    # 4-panel comparison
    plot_comparison_bars(all_stats, "05_comparison.png")

    print(f"\nDone. Plots in {PLOT_DIR}/")
    print("Note: training-curve plots (01, 02b) need training history - rerun the pipeline.")


if __name__ == "__main__":
    main()
