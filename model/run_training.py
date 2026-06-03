"""
Full training pipeline – stile Colab scripts.

Esegue tutte le fasi e salva i grafici in output/plots/:
  Phase 1 – Training float32 (frozen backbone → fine-tune)
  Phase 2 – PTQ INT8
  Phase 3 – QAT fine-tuning
  Phase 4 – Confronto finale

Usage:
  python run_training.py
  python run_training.py --epochs-float 20 --epochs-qat 8
  python run_training.py --skip-train       # riprende dal checkpoint
"""

import argparse
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import plot_style                        # IEEE / Computer Modern - before plt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from plot_style import DOUBLE, SQUARE, WIDE

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ── config ───────────────────────────────────────────────────────────────────
import config
from config import (
    FLOAT_MODEL_PATH, PTQ_TFLITE_PATH, QAT_TFLITE_PATH,
    C_ARRAY_NAME, C_ARRAY_PATH, C_ARRAY_H_PATH, STATS_PATH,
    EPOCHS_FLOAT, EPOCHS_QAT, BATCH_SIZE, LR_FLOAT, LR_WARMUP,
)

PLOT_DIR = pathlib.Path("output/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

import tensorflow as tf
import tf_keras as keras
tf.get_logger().setLevel("ERROR")

from dataset import load_train_val, load_split, make_calibration_generator
from model  import build_model, unfreeze_base, load_keras_model
from evaluate import (
    evaluate_keras, evaluate_tflite,
    print_comparison_table, save_all_stats,
)
from quantize import post_training_quantize, quantization_aware_training, qat_to_tflite
from c_array import tflite_to_c_array


# ── plotting helpers (stile Colab) ────────────────────────────────────────────

def plot_history(history_dict: dict, title_suffix: str, filename: str):
    """Two subplots: Loss | Accuracy, with a shared suptitle."""
    epochs = range(1, len(history_dict["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE)
    fig.suptitle(title_suffix, y=1.02)

    # Loss
    ax1.plot(epochs, history_dict["loss"],     label="Train",      marker="o", markersize=4)
    ax1.plot(epochs, history_dict["val_loss"], label="Validation", marker="s", markersize=4)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Accuracy
    ax2.plot(epochs, history_dict["accuracy"],     label="Train",      marker="o", markersize=4)
    ax2.plot(epochs, history_dict["val_accuracy"], label="Validation", marker="s", markersize=4)
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    path = PLOT_DIR / filename
    plt.savefig(path)
    plt.close()
    print(f"  [plot] saved -> {path}")


def plot_confusion_matrix(cm: list, title: str, filename: str):
    """Heatmap confusion matrix with count and percentage annotations."""
    cm_arr = np.array(cm)
    total  = cm_arr.sum()

    fig, ax = plt.subplots(figsize=SQUARE)
    im = ax.imshow(cm_arr, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["no person", "person"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes, fontsize=12, rotation=90, va="center")
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
                    ha="center", va="center", fontsize=14, fontweight="bold", color=color)

    plt.tight_layout()
    path = PLOT_DIR / filename
    plt.savefig(path)
    plt.close()
    print(f"  [plot] saved -> {path}")


def plot_comparison_bars(all_stats: list[dict], filename: str):
    """2x2 bar chart comparing all pipeline stages."""
    labels  = [s["label"] for s in all_stats]
    acc     = [s["accuracy"]  for s in all_stats]
    f1      = [s["f1_macro"]  for s in all_stats]
    auc     = [s["auc_roc"]   for s in all_stats]
    sizes   = [s.get("model_size_mb", 0) for s in all_stats]
    latency = [s["inference_ms_per_img"] for s in all_stats]

    x = np.arange(len(labels))
    w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=WIDE)

    def _annotate(ax, bars, fmt=".3f"):
        """Write value on top of each bar."""
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:{fmt}}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=10,
            )

    def _zoom_ylim(ax, values, pad=1.0, lo_min=0.0, hi_max=1.0):
        """Zoom y-axis to the data range so small differences are visible."""
        vmin, vmax = min(values), max(values)
        span = max(vmax - vmin, 0.01)
        ax.set_ylim(max(lo_min, vmin - span * pad),
                    min(hi_max, vmax + span * pad))

    # ── Accuracy & F1 macro ───────────────────────────────────────────────────
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

    # ── AUC-ROC ───────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    bars = ax.bar(x, auc, color="steelblue")
    _annotate(ax, bars)
    _zoom_ylim(ax, auc)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC")
    ax.grid(axis="y", alpha=0.5)

    # ── Model size ────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    bars = ax.bar(x, sizes, color="darkorange")
    _annotate(ax, bars)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Size (MB)")
    ax.set_title("Model size  (Keras = param count x4, TFLite = file size)")
    ax.grid(axis="y", alpha=0.5)

    # ── Inference latency ─────────────────────────────────────────────────────
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
    print(f"  [plot] saved -> {path}")


def _merge_histories(h1, h2):
    """Concatenate two Keras history dicts for combined plotting."""
    merged = {}
    for k in h1:
        merged[k] = list(h1[k]) + list(h2[k])
    return merged


def print_eval_header(label: str):
    print(f"\n{'─'*60}")
    print(f"  Evaluation: {label}")
    print(f"{'─'*60}")


def print_keras_eval(model, ds, label):
    print_eval_header(label)
    results = model.evaluate(ds, verbose=0)
    names   = [m.name for m in model.metrics]
    for n, v in zip(names, results):
        print(f"  {n:<16}: {v:.4f}")


# ── training ──────────────────────────────────────────────────────────────────

def train_float(train_ds, val_ds, epochs: int, skip: bool) -> tuple:
    """
    Phase 1a: frozen backbone (train head only)
    Phase 1b: unfreeze all layers (full fine-tune at low LR)
    Returns (model, merged_history_dict)
    """
    if skip and pathlib.Path(FLOAT_MODEL_PATH).exists():
        print(f"\n[Phase 1] Loading checkpoint: {FLOAT_MODEL_PATH}")
        model = load_keras_model(FLOAT_MODEL_PATH)
        return model, None

    phase1 = max(1, epochs // 2)
    phase2 = epochs - phase1

    # ── Phase 1a: head only ───────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Phase 1a – Head-only training  ({phase1} epochs)")
    print(f"{'═'*60}")
    model = build_model(trainable_base=False, learning_rate=LR_FLOAT)
    model.summary()

    pathlib.Path(FLOAT_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    callbacks_1 = [
        keras.callbacks.ModelCheckpoint(
            FLOAT_MODEL_PATH, monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2,
            min_lr=1e-7, verbose=1,
        ),
    ]
    hist1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=phase1, callbacks=callbacks_1, verbose=1,
    )

    # ── Phase 1b: full fine-tune ──────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Phase 1b – Full fine-tuning  ({phase2} epochs, lr={LR_WARMUP})")
    print(f"{'═'*60}")
    unfreeze_base(model, lr=LR_WARMUP)

    callbacks_2 = [
        keras.callbacks.ModelCheckpoint(
            FLOAT_MODEL_PATH, monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2,
            min_lr=1e-8, verbose=1,
        ),
    ]
    hist2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=phase1 + phase2, initial_epoch=phase1,
        callbacks=callbacks_2, verbose=1,
    )

    # merge histories for plotting
    full_history = _merge_histories(hist1.history, hist2.history)
    return model, full_history


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs-float", type=int, default=EPOCHS_FLOAT)
    p.add_argument("--epochs-qat",   type=int, default=EPOCHS_QAT)
    p.add_argument("--skip-train",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # GPU
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[env] GPUs: {gpus or 'none – CPU only'}")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Step 1 – Loading COCO 2017")
    print(f"{'═'*60}")
    train_ds, val_ds = load_train_val(batch_size=BATCH_SIZE)

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Step 2 – Float32 training")
    print(f"{'═'*60}")
    float_model, full_history = train_float(
        train_ds, val_ds,
        epochs=args.epochs_float,
        skip=args.skip_train,
    )

    if full_history is not None:
        plot_history(full_history, r"Float32 MobileNetV1 $\alpha$=0.25", "01_float_training.png")
        print_keras_eval(float_model, val_ds, "Float32 – validation set")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Step 3 – Evaluate float32")
    print(f"{'═'*60}")
    stats_float = evaluate_keras(float_model, val_ds, label="float32")
    plot_confusion_matrix(stats_float["confusion_matrix"],
                          "Confusion Matrix – Float32",
                          "02_cm_float.png")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Step 4 – Post-Training Quantisation (PTQ INT8)")
    print(f"{'═'*60}")
    calib = make_calibration_generator(config.NUM_CALIB_SAMPLES)
    post_training_quantize(float_model, calib, PTQ_TFLITE_PATH)

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Step 5 – Evaluate PTQ INT8")
    print(f"{'═'*60}")
    stats_ptq = evaluate_tflite(PTQ_TFLITE_PATH, val_ds, label="PTQ_int8")
    plot_confusion_matrix(stats_ptq["confusion_matrix"],
                          "Confusion Matrix – PTQ INT8",
                          "03_cm_ptq.png")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Step 6 – Quantisation-Aware Training ({args.epochs_qat} epochs)")
    print(f"{'═'*60}")
    base = load_keras_model(FLOAT_MODEL_PATH)
    qat_model = quantization_aware_training(
        base, train_ds, val_ds, epochs=args.epochs_qat,
    )

    # Plot QAT training curves (se disponibili)
    if hasattr(qat_model, "_qat_history"):
        plot_history(qat_model._qat_history,
                     r"QAT fine-tuning MobileNetV1 $\alpha$=0.25",
                     "02b_qat_training.png")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Step 7 – Evaluate QAT (Keras)")
    print(f"{'═'*60}")
    stats_qat_keras = evaluate_keras(qat_model, val_ds, label="QAT_keras")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Step 8 – Export QAT → TFLite INT8")
    print(f"{'═'*60}")
    calib2 = make_calibration_generator(config.NUM_CALIB_SAMPLES)
    qat_to_tflite(qat_model, calib2, QAT_TFLITE_PATH)

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Step 9 – Evaluate QAT TFLite INT8")
    print(f"{'═'*60}")
    stats_qat_tflite = evaluate_tflite(QAT_TFLITE_PATH, val_ds, label="QAT_int8")
    plot_confusion_matrix(stats_qat_tflite["confusion_matrix"],
                          "Confusion Matrix – QAT INT8 TFLite",
                          "04_cm_qat_tflite.png")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Step 10 – C array")
    print(f"{'═'*60}")
    tflite_to_c_array(QAT_TFLITE_PATH, C_ARRAY_PATH, C_ARRAY_H_PATH,
                      array_name=C_ARRAY_NAME)

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Step 11 – Comparison summary")
    print(f"{'═'*60}")
    all_stats = [stats_float, stats_ptq, stats_qat_keras, stats_qat_tflite]
    save_all_stats(all_stats, STATS_PATH)
    print_comparison_table(all_stats)
    plot_comparison_bars(all_stats, "05_comparison.png")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Output files")
    print(f"{'═'*60}")
    print(f"  Float checkpoint : {FLOAT_MODEL_PATH}")
    print(f"  PTQ TFLite       : {PTQ_TFLITE_PATH}")
    print(f"  QAT TFLite       : {QAT_TFLITE_PATH}")
    print(f"  C source         : {C_ARRAY_PATH}")
    print(f"  C header         : {C_ARRAY_H_PATH}")
    print(f"  Stats JSON       : {STATS_PATH}")
    print(f"  Plots            : {PLOT_DIR}/")
    for p in sorted(PLOT_DIR.glob("*.png")):
        print(f"    {p}")
    print()


if __name__ == "__main__":
    main()
