"""
Evaluation utilities.

Works for three kinds of models:
  - Keras float model
  - TFLite model (file path or bytes)
  - QAT Keras model (same as float path)
"""

import time
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    f1_score,
)

import plot_style                        # IEEE / Computer Modern
from plot_style import SQUARE, DOUBLE
from config import NUM_CLASSES

PLOT_DIR = pathlib.Path("output/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ── Keras evaluation ──────────────────────────────────────────────────────────

def evaluate_keras(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    label: str = "float32",
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Evaluating Keras model: {label}")
    print(f"{'='*60}")

    y_true, y_pred_prob = [], []
    t0 = time.perf_counter()
    for images, labels in dataset:
        out = model(images, training=False).numpy()   # [batch, 2]
        y_pred_prob.extend(out[:, 1].tolist())        # person probability (index 1)
        y_true.extend(labels.numpy().flatten().tolist())
    elapsed = time.perf_counter() - t0

    stats = _compute_stats(
        np.array(y_true), np.array(y_pred_prob),
        n_samples=len(y_true), elapsed_s=elapsed, label=label,
    )

    keras_metrics = model.evaluate(dataset, verbose=0)
    for name, val in zip([m.name for m in model.metrics], keras_metrics):
        stats[f"keras_{name}"] = float(val)

    stats["model_size_mb"] = _keras_size_mb(model)
    _print_stats(stats)
    return stats


# ── TFLite evaluation ─────────────────────────────────────────────────────────

def evaluate_tflite(
    tflite_path: str,
    dataset: tf.data.Dataset,
    label: str = "int8",
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Evaluating TFLite model: {label}")
    print(f"{'='*60}")

    interpreter = _load_interpreter(tflite_path)
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale,  input_zero_point  = _quant_params(input_details[0])
    output_scale, output_zero_point = _quant_params(output_details[0])

    y_true, y_pred_prob = [], []
    t0 = time.perf_counter()

    for images, labels in dataset:
        for i in range(images.shape[0]):
            img = images[i:i+1].numpy()
            if input_details[0]["dtype"] == np.int8:
                img = (img / input_scale + input_zero_point).astype(np.int8)
            elif input_details[0]["dtype"] == np.uint8:
                img = (img / input_scale + input_zero_point).astype(np.uint8)
            interpreter.set_tensor(input_details[0]["index"], img)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]["index"])
            if output_details[0]["dtype"] in (np.int8, np.uint8):
                out = (out.astype(np.float32) - output_zero_point) * output_scale
            # out shape: [1, 2] — index 1 is person probability
            y_pred_prob.append(float(out.flatten()[1]))
            y_true.append(float(labels[i].numpy()))

    elapsed = time.perf_counter() - t0

    stats = _compute_stats(
        np.array(y_true), np.array(y_pred_prob),
        n_samples=len(y_true), elapsed_s=elapsed, label=label,
    )
    stats["model_size_mb"] = pathlib.Path(tflite_path).stat().st_size / 1e6
    stats["tflite_path"]   = str(tflite_path)
    _print_stats(stats)
    return stats


# ── Core stats computation ────────────────────────────────────────────────────

def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Youden's J: threshold che massimizza TPR - FPR sulla curva ROC."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    return float(thresholds[np.argmax(j_scores)])


def _compute_stats(
    y_true:     np.ndarray,
    y_pred_prob: np.ndarray,
    n_samples:  int,
    elapsed_s:  float,
    label:      str,
) -> dict:
    y_true_int = y_true.astype(int)

    # soglia ottimale via Youden's J
    thr_opt = _optimal_threshold(y_true_int, y_pred_prob)
    thr_50  = 0.5

    def _metrics(threshold):
        y_pred = (y_pred_prob >= threshold).astype(int)
        cm     = confusion_matrix(y_true_int, y_pred).tolist()
        rep    = classification_report(
            y_true_int, y_pred,
            target_names=["no_person", "person"],
            output_dict=True,
        )
        return {
            "accuracy":            float(rep["accuracy"]),
            "precision_person":    float(rep["person"]["precision"]),
            "recall_person":       float(rep["person"]["recall"]),
            "f1_person":           float(rep["person"]["f1-score"]),
            "precision_no_person": float(rep["no_person"]["precision"]),
            "recall_no_person":    float(rep["no_person"]["recall"]),
            "f1_no_person":        float(rep["no_person"]["f1-score"]),
            "f1_macro":            float(f1_score(y_true_int, y_pred, average="macro")),
            "confusion_matrix":    cm,
        }

    m50  = _metrics(thr_50)
    mopt = _metrics(thr_opt)

    # salva curva ROC per i plot
    fpr, tpr, _ = roc_curve(y_true_int, y_pred_prob)
    auc = float(roc_auc_score(y_true_int, y_pred_prob))

    return {
        "label":                 label,
        "n_samples":             n_samples,
        # metriche @ threshold 0.5
        **{f"{k}_t50": v for k, v in m50.items()},
        # metriche @ threshold ottimale (default per tabelle e CM)
        **mopt,
        "threshold_optimal":     thr_opt,
        "auc_roc":               auc,
        "_roc_fpr":              fpr.tolist(),
        "_roc_tpr":              tpr.tolist(),
        "inference_ms_per_img":  round(elapsed_s * 1000 / max(n_samples, 1), 3),
        "total_inference_s":     round(elapsed_s, 3),
    }


def _print_stats(stats: dict) -> None:
    thr = stats["threshold_optimal"]
    print(f"\n  Label              : {stats['label']}")
    print(f"  Samples            : {stats['n_samples']}")
    print(f"  AUC-ROC            : {stats['auc_roc']:.4f}")
    print(f"\n  --- threshold = 0.50 ---")
    print(f"  Accuracy           : {stats['accuracy_t50']:.4f}")
    print(f"  F1  macro          : {stats['f1_macro_t50']:.4f}")
    print(f"  Precision (person) : {stats['precision_person_t50']:.4f}")
    print(f"  Recall    (person) : {stats['recall_person_t50']:.4f}")
    cm = stats["confusion_matrix_t50"]
    print(f"  Confusion matrix   :  TN={cm[0][0]:6d}  FP={cm[0][1]:6d}")
    print(f"                        FN={cm[1][0]:6d}  TP={cm[1][1]:6d}")
    print(f"\n  --- threshold = {thr:.3f} (Youden optimal) ---")
    print(f"  Accuracy           : {stats['accuracy']:.4f}")
    print(f"  F1  macro          : {stats['f1_macro']:.4f}")
    print(f"  Precision (person) : {stats['precision_person']:.4f}")
    print(f"  Recall    (person) : {stats['recall_person']:.4f}")
    cm = stats["confusion_matrix"]
    print(f"  Confusion matrix   :  TN={cm[0][0]:6d}  FP={cm[0][1]:6d}")
    print(f"                        FN={cm[1][0]:6d}  TP={cm[1][1]:6d}")
    print(f"  Inference          : {stats['inference_ms_per_img']:.2f} ms/img")
    print(f"  Model size         : {stats.get('model_size_mb', 0):.3f} MB")


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm_list: list, title: str, filename: str,
                          threshold_label: str = "") -> None:
    cm = np.array(cm_list)
    total = cm.sum()

    fig, ax = plt.subplots(figsize=SQUARE)
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["no person", "person"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes, fontsize=12, rotation=90, va="center")
    ax.set_xlabel("Predicted label", labelpad=8)
    ax.set_ylabel("True label",      labelpad=8)
    full_title = f"{title}\n{threshold_label}" if threshold_label else title
    ax.set_title(full_title, pad=12)
    ax.grid(False)

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = count / total * 100
            color = "white" if count > cm.max() / 2 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=14, fontweight="bold", color=color)

    plt.tight_layout()
    path = PLOT_DIR / filename
    plt.savefig(path); plt.close()
    print(f"  [plot] -> {path}")


def plot_roc_curves(all_stats: list[dict], filename: str) -> None:
    """Sovrappone le curve ROC di tutti i modelli in un singolo grafico."""
    fig, ax = plt.subplots(figsize=SQUARE)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (AUC = 0.50)")

    for s in all_stats:
        fpr = np.array(s["_roc_fpr"])
        tpr = np.array(s["_roc_tpr"])
        auc = s["auc_roc"]
        thr = s["threshold_optimal"]
        # punto ottimale sulla curva
        fpr_r, tpr_r, thrs = roc_curve(
            [0]*1 + [1]*1, [0.0, 1.0]   # dummy, ricostruiamo dal saved data
        )
        line, = ax.plot(fpr, tpr, lw=1.5,
                        label=f"{s['label']}  (AUC = {auc:.3f})")
        # segna il punto Youden (threshold ottimale)
        j = np.argmax(tpr - fpr)
        ax.scatter(fpr[j], tpr[j], color=line.get_color(),
                   s=40, zorder=5, marker="o")
        ax.annotate(f"  $\\tau$={thr:.2f}",
                    xy=(fpr[j], tpr[j]),
                    fontsize=7.5, color=line.get_color(),
                    va="bottom")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.legend(loc="lower right")
    path = PLOT_DIR / filename
    plt.savefig(path); plt.close()
    print(f"  [plot] → {path}")


# ── Stats persistence ─────────────────────────────────────────────────────────

def save_all_stats(all_stats: list[dict], path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    # rimuovi i dati ROC (troppo lunghi per il JSON leggibile)
    clean = [{k: v for k, v in s.items() if not k.startswith("_")}
             for s in all_stats]
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\n[stats] saved to {path}")


def print_comparison_table(all_stats: list[dict]) -> None:
    keys = [
        ("accuracy",           "Accuracy (opt. thr.)"),
        ("accuracy_t50",       "Accuracy (thr.=0.50)"),
        ("f1_macro",           "F1 macro (opt. thr.)"),
        ("auc_roc",            "AUC-ROC"),
        ("threshold_optimal",  "Optimal threshold"),
        ("precision_person",   "Precision person"),
        ("recall_person",      "Recall person"),
        ("inference_ms_per_img","Inference ms/img"),
        ("model_size_mb",      "Model size MB"),
    ]
    header = f"{'Metric':<28}" + "".join(f"{s['label']:>20}" for s in all_stats)
    print(f"\n{'='*len(header)}")
    print("  COMPARISON TABLE")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for key, display in keys:
        row = f"  {display:<26}"
        for s in all_stats:
            val = s.get(key, float("nan"))
            row += f"{val:>20.4f}" if isinstance(val, float) else f"{'N/A':>20}"
        print(row)
    print(f"{'='*len(header)}\n")


# ── Private helpers ───────────────────────────────────────────────────────────

def _keras_size_mb(model) -> float:
    total = sum(np.prod(w.shape) for w in model.trainable_weights)
    return total * 4 / 1e6

def _load_interpreter(path: str) -> tf.lite.Interpreter:
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    return interp

def _quant_params(detail: dict) -> tuple[float, int]:
    qp = detail.get("quantization_parameters", {})
    scales      = qp.get("scales",      [1.0])
    zero_points = qp.get("zero_points", [0])
    return (float(scales[0]) if scales else 1.0,
            int(zero_points[0]) if zero_points else 0)

# re-export roc_curve for callers that plot ROC from stored data
from sklearn.metrics import roc_curve
