"""
Valuta i modelli (float32, PTQ, QAT) su un dataset esterno.

Strutture supportate automaticamente:
  dataset/
    positive/ (o human/, person/, 1/, yes/)   ← immagini con persone
    negative/ (o no_human/, background/, 0/, no/)  ← immagini senza

  oppure struttura flat con nomi file che iniziano per 1_ / 0_

Usage:
  python test_external.py --data test_dataset/
  python test_external.py --data test_dataset/ --models float ptq qat
"""

import argparse
import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import plot_style                        # IEEE / Computer Modern – must be before plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
)
from plot_style import SQUARE, DOUBLE

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import tf_keras as keras
tf.get_logger().setLevel("ERROR")

from model import load_keras_model
from config import (
    IMAGE_SIZE, FLOAT_MODEL_PATH,
    PTQ_TFLITE_PATH, QAT_TFLITE_PATH,
)

PLOT_DIR = pathlib.Path("output/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

POSITIVE_NAMES = {"positive", "human", "person", "humans", "persons", "1", "yes"}
NEGATIVE_NAMES = {"negative", "no_human", "no_person", "background", "0", "no", "none"}
IMG_EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Dataset loader ────────────────────────────────────────────────────────────

def discover_dataset(root: pathlib.Path) -> tuple[list[pathlib.Path], list[int]]:
    """
    Trova immagini e label in una cartella con struttura sconosciuta.
    Ritorna (paths, labels) dove label=1 → persona, label=0 → no-persona.
    """
    root = root.resolve()
    paths, labels = [], []

    # ── Caso 1: sottocartelle per classe ──────────────────────────────────────
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    pos_dirs = [d for d in subdirs if d.name.lower() in POSITIVE_NAMES]
    neg_dirs = [d for d in subdirs if d.name.lower() in NEGATIVE_NAMES]

    if pos_dirs or neg_dirs:
        for d in pos_dirs:
            imgs = [f for f in d.rglob("*") if f.suffix.lower() in IMG_EXTS]
            paths  += imgs
            labels += [1] * len(imgs)
            print(f"  [+] {d.name:20s}: {len(imgs):5d} immagini (person)")
        for d in neg_dirs:
            imgs = [f for f in d.rglob("*") if f.suffix.lower() in IMG_EXTS]
            paths  += imgs
            labels += [0] * len(imgs)
            print(f"  [-] {d.name:20s}: {len(imgs):5d} immagini (no person)")

        if paths:
            return paths, labels

    # ── Caso 2: struttura train/test con sottocartelle ────────────────────────
    split_dirs = [d for d in subdirs if d.name.lower() in {"train","test","val","valid"}]
    if split_dirs:
        print("  [auto] trovate split dirs, uso tutte le immagini")
        for split_dir in split_dirs:
            sub = [d for d in split_dir.iterdir() if d.is_dir()]
            pos = [d for d in sub if d.name.lower() in POSITIVE_NAMES]
            neg = [d for d in sub if d.name.lower() in NEGATIVE_NAMES]
            for d in pos:
                imgs = [f for f in d.rglob("*") if f.suffix.lower() in IMG_EXTS]
                paths += imgs; labels += [1]*len(imgs)
                print(f"  [+] {split_dir.name}/{d.name}: {len(imgs)} immagini (person)")
            for d in neg:
                imgs = [f for f in d.rglob("*") if f.suffix.lower() in IMG_EXTS]
                paths += imgs; labels += [0]*len(imgs)
                print(f"  [-] {split_dir.name}/{d.name}: {len(imgs)} immagini (no person)")
        if paths:
            return paths, labels

    # ── Caso 3: flat con nomi tipo 1_xxx.jpg / 0_xxx.jpg ─────────────────────
    all_imgs = [f for f in root.rglob("*") if f.suffix.lower() in IMG_EXTS]
    if all_imgs:
        for f in all_imgs:
            if f.stem.startswith("1"):
                paths.append(f); labels.append(1)
            elif f.stem.startswith("0"):
                paths.append(f); labels.append(0)
        if paths:
            print(f"  [auto] flat naming: {sum(l==1 for l in labels)} person, "
                  f"{sum(l==0 for l in labels)} no-person")
            return paths, labels

    raise RuntimeError(
        f"Struttura dataset non riconosciuta in {root}\n"
        f"Sottocartelle trovate: {[d.name for d in subdirs]}"
    )


def load_image(path: pathlib.Path) -> np.ndarray:
    raw   = tf.io.read_file(str(path))
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = keras.applications.mobilenet.preprocess_input(image)
    return image.numpy()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_keras_model(
    model_path: str,
    paths: list,
    labels: list,
    label: str,
) -> dict:
    print(f"\n{'─'*60}\n  Modello: {label}\n{'─'*60}")
    model = load_keras_model(model_path)

    y_true, y_prob = [], []
    t0 = time.perf_counter()
    for p, l in zip(paths, labels):
        img  = load_image(p)[np.newaxis]          # (1, H, W, 3)
        prob = float(model(img, training=False).numpy().flatten()[0])
        y_prob.append(prob)
        y_true.append(l)
    elapsed = time.perf_counter() - t0

    return _compute_and_print(np.array(y_true), np.array(y_prob),
                               elapsed, label, model_path)


def _quant_params(detail):
    qp = detail.get("quantization_parameters", {})
    s  = qp.get("scales",      [1.0])
    z  = qp.get("zero_points", [0])
    return float(s[0]) if s else 1.0, int(z[0]) if z else 0


def evaluate_tflite_model(
    tflite_path: str,
    paths: list,
    labels: list,
    label: str,
) -> dict:
    print(f"\n{'─'*60}\n  Modello: {label}\n{'─'*60}")
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale,  in_zp  = _quant_params(inp)
    out_scale, out_zp = _quant_params(out)

    y_true, y_prob = [], []
    t0 = time.perf_counter()
    for p, l in zip(paths, labels):
        img = load_image(p)[np.newaxis].astype(np.float32)
        if inp["dtype"] == np.int8:
            img = (img / in_scale + in_zp).astype(np.int8)
        elif inp["dtype"] == np.uint8:
            img = (img / in_scale + in_zp).astype(np.uint8)
        interp.set_tensor(inp["index"], img)
        interp.invoke()
        raw = interp.get_tensor(out["index"])
        if out["dtype"] in (np.int8, np.uint8):
            raw = (raw.astype(np.float32) - out_zp) * out_scale
        y_prob.append(float(raw.flatten()[0]))
        y_true.append(l)
    elapsed = time.perf_counter() - t0

    return _compute_and_print(np.array(y_true), np.array(y_prob),
                               elapsed, label, tflite_path)


def _optimal_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[np.argmax(tpr - fpr)]), fpr, tpr

def _compute_and_print(y_true, y_prob, elapsed, label, model_path):
    n   = len(y_true)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    thr_opt, fpr, tpr = _optimal_threshold(y_true, y_prob)

    def _m(thr):
        yp  = (y_prob >= thr).astype(int)
        cm  = confusion_matrix(y_true, yp)
        rep = classification_report(y_true, yp,
                                    target_names=["no_person","person"],
                                    output_dict=True)
        return yp, cm, rep

    yp50,  cm50,  rep50  = _m(0.5)
    ypopt, cmopt, repopt = _m(thr_opt)

    stats = {
        "label":               label,
        "n_samples":           n,
        # @ optimal threshold (default)
        "accuracy":            float(repopt["accuracy"]),
        "precision_person":    float(repopt["person"]["precision"]),
        "recall_person":       float(repopt["person"]["recall"]),
        "f1_person":           float(repopt["person"]["f1-score"]),
        "f1_macro":            float(f1_score(y_true, ypopt, average="macro")),
        "confusion_matrix":    cmopt.tolist(),
        "threshold_optimal":   thr_opt,
        # @ threshold 0.5
        "accuracy_t50":        float(rep50["accuracy"]),
        "f1_macro_t50":        float(f1_score(y_true, yp50, average="macro")),
        "confusion_matrix_t50": cm50.tolist(),
        # shared
        "auc_roc":             auc,
        "_roc_fpr":            fpr.tolist(),
        "_roc_tpr":            tpr.tolist(),
        "inference_ms_per_img": round(elapsed * 1000 / n, 2),
        "model_size_mb":       pathlib.Path(model_path).stat().st_size / 1e6,
    }

    print(f"  Campioni         : {n}")
    print(f"  AUC-ROC          : {auc:.4f}")
    print(f"\n  --- threshold = 0.50 ---")
    tn,fp,fn,tp = cm50.ravel()
    print(f"  Accuracy         : {stats['accuracy_t50']:.4f}  F1 macro: {stats['f1_macro_t50']:.4f}")
    print(f"  TN={tn:5d} FP={fp:5d} / FN={fn:5d} TP={tp:5d}")
    print(f"\n  --- threshold = {thr_opt:.3f} (Youden optimal) ---")
    tn,fp,fn,tp = cmopt.ravel()
    print(f"  Accuracy         : {stats['accuracy']:.4f}  F1 macro: {stats['f1_macro']:.4f}")
    print(f"  Precision (pers) : {stats['precision_person']:.4f}")
    print(f"  Recall    (pers) : {stats['recall_person']:.4f}")
    print(f"  TN={tn:5d} FP={fp:5d} / FN={fn:5d} TP={tp:5d}")
    print(f"\n  Inference        : {stats['inference_ms_per_img']:.2f} ms/img")
    print(f"  Model size       : {stats['model_size_mb']:.3f} MB")
    print()
    print(classification_report(y_true, ypopt, target_names=["no_person","person"]))
    return stats


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm_list, title, filename, threshold_label=""):
    cm = np.array(cm_list)
    fig, ax = plt.subplots(figsize=SQUARE)
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["no person", "person"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["no person", "person"])
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(f"{title}\n{threshold_label}" if threshold_label else title)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    path = PLOT_DIR / filename
    plt.savefig(path); plt.close()
    print(f"  [plot] → {path}")


def plot_roc(all_stats, filename):
    fig, ax = plt.subplots(figsize=SQUARE)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (AUC = 0.50)")
    for s in all_stats:
        fpr = np.array(s["_roc_fpr"])
        tpr = np.array(s["_roc_tpr"])
        j   = np.argmax(tpr - fpr)
        thr = s["threshold_optimal"]
        line, = ax.plot(fpr, tpr, lw=1.5,
                        label=f"{s['label']}  (AUC = {s['auc_roc']:.3f})")
        ax.scatter(fpr[j], tpr[j], color=line.get_color(), s=40, zorder=5)
        ax.annotate(f"  $\\tau$={thr:.2f}", xy=(fpr[j], tpr[j]),
                    fontsize=7.5, color=line.get_color(), va="bottom")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.legend(loc="lower right")
    path = PLOT_DIR / filename
    plt.savefig(path); plt.close()
    print(f"  [plot] → {path}")


def plot_comparison(all_stats, filename):
    labels = [s["label"]                for s in all_stats]
    acc    = [s["accuracy"]             for s in all_stats]
    f1     = [s["f1_macro"]             for s in all_stats]
    prec   = [s["precision_person"]     for s in all_stats]
    rec    = [s["recall_person"]        for s in all_stats]
    lat    = [s["inference_ms_per_img"] for s in all_stats]
    sizes  = [s["model_size_mb"]        for s in all_stats]

    x, w = np.arange(len(labels)), 0.2
    fig, axes = plt.subplots(2, 2, figsize=DOUBLE)

    ax = axes[0, 0]
    ax.bar(x - w*1.5, acc,  w, label="Accuracy")
    ax.bar(x - w*0.5, f1,   w, label="F1 macro")
    ax.bar(x + w*0.5, prec, w, label="Precision")
    ax.bar(x + w*1.5, rec,  w, label="Recall")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05); ax.set_title("Classification metrics (optimal threshold)")
    ax.legend(); ax.grid(axis="y")

    ax = axes[0, 1]
    ax.bar(labels, [s["auc_roc"] for s in all_stats], color="steelblue")
    ax.set_ylim(0, 1.05); ax.set_title("AUC-ROC")
    ax.set_xticklabels(labels, rotation=15, ha="right"); ax.grid(axis="y")

    ax = axes[1, 0]
    ax.bar(labels, sizes, color="C1")
    ax.set_ylabel("Size (MB)"); ax.set_title("Model size")
    ax.set_xticklabels(labels, rotation=15, ha="right"); ax.grid(axis="y")

    ax = axes[1, 1]
    ax.bar(labels, lat, color="C2")
    ax.set_ylabel("ms / image"); ax.set_title("CPU inference latency")
    ax.set_xticklabels(labels, rotation=15, ha="right"); ax.grid(axis="y")

    plt.suptitle("Human Detection Dataset -- Model comparison", y=1.02)
    plt.tight_layout()
    path = PLOT_DIR / filename
    plt.savefig(path); plt.close()
    print(f"  [plot] → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="test_dataset",
                   help="cartella del dataset (default: test_dataset/)")
    p.add_argument("--models", nargs="+",
                   choices=["float", "ptq", "qat"], default=["float", "ptq", "qat"],
                   help="quali modelli valutare")
    return p.parse_args()


def main():
    args  = parse_args()
    root  = pathlib.Path(args.data)

    print(f"\n{'═'*60}")
    print(f"  Human Detection Dataset – Valutazione esterna")
    print(f"  Dataset: {root.resolve()}")
    print(f"{'═'*60}")

    if not root.exists():
        raise FileNotFoundError(
            f"Cartella non trovata: {root}\n"
            f"Scarica il dataset da Kaggle e mettilo in: {root.resolve()}"
        )

    print("\n[1] Scansione dataset …")
    paths, labels = discover_dataset(root)
    n_pos = sum(l == 1 for l in labels)
    n_neg = sum(l == 0 for l in labels)
    print(f"\n  Totale: {len(paths)} immagini  "
          f"(person={n_pos}, no-person={n_neg})")

    all_stats = []

    if "float" in args.models and pathlib.Path(FLOAT_MODEL_PATH).exists():
        print(f"\n[2] Float32 – {FLOAT_MODEL_PATH}")
        s = evaluate_keras_model(FLOAT_MODEL_PATH, paths, labels, "float32")
        thr_lbl = f"threshold = {s['threshold_optimal']:.3f} (Youden)"
        plot_confusion_matrix(s["confusion_matrix"],
                              "Confusion Matrix -- Float32",
                              "ext_cm_float.png", thr_lbl)
        all_stats.append(s)

    if "ptq" in args.models and pathlib.Path(PTQ_TFLITE_PATH).exists():
        print(f"\n[3] PTQ INT8 – {PTQ_TFLITE_PATH}")
        s = evaluate_tflite_model(PTQ_TFLITE_PATH, paths, labels, "PTQ int8")
        thr_lbl = f"threshold = {s['threshold_optimal']:.3f} (Youden)"
        plot_confusion_matrix(s["confusion_matrix"],
                              "Confusion Matrix -- PTQ INT8",
                              "ext_cm_ptq.png", thr_lbl)
        all_stats.append(s)

    if "qat" in args.models and pathlib.Path(QAT_TFLITE_PATH).exists():
        print(f"\n[4] QAT INT8 – {QAT_TFLITE_PATH}")
        s = evaluate_tflite_model(QAT_TFLITE_PATH, paths, labels, "QAT int8")
        thr_lbl = f"threshold = {s['threshold_optimal']:.3f} (Youden)"
        plot_confusion_matrix(s["confusion_matrix"],
                              "Confusion Matrix -- QAT INT8",
                              "ext_cm_qat.png", thr_lbl)
        all_stats.append(s)

    if len(all_stats) > 1:
        print(f"\n[5] Grafici di confronto …")
        plot_roc(all_stats, "ext_roc.png")
        plot_comparison(all_stats, "ext_comparison.png")

    # ── Tabella riassuntiva ───────────────────────────────────────────────────
    keys = ["accuracy", "accuracy_t50", "f1_macro", "auc_roc",
            "threshold_optimal", "precision_person", "recall_person",
            "inference_ms_per_img"]
    header = f"{'Metrica':<28}" + "".join(f"{s['label']:>18}" for s in all_stats)
    print(f"\n{'═'*len(header)}")
    print("  CONFRONTO FINALE – Human Detection Dataset")
    print(f"{'═'*len(header)}")
    print(header)
    print("─" * len(header))
    for k in keys:
        row = f"  {k:<26}"
        for s in all_stats:
            row += f"{s.get(k, float('nan')):>18.4f}"
        print(row)
    print(f"{'═'*len(header)}\n")

    print("Grafici salvati in output/plots/:")
    for p in sorted(PLOT_DIR.glob("ext_*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
