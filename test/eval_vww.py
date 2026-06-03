#!/usr/bin/env python3
"""
eval_vww.py — Person detection evaluation on Visual Wake Words (VWW)

Evaluates the TFLite int8 model (MobileNetV1 Q8) used by the Smart Visual
Alarm firmware.  Preprocessing matches image_provider.cc exactly.

Metrics produced
────────────────
  Accuracy, Precision, Recall, F1, FAR (False Alarm Rate)
  ROC curve + AUC
  Precision-Recall curve + Average Precision
  Confusion matrix
  Threshold sweep  θ ∈ [0.50, 0.95]
  Moving-average sensitivity  W ∈ {1, 3, 5, 7}  (firmware uses W=5)
  95 % bootstrap confidence intervals (n=1000)

Dataset
───────
  Primary  : Visual Wake Words (TFDS)   → pip install tensorflow-datasets
  Fallback : local directory with layout
               <dataset-dir>/person/     *.jpg / *.png
               <dataset-dir>/non_person/ *.jpg / *.png

Model
─────
  Auto-detected from ../main/person_detect_model_data.cc, or supply:
    --model       path/to/model.tflite
    --model-cc    path/to/person_detect_model_data.cc

Usage examples
──────────────
  python eval_vww.py
  python eval_vww.py --theta 0.65 --max-samples 1000
  python eval_vww.py --model-cc ../main/person_detect_model_data.cc
  python eval_vww.py --dataset-dir ~/datasets/vww
  python eval_vww.py --no-bootstrap --max-samples 200   # quick sanity run
"""

import argparse
import json
import queue
import re
import sys
import threading
from pathlib import Path

import numpy as np

# ── matplotlib non-interactive backend (safe for headless servers) ─────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _setup_ieee_style():
    """
    Apply a matplotlib style that matches IEEEtran:
      - Times Roman body font  +  Computer Modern math
      - Font sizes tuned for a single-column figure (~3.5 in wide)
    Uses text.usetex=True when a LaTeX installation is available,
    otherwise falls back to STIX (close visual match, no LaTeX needed).
    """
    import shutil
    use_latex = shutil.which("latex") is not None
    base = {
        "font.family":        "serif",
        "axes.labelsize":     9,
        "font.size":          9,
        "legend.fontsize":    7,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "axes.linewidth":     0.6,
        "grid.linewidth":     0.4,
        "lines.linewidth":    1.5,
        "figure.dpi":         200,
    }
    if use_latex:
        base.update({
            "text.usetex":         True,
            "text.latex.preamble": r"\usepackage{amsmath}",
        })
    else:
        base.update({
            "text.usetex":      False,
            "font.serif":       ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "cm",
        })
    plt.rcParams.update(base)

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ── TFLite interpreter ────────────────────────────────────────────────────────
# ai_edge_litert is the recommended replacement for tf.lite.Interpreter and
# correctly handles old TFLite Micro models (pre-2020 quantized_dimension quirk).
try:
    from ai_edge_litert.interpreter import Interpreter as _Interpreter
except ImportError:
    try:
        import tensorflow as tf
        _Interpreter = tf.lite.Interpreter
    except ImportError:
        try:
            import tflite_runtime.interpreter as _tflite
            _Interpreter = _tflite.Interpreter
        except ImportError:
            sys.exit(
                "No TFLite runtime found.\n"
                "  pip install ai-edge-litert      # recommended\n"
                "  pip install tensorflow          # alternative"
            )

# ── VWW via tensorflow-datasets (optional) ────────────────────────────────────
try:
    import tensorflow_datasets as tfds
    _HAS_TFDS = True
except ImportError:
    _HAS_TFDS = False

from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
IMG_SIZE = 96      # firmware model input (px)

# BT.601 luma weights — matches image_provider.cc:
#   (305*R + 600*G + 119*B) >> 10
_LW = np.array([305.0, 600.0, 119.0]) / 1024.0


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing  (must match firmware exactly)
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """
    RGB uint8 H×W×3  →  int8 96×96  (same pipeline as image_provider.cc).

    Resize → BT.601 luma → subtract 128 (= uint8 XOR 0x80).
    """
    pil = Image.fromarray(img_rgb).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32)           # 96×96×3
    gray = (arr @ _LW).clip(0, 255).astype(np.uint8)  # 96×96
    return (gray.astype(np.int16) - 128).astype(np.int8)  # 96×96, int8


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────
def _bytes_from_tflite(path: Path) -> bytes:
    return path.read_bytes()


def _bytes_from_cc(path: Path) -> bytes:
    """
    Parse  alignas(8) const unsigned char g_person_detect_model_data[] = { … };
    and return the raw TFLite flatbuffer bytes.
    """
    text = path.read_text(errors="replace")
    m = re.search(r'=\s*\{([^}]+)\}', text, re.DOTALL)
    if not m:
        sys.exit(f"[model] Could not find C array body in {path}")
    hex_vals = re.findall(r'0x([0-9a-fA-F]{2})', m.group(1))
    if not hex_vals:
        sys.exit(f"[model] No hex bytes found in {path}")
    print(f"[model] Extracted {len(hex_vals)/1024:.1f} KB from {path.name}")
    return bytes(int(h, 16) for h in hex_vals)


def _fix_quant_dimensions(model_bytes: bytes) -> bytes:
    """
    Patch old TFLite Micro models (pre-2020) where QuantizationParameters has
    quantized_dimension >= tensor_rank, causing TFLite runtimes >= 2.18 to reject
    the model.  Navigates the flatbuffer directly (no external schema needed):
      Model → subgraphs → tensors → quantization → quantized_dimension
    and zeroes any field whose value exceeds the tensor's rank.
    """
    import struct

    buf = bytearray(model_bytes)

    def u32(pos):  return struct.unpack_from('<I', buf, pos)[0]
    def i32(pos):  return struct.unpack_from('<i', buf, pos)[0]
    def u16(pos):  return struct.unpack_from('<H', buf, pos)[0]

    def field_pos(table, idx):
        # Returns absolute position of field data inside table, or None.
        vtable = table - i32(table)       # vtable is at table - soffset
        vtsize = u16(vtable)
        slot   = 4 + idx * 2
        if slot >= vtsize:
            return None
        off = u16(vtable + slot)
        return (table + off) if off else None

    def deref(pos):
        # Follow a UOffset32 stored at pos: result = pos + value.
        return pos + u32(pos)

    def vec_elem(vec, i):
        # Element i of a flatbuffers vector of tables.
        return deref(vec + 4 + i * 4)

    # TFLite flatbuffer layout:
    #   bytes 0-3 : UOffset32 → Model table
    #   bytes 4-7 : file id "TFL3"
    model = deref(0)                           # Model table

    sg_vec_ref = field_pos(model, 2)           # Model.subgraphs  (field 2)
    if sg_vec_ref is None:
        return bytes(buf)
    sg_vec = deref(sg_vec_ref)

    patches = 0
    for sg_i in range(u32(sg_vec)):
        sg = vec_elem(sg_vec, sg_i)

        t_vec_ref = field_pos(sg, 0)           # SubGraph.tensors  (field 0)
        if t_vec_ref is None:
            continue
        t_vec = deref(t_vec_ref)

        for t_i in range(u32(t_vec)):
            tensor = vec_elem(t_vec, t_i)

            q_ref = field_pos(tensor, 4)       # Tensor.quantization  (field 4)
            if q_ref is None:
                continue
            quant = deref(q_ref)

            qd_pos = field_pos(quant, 6)       # QuantizationParameters.quantized_dimension (field 6)
            if qd_pos is None:
                continue
            qd_val = i32(qd_pos)
            if qd_val == 0:
                continue

            # Validate against tensor rank
            sh_ref = field_pos(tensor, 0)      # Tensor.shape  (field 0)
            rank = u32(deref(sh_ref)) if sh_ref else 0
            if qd_val >= rank:
                struct.pack_into('<i', buf, qd_pos, 0)
                patches += 1

    if patches:
        print(f"[model]  Patched {patches} invalid quantized_dimension field(s) "
              f"(old TFLite Micro model format)")
    return bytes(buf)


def build_interpreter(model_bytes: bytes):
    patched = _fix_quant_dimensions(model_bytes)
    interp  = _Interpreter(model_content=patched)
    interp.allocate_tensors()
    return interp


def _auto_find_model() -> tuple[bytes, str]:
    """Try to locate the model relative to this script."""
    here = Path(__file__).parent
    cc_candidates = [
        here / "../main/person_detect_model_data.cc",
        here / "../../main/person_detect_model_data.cc",
    ]
    tfl_candidates = [
        here / "../build/model.tflite",
        here / "model.tflite",
    ]
    for p in cc_candidates:
        if p.resolve().exists():
            return _bytes_from_cc(p.resolve()), str(p.resolve())
    for p in tfl_candidates:
        if p.resolve().exists():
            return _bytes_from_tflite(p.resolve()), str(p.resolve())
    sys.exit(
        "[model] Cannot auto-locate model.\n"
        "  Use --model <file.tflite>  or  --model-cc <person_detect_model_data.cc>"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
def infer(interp, img_int8: np.ndarray) -> float:
    """
    Return person probability ∈ [0,1].

    Model output layout (from model_settings.cc):
      index 0 → "notperson"
      index 1 → "person"
    The model's last op is Softmax, so the dequantised output is already a
    probability vector summing to 1.  Return index 1 directly — no second
    softmax needed (matches the firmware's RespondToDetection behaviour).
    """
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    x = img_int8.reshape(inp_det["shape"]).astype(np.int8)
    interp.set_tensor(inp_det["index"], x)
    interp.invoke()

    raw = interp.get_tensor(out_det["index"])[0]  # shape [2]
    zp    = out_det["quantization_parameters"]["zero_points"]
    scale = out_det["quantization_parameters"]["scales"]

    if len(scale) > 0 and scale[0] != 0.0:
        probs = (raw.astype(np.float32) - zp) * scale
    else:
        probs = raw.astype(np.float32) / 128.0

    return float(probs[1])   # P(person), already post-softmax from the model


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────────
def _load_static_samples() -> tuple | None:
    """
    Load the 10 raw 96×96 grayscale samples shipped with the project
    (static_images/sample_images/).  Labels from the README.
    Returns (images_as_rgb_uint8, labels) or None if not found.

    WARNING: n=10 is not statistically meaningful.  Use only to verify
    the inference pipeline works before running on a real dataset.
    """
    here   = Path(__file__).parent
    folder = here / "../static_images/sample_images"
    if not folder.exists():
        return None

    # Labels from static_images/sample_images/README.md
    label_map = {
        "image0": 1, "image1": 0, "image2": 1, "image3": 0,
        "image4": 1, "image5": 0, "image6": 1, "image7": 1,
        "image8": 1, "image9": 1,
    }
    images, labels = [], []
    for name, lbl in sorted(label_map.items()):
        p = folder / name
        if not p.exists():
            continue
        raw = np.frombuffer(p.read_bytes(), dtype=np.uint8)
        if raw.size != 96 * 96:
            continue
        # Raw is 96×96 grayscale → convert to RGB so preprocess() can handle it
        gray = raw.reshape(96, 96)
        rgb  = np.stack([gray, gray, gray], axis=-1)  # H×W×3
        images.append(rgb)
        labels.append(lbl)

    if not images:
        return None
    print(f"[dataset] Loaded {len(images)} static samples from project "
          f"(person={sum(labels)}, no_person={len(labels)-sum(labels)})")
    print("[dataset] WARNING: n=10 — pipeline smoke-test only, not a real benchmark.")
    return images, np.array(labels, dtype=int)


def _load_tfds(split: str):
    if not _HAS_TFDS:
        return None
    # Only try datasets that don't require a massive download.
    # visual_wake_words: needs COCO pre-downloaded locally (~780 MB).
    # wake_vision:       ~150 GB — skipped automatically.
    candidates = [
        # (dataset_name, label_key, person_value, max_download_gb)
        ("visual_wake_words", "label", 1, 1.0),
    ]
    split_map = {"test": "test", "val": "validation", "train": "train"}

    for ds_name, label_key, person_val, _ in candidates:
        actual_split = split_map.get(split, split)
        try:
            print(f"[dataset] Trying TFDS '{ds_name}' ({actual_split}) …")
            ds = tfds.load(ds_name, split=actual_split, shuffle_files=False)
            images, labels = [], []
            for ex in tfds.as_numpy(ds):
                images.append(ex["image"])
                labels.append(1 if ex[label_key] == person_val else 0)
            n, pos = len(images), sum(labels)
            print(f"[dataset] {ds_name}: {n} images  (person={pos}, no_person={n-pos})")
            return images, np.array(labels, dtype=int)
        except Exception as e:
            # Suppress the enormous "available datasets" list
            first_line = str(e).split("\n")[0]
            print(f"[dataset] '{ds_name}' unavailable: {first_line}")

    return None


def _load_dir(root: Path):
    """
    Expects:
      root/person/     *.jpg *.png
      root/non_person/ *.jpg *.png
    Returns paths (not arrays) to avoid loading everything into RAM.
    """
    paths, labels = [], []
    for label, folder in [(1, "person"), (0, "non_person")]:
        d = root / folder
        if not d.exists():
            print(f"[dataset] Warning: {d} not found, skipping")
            continue
        for p in sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")):
            paths.append(p)
            labels.append(label)
    if not paths:
        return None
    n, pos = len(paths), sum(labels)
    print(f"[dataset] {n} images from {root}  (person={pos}, no_person={n-pos})")
    return paths, np.array(labels, dtype=int)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ──────────────────────────────────────────────────────────────────────────────
def _scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return dict(
        TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
        accuracy  = float(accuracy_score(y_true, y_pred)),
        precision = float(precision_score(y_true, y_pred, zero_division=0)),
        recall    = float(recall_score(y_true, y_pred, zero_division=0)),
        f1        = float(f1_score(y_true, y_pred, zero_division=0)),
        FAR       = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
    )


def _bootstrap_ci(y_true: np.ndarray, y_scores: np.ndarray,
                   theta: float, n_boot: int = 1000, seed: int = 42) -> dict:
    """95 % bootstrap CIs for Accuracy, F1, FAR."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc_b, f1_b, far_b = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = _scalar_metrics(y_true[idx], (y_scores[idx] >= theta).astype(int))
        acc_b.append(m["accuracy"])
        f1_b.append(m["f1"])
        far_b.append(m["FAR"])
    lo, hi = 0.025, 0.975
    return {
        "accuracy_ci": (float(np.quantile(acc_b, lo)), float(np.quantile(acc_b, hi))),
        "f1_ci":       (float(np.quantile(f1_b,  lo)), float(np.quantile(f1_b,  hi))),
        "FAR_ci":      (float(np.quantile(far_b,  lo)), float(np.quantile(far_b,  hi))),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot]   {path.name}")


def plot_roc(y_true, y_scores, roc_auc, out_dir):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=1.5, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve --- MobileNetV1 Q8 on VWW")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "roc_curve.png")


def plot_pr(y_true, y_scores, ap, out_dir):
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    baseline = y_true.mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, lw=1.5, label=f"AP = {ap:.3f}")
    ax.axhline(baseline, linestyle="--", color="grey", lw=1,
               label=f"Baseline = {baseline:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve --- MobileNetV1 Q8 on VWW")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "pr_curve.png")


def plot_confusion(y_true, y_scores, thetas, out_dir):
    """2×2 grid of confusion matrices, one per threshold."""
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7, 5.5))
    # Tighten horizontal gap between left and right pair, keep more vertical room
    fig.subplots_adjust(wspace=0.12, hspace=0.38)

    for idx, (ax, theta) in enumerate(zip(axes.flat, thetas)):
        row, col = divmod(idx, ncols)
        pred = (y_scores >= theta).astype(int)
        m    = _scalar_metrics(y_true, pred)
        cm   = confusion_matrix(y_true, pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=[r"no\_person", "person"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format='d')

        ax.set_title(
            r"$\boldsymbol{\theta=" + f"{theta:.2f}" + r"}$"
            f"\nF1={m['f1']:.3f}  FAR={m['FAR']:.3f}",
            fontsize=9,
        )
        ax.tick_params(labelsize=8)

        # x-label only on bottom row
        if row < nrows - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Predicted label", fontsize=9)

        # y-label only on left column
        if col > 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("True label", fontsize=9)

    fig.suptitle("Confusion Matrices at Different Thresholds", fontsize=10)
    _save(fig, out_dir / "confusion_matrix.png")


def find_optimal_thresholds(y_true, y_scores):
    """
    Sweep θ ∈ [0.01, 0.99] in steps of 0.01 and return:
      - theta_f1:       argmax F1
      - theta_recall80: smallest θ where Recall ≥ 0.80
      - theta_recall90: smallest θ where Recall ≥ 0.90
    along with the full metrics table for plotting.
    """
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    rows = [_scalar_metrics(y_true, (y_scores >= t).astype(int))
            for t in thresholds]

    f1s     = np.array([r["f1"]     for r in rows])
    recalls = np.array([r["recall"] for r in rows])

    best_f1_idx = int(np.argmax(f1s))
    theta_f1    = float(thresholds[best_f1_idx])

    # Thresholds are decreasing in recall as θ increases → search from low θ
    def _first_recall_above(target):
        for i, (t, r) in enumerate(zip(thresholds, recalls)):
            if r >= target:
                return float(t), rows[i]
        return None, None

    # Recall decreases with θ, so iterate reversed
    def _last_recall_above(target):
        result_t, result_m = None, None
        for t, r, row in zip(thresholds, recalls, rows):
            if r >= target:
                result_t, result_m = float(t), row
        return result_t, result_m

    theta_r80, m_r80 = _last_recall_above(0.80)
    theta_r90, m_r90 = _last_recall_above(0.90)

    return {
        "thresholds": thresholds,
        "rows":       rows,
        "theta_f1":   theta_f1,
        "metrics_f1": rows[best_f1_idx],
        "theta_recall80": theta_r80,
        "metrics_recall80": m_r80,
        "theta_recall90": theta_r90,
        "metrics_recall90": m_r90,
    }


def plot_threshold_sweep(y_true, y_scores, sweep, default_theta, out_dir):
    thresholds = sweep["thresholds"]
    rows       = sweep["rows"]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(thresholds, [r["f1"]        for r in rows], lw=1.5, label="F1")
    ax.plot(thresholds, [r["precision"] for r in rows], "--",   label="Precision", lw=1.2)
    ax.plot(thresholds, [r["recall"]    for r in rows], "--",   label="Recall",    lw=1.2)
    ax.plot(thresholds, [r["FAR"]       for r in rows], ":",
            color="tomato", lw=1.2, label="FAR")

    # Chosen firmware threshold
    ax.axvline(0.70, color="steelblue", linestyle="--", lw=0.9,
               label=r"$\theta=0.70$ (firmware)")

    ax.set_xlabel("Detection threshold $\\theta$")
    ax.set_ylabel("Score")
    ax.set_title(r"Metrics vs.\ Detection Threshold (step = 0.01)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    _save(fig, out_dir / "threshold_sweep.png")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", type=Path,
                   help=".tflite binary model file")
    p.add_argument("--model-cc", type=Path,
                   help="C source containing the model byte array "
                        "(default: ../main/person_detect_model_data.cc)")
    p.add_argument("--dataset-dir", type=Path,
                   help="Local VWW root (person/ non_person/ layout)")
    p.add_argument("--split", default="test",
                   help="TFDS split  (default: test)")
    p.add_argument("--theta", type=float, default=0.50,
                   help="Detection threshold (default: 0.50 = firmware default)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit test images for quick runs")
    p.add_argument("--out-dir", type=Path, default=Path("results"),
                   help="Output directory  (default: test/results/)")
    p.add_argument("--no-bootstrap", action="store_true",
                   help="Skip bootstrap CI (much faster)")
    p.add_argument("--demo", action="store_true",
                   help="Use the 10 static project images (pipeline smoke-test, n=10)")
    p.add_argument("--save-scores", type=Path, default=None,
                   metavar="FILE",
                   help="Save inference scores and labels to a .npz file after running")
    p.add_argument("--load-scores", type=Path, default=None,
                   metavar="FILE",
                   help="Load scores and labels from a .npz file, skip inference")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _setup_ieee_style()

    # ── Load model (skipped when --load-scores is used) ──────────────────────
    interp   = None
    model_src = str(args.load_scores) if args.load_scores else None
    if not args.load_scores:
        if args.model:
            model_bytes = _bytes_from_tflite(args.model)
            model_src   = str(args.model)
        elif args.model_cc:
            model_bytes = _bytes_from_cc(args.model_cc)
            model_src   = str(args.model_cc)
        else:
            model_bytes, model_src = _auto_find_model()
        print(f"[model]  {model_src}  ({len(model_bytes)/1024:.1f} KB)")
        interp = build_interpreter(model_bytes)

    # ── Load scores from cache or run inference ───────────────────────────────
    if args.load_scores:
        data   = np.load(args.load_scores)
        scores = data["scores"]
        labels = data["labels"]
        n      = len(labels)
        print(f"[scores] Loaded {n} scores from {args.load_scores}")
    else:
        dataset = None
        if args.demo:
            dataset = _load_static_samples()
            if dataset is None:
                sys.exit("[dataset] --demo: static_images/sample_images/ not found")
        elif args.dataset_dir:
            dataset = _load_dir(args.dataset_dir)
        if dataset is None:
            dataset = _load_tfds(args.split)
        if dataset is None:
            sys.exit(
                "\n[dataset] No dataset available. Options:\n"
                "  --demo                     10 project images (pipeline smoke-test)\n"
                "  --dataset-dir ./vww_data   local person/ non_person/ folder\n"
                "                             (run download_coco_vww.py to build it)\n"
                "  TFDS visual_wake_words     needs COCO pre-downloaded (~780 MB)"
            )

        images, labels = dataset
        if args.max_samples:
            images = images[: args.max_samples]
            labels = labels[: args.max_samples]

        n = len(images)

        def _prefetch(items, buffer_size=64):
            q = queue.Queue(maxsize=buffer_size)
            sentinel = object()

            def producer():
                for item in items:
                    img = np.asarray(Image.open(item).convert("RGB")) if isinstance(item, Path) else item
                    q.put(img)
                q.put(sentinel)

            threading.Thread(target=producer, daemon=True).start()
            while True:
                img = q.get()
                if img is sentinel:
                    break
                yield img

        print(f"[eval]   Running inference on {n} images …")
        scores = np.empty(n, dtype=np.float32)
        for i, img in enumerate(_prefetch(images)):
            if i % 500 == 0:
                print(f"         {i:>5}/{n}", end="\r", flush=True)
            scores[i] = infer(interp, preprocess(img))
        print(f"         {n}/{n}  done.    ")

        if args.save_scores:
            np.savez_compressed(args.save_scores, scores=scores, labels=labels)
            print(f"[scores] Saved to {args.save_scores}")

    y_pred = (scores >= args.theta).astype(int)

    # ── Scalar metrics at default threshold ───────────────────────────────────
    m = _scalar_metrics(labels, y_pred)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    ap      = average_precision_score(labels, scores)

    bar = "=" * 52
    print(f"\n{bar}")
    print(f"  Model   : {Path(model_src).name}")
    print(f"  Dataset : VWW {args.split}  ({n} images)")
    print(f"  θ       : {args.theta:.2f}  (firmware default: 0.50)")
    print(bar)
    print(f"  Accuracy   : {m['accuracy']:.4f}")
    print(f"  Precision  : {m['precision']:.4f}")
    print(f"  Recall     : {m['recall']:.4f}")
    print(f"  F1-score   : {m['f1']:.4f}")
    print(f"  FAR        : {m['FAR']:.4f}")
    print(f"  ROC-AUC    : {roc_auc:.4f}")
    print(f"  Avg Prec   : {ap:.4f}")
    print(f"  TP={m['TP']}  FP={m['FP']}  TN={m['TN']}  FN={m['FN']}")
    print(bar)

    # ── Bootstrap CI ──────────────────────────────────────────────────────────
    ci = {}
    if not args.no_bootstrap:
        print("[boot]   Computing 95 % CIs (n=1000) …")
        ci = _bootstrap_ci(labels, scores, args.theta)
        print(f"  Accuracy  95% CI : [{ci['accuracy_ci'][0]:.4f}, {ci['accuracy_ci'][1]:.4f}]")
        print(f"  F1        95% CI : [{ci['f1_ci'][0]:.4f},       {ci['f1_ci'][1]:.4f}]")
        print(f"  FAR       95% CI : [{ci['FAR_ci'][0]:.4f},      {ci['FAR_ci'][1]:.4f}]")

    # ── Threshold sweep (step = 0.01) ─────────────────────────────────────────
    print("[sweep]  Computing threshold sweep (θ = 0.01 … 0.99) …")
    sweep = find_optimal_thresholds(labels, scores)

    print(f"\n  Optimal operating points:")
    print(f"  {'Point':<22} {'θ':>5}  {'F1':>6}  {'Recall':>7}  {'Prec':>6}  {'FAR':>6}")
    print(f"  {'-'*58}")
    def _row(label, theta, mets):
        if theta is None:
            print(f"  {label:<22}   n/a   (not achievable)")
            return
        print(f"  {label:<22} {theta:>5.2f}  {mets['f1']:>6.4f}  "
              f"{mets['recall']:>7.4f}  {mets['precision']:>6.4f}  {mets['FAR']:>6.4f}")

    _row("max F1",                sweep["theta_f1"],      sweep["metrics_f1"])
    _row("Recall ≥ 0.80",         sweep["theta_recall80"], sweep["metrics_recall80"])
    _row("Recall ≥ 0.90",         sweep["theta_recall90"], sweep["metrics_recall90"])
    _row(f"firmware (θ={args.theta:.2f})", args.theta,    m)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    def _j(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v

    def _mj(mets):
        return {k: _j(v) for k, v in mets.items()} if mets else None

    output = {
        "model": model_src,
        "dataset_split": args.split,
        "n_samples": n,
        "theta_firmware": float(args.theta),
        "metrics_at_firmware_theta": {k: _j(v) for k, v in m.items()},
        "roc_auc": float(roc_auc),
        "avg_precision": float(ap),
        "bootstrap_ci": ci,
        "optimal_thresholds": {
            "theta_max_f1":     sweep["theta_f1"],
            "metrics_max_f1":   _mj(sweep["metrics_f1"]),
            "theta_recall80":   sweep["theta_recall80"],
            "metrics_recall80": _mj(sweep["metrics_recall80"]),
            "theta_recall90":   sweep["theta_recall90"],
            "metrics_recall90": _mj(sweep["metrics_recall90"]),
        },
    }
    json_path = args.out_dir / "metrics.json"
    json_path.write_text(json.dumps(output, indent=2))
    print(f"\n[out]    {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_roc(labels, scores, roc_auc, args.out_dir)
    plot_pr(labels, scores, ap, args.out_dir)
    plot_confusion(labels, scores, [0.26, 0.50, 0.70, 0.92], args.out_dir)
    plot_threshold_sweep(labels, scores, sweep, args.theta, args.out_dir)

    print(f"[out]    plots → {args.out_dir}/\n")
    print("Done.")


if __name__ == "__main__":
    main()
