#!/usr/bin/env python3
"""
compare_models.py — PTQ vs QAT MobileNetV1 int8 evaluation.

Evaluates both models on the same dataset and produces:
  - Console comparison table with Δ column
  - Side-by-side plots: ROC, PR, threshold sweep, confusion matrices
  - JSON report  → <out-dir>/comparison.json
  - (optional) TinyML efficiency CSV → <out-dir>/efficiency.csv  (--efficiency)

Usage
─────
  # Full test set (COCO train images 70 000+, never seen during QAT training):
  python compare_models.py \\
      --coco-dir /path/to/coco \\
      --coco-split train \\
      --coco-offset 70000 \\
      --theta 0.60

  # Reuse cached scores (skip slow inference):
  python compare_models.py \\
      --load-ptq results/comparison/ptq_scores.npz \\
      --load-qat results/comparison/qat_scores.npz

  # Also compute and save TinyML efficiency metrics:
  python compare_models.py ... --efficiency
"""

import argparse
import csv
import json
import queue
import re
import struct
import sys
import threading
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
)
from PIL import Image

# ── TFLite interpreter ────────────────────────────────────────────────────────
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

try:
    import tensorflow_datasets as tfds
    _HAS_TFDS = True
except ImportError:
    _HAS_TFDS = False

# ──────────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────────
def _setup_ieee_style():
    import shutil
    use_latex = shutil.which("latex") is not None
    base = {
        "font.family": "serif", "axes.labelsize": 9, "font.size": 9,
        "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.6, "grid.linewidth": 0.4,
        "lines.linewidth": 1.5, "figure.dpi": 200,
        "text.usetex": False,
    }
    if use_latex:
        base.update({"text.usetex": True,
                     "text.latex.preamble": r"\usepackage{amsmath}"})
    else:
        base.update({"font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "cm"})
    plt.rcParams.update(base)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────
def _bytes_from_tflite(path: Path) -> bytes:
    return path.read_bytes()


def _bytes_from_cc(path: Path) -> bytes:
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
    buf = bytearray(model_bytes)

    def u32(p): return struct.unpack_from('<I', buf, p)[0]
    def i32(p): return struct.unpack_from('<i', buf, p)[0]
    def u16(p): return struct.unpack_from('<H', buf, p)[0]

    def field_pos(table, idx):
        vtable = table - i32(table)
        vtsize = u16(vtable)
        slot = 4 + idx * 2
        if slot >= vtsize:
            return None
        off = u16(vtable + slot)
        return (table + off) if off else None

    def deref(pos): return pos + u32(pos)
    def vec_elem(vec, i): return deref(vec + 4 + i * 4)

    model = deref(0)
    sg_vec_ref = field_pos(model, 2)
    if sg_vec_ref is None:
        return bytes(buf)
    sg_vec = deref(sg_vec_ref)
    patches = 0
    for sg_i in range(u32(sg_vec)):
        sg = vec_elem(sg_vec, sg_i)
        t_vec_ref = field_pos(sg, 0)
        if t_vec_ref is None:
            continue
        t_vec = deref(t_vec_ref)
        for t_i in range(u32(t_vec)):
            tensor = vec_elem(t_vec, t_i)
            q_ref = field_pos(tensor, 4)
            if q_ref is None:
                continue
            quant = deref(q_ref)
            qd_pos = field_pos(quant, 6)
            if qd_pos is None:
                continue
            qd_val = i32(qd_pos)
            if qd_val == 0:
                continue
            sh_ref = field_pos(tensor, 0)
            rank = u32(deref(sh_ref)) if sh_ref else 0
            if qd_val >= rank:
                struct.pack_into('<i', buf, qd_pos, 0)
                patches += 1
    if patches:
        print(f"[model]  Patched {patches} invalid quantized_dimension field(s)")
    return bytes(buf)


def build_interpreter(model_bytes: bytes):
    patched = _fix_quant_dimensions(model_bytes)
    interp = _Interpreter(model_content=patched)
    interp.allocate_tensors()
    return interp


def _auto_find_model() -> tuple[bytes, str]:
    here = Path(__file__).parent
    for p in [here / "../main/person_detect_model_data.cc",
              here / "../../main/person_detect_model_data.cc"]:
        if p.resolve().exists():
            return _bytes_from_cc(p.resolve()), str(p.resolve())
    sys.exit("[model] Cannot auto-locate model. Use --ptq / --qat flags.")


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing  (must match image_provider.cc exactly)
# ──────────────────────────────────────────────────────────────────────────────
IMG_SIZE = 96
_LW = np.array([305.0, 600.0, 119.0]) / 1024.0  # BT.601 luma weights


def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(img_rgb).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32)
    gray = (arr @ _LW).clip(0, 255).astype(np.uint8)
    return (gray.astype(np.int16) - 128).astype(np.int8)


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
def infer(interp, img_int8: np.ndarray) -> float:
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    x = img_int8.reshape(inp_det["shape"]).astype(np.int8)
    interp.set_tensor(inp_det["index"], x)
    interp.invoke()
    raw = interp.get_tensor(out_det["index"])[0]
    zp    = out_det["quantization_parameters"]["zero_points"]
    scale = out_det["quantization_parameters"]["scales"]
    if len(scale) > 0 and scale[0] != 0.0:
        probs = (raw.astype(np.float32) - zp) * scale
    else:
        probs = raw.astype(np.float32) / 128.0
    return float(probs[1])


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────────
def _load_coco(coco_dir: Path | None, split: str = "train",
               offset: int = 0, max_samples: int | None = None,
               img_dir: Path | None = None, ann_file: Path | None = None):
    try:
        from pycocotools.coco import COCO
    except ImportError:
        sys.exit("[coco] pycocotools not installed: pip install pycocotools")

    if img_dir is None:
        img_dir = coco_dir / f"{split}2017"
    if ann_file is None:
        ann_file = coco_dir / "annotations" / f"instances_{split}2017.json"

    if not ann_file.exists():
        sys.exit(f"[coco] Annotation file not found: {ann_file}")
    if not img_dir.exists():
        sys.exit(f"[coco] Image directory not found: {img_dir}")

    print(f"[coco]   Loading {ann_file.name} …")
    coco = COCO(str(ann_file))
    person_img_ids = {
        ann["image_id"]
        for ann in coco.dataset["annotations"]
        if ann["category_id"] == 1 and not ann.get("iscrowd", 0)
    }

    paths, labels = [], []
    for img_info in coco.dataset["images"]:
        p = img_dir / img_info["file_name"]
        if not p.exists():
            continue
        paths.append(p)
        labels.append(1 if img_info["id"] in person_img_ids else 0)

    labels = np.array(labels, dtype=int)
    if offset:
        paths  = paths[offset:]
        labels = labels[offset:]
    if max_samples is not None:
        paths  = paths[:max_samples]
        labels = labels[:max_samples]

    n, pos = len(paths), int(labels.sum())
    print(f"[coco]   {split}[{offset}:{offset + n}]  "
          f"{n} images  (person={pos}, no_person={n-pos})")
    return paths, labels


def _load_tfds(split: str):
    if not _HAS_TFDS:
        return None
    split_map = {"test": "test", "val": "validation", "train": "train"}
    actual_split = split_map.get(split, split)
    try:
        print(f"[dataset] Trying TFDS visual_wake_words ({actual_split}) …")
        ds = tfds.load("visual_wake_words", split=actual_split, shuffle_files=False)
        images, labels = [], []
        for ex in tfds.as_numpy(ds):
            images.append(ex["image"])
            labels.append(1 if ex["label"] == 1 else 0)
        n, pos = len(images), sum(labels)
        print(f"[dataset] visual_wake_words: {n} images  (person={pos}, no_person={n-pos})")
        return images, np.array(labels, dtype=int)
    except Exception as e:
        print(f"[dataset] TFDS unavailable: {str(e).split(chr(10))[0]}")
    return None


def _load_dir(root: Path):
    paths, labels = [], []
    for label, folder in [(1, "person"), (0, "non_person")]:
        d = root / folder
        if not d.exists():
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
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc_b, f1_b, far_b = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = _scalar_metrics(y_true[idx], (y_scores[idx] >= theta).astype(int))
        acc_b.append(m["accuracy"]); f1_b.append(m["f1"]); far_b.append(m["FAR"])
    lo, hi = 0.025, 0.975
    return {
        "accuracy_ci": (float(np.quantile(acc_b, lo)), float(np.quantile(acc_b, hi))),
        "f1_ci":       (float(np.quantile(f1_b,  lo)), float(np.quantile(f1_b,  hi))),
        "FAR_ci":      (float(np.quantile(far_b,  lo)), float(np.quantile(far_b,  hi))),
    }


def find_optimal_thresholds(y_true, y_scores):
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    rows = [_scalar_metrics(y_true, (y_scores >= t).astype(int)) for t in thresholds]
    f1s     = np.array([r["f1"]     for r in rows])
    recalls = np.array([r["recall"] for r in rows])
    best_f1_idx = int(np.argmax(f1s))

    def _last_recall_above(target):
        result_t, result_m = None, None
        for t, r, row in zip(thresholds, recalls, rows):
            if r >= target:
                result_t, result_m = float(t), row
        return result_t, result_m

    theta_r80, m_r80 = _last_recall_above(0.80)
    theta_r90, m_r90 = _last_recall_above(0.90)
    return {
        "thresholds": thresholds, "rows": rows,
        "theta_f1": float(thresholds[best_f1_idx]), "metrics_f1": rows[best_f1_idx],
        "theta_recall80": theta_r80, "metrics_recall80": m_r80,
        "theta_recall90": theta_r90, "metrics_recall90": m_r90,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Efficiency analysis  (from model_efficiency.py)
# ──────────────────────────────────────────────────────────────────────────────
def _prod(shape):
    r = 1
    for s in shape: r *= s
    return r


def analyze_efficiency(model_bytes: bytes, label: str,
                       measured_latency_ms: float = 351.3) -> dict:
    patched = _fix_quant_dimensions(model_bytes)
    interp  = _Interpreter(model_content=patched)
    interp.allocate_tensors()

    input_details  = interp.get_input_details()
    output_details = interp.get_output_details()
    tensor_details = interp.get_tensor_details()
    input_indices  = {d['index'] for d in input_details}

    n_params = 0
    weight_tensors = []
    for t in tensor_details:
        try:
            data = interp.get_tensor(t['index'])
            nelems = _prod(data.shape) if data.shape else 0
            if nelems > 0 and t['index'] not in input_indices:
                weight_tensors.append((t['index'], t['name'], data.shape, nelems))
                n_params += nelems
        except Exception:
            pass

    fps = 1000.0 / measured_latency_ms
    return {
        "label":               label,
        "model_size_int8_kb":  len(model_bytes) / 1024,
        "model_size_fp32_kb":  len(model_bytes) * 4 / 1024,
        "n_params":            n_params,
        "input_shape":         list(input_details[0]['shape']),
        "input_dtype":         str(input_details[0]['dtype']),
        "latency_ms":          measured_latency_ms,
        "fps":                 fps,
        "n_tensors":           len(tensor_details),
        "n_weight_tensors":    len(weight_tensors),
    }


def save_efficiency_csv(ptq_eff: dict, qat_eff: dict, path: Path):
    fields = ["label", "model_size_int8_kb", "model_size_fp32_kb",
              "n_params", "input_shape", "input_dtype",
              "latency_ms", "fps", "n_tensors", "n_weight_tensors"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerow(ptq_eff)
        w.writerow(qat_eff)
    print(f"[out]    {path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Inference runner
# ──────────────────────────────────────────────────────────────────────────────
def _prefetch(items):
    q = queue.Queue(maxsize=64)
    sentinel = object()
    def producer():
        for item in items:
            img = (np.asarray(Image.open(item).convert("RGB"))
                   if isinstance(item, Path) else item)
            q.put(img)
        q.put(sentinel)
    threading.Thread(target=producer, daemon=True).start()
    while True:
        img = q.get()
        if img is sentinel:
            break
        yield img


def run_inference(interp, images, n: int, name: str) -> np.ndarray:
    scores = np.empty(n, dtype=np.float32)
    print(f"[eval]   {name}: running inference on {n} images …")
    for i, img in enumerate(_prefetch(images)):
        if i % 500 == 0:
            print(f"         {i:>5}/{n}", end="\r", flush=True)
        scores[i] = infer(interp, preprocess(img))
    print(f"         {n}/{n}  done.    ")
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Comparison plots
# ──────────────────────────────────────────────────────────────────────────────
def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot]   {path.name}")


def plot_roc_comparison(labels, ptq_scores, qat_scores, out_dir: Path):
    fpr_p, tpr_p, _ = roc_curve(labels, ptq_scores)
    fpr_q, tpr_q, _ = roc_curve(labels, qat_scores)
    auc_p = auc(fpr_p, tpr_p); auc_q = auc(fpr_q, tpr_q)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr_p, tpr_p, lw=1.5, label=f"PTQ  AUC={auc_p:.3f}")
    ax.plot(fpr_q, tpr_q, lw=1.5, ls="--", label=f"QAT  AUC={auc_q:.3f}")
    ax.plot([0, 1], [0, 1], ":", color="grey", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - PTQ vs QAT"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); _save(fig, out_dir / "cmp_roc.png")
    return float(auc_p), float(auc_q)


def plot_pr_comparison(labels, ptq_scores, qat_scores, out_dir: Path):
    prec_p, rec_p, _ = precision_recall_curve(labels, ptq_scores)
    prec_q, rec_q, _ = precision_recall_curve(labels, qat_scores)
    ap_p = average_precision_score(labels, ptq_scores)
    ap_q = average_precision_score(labels, qat_scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec_p, prec_p, lw=1.5, label=f"PTQ  AP={ap_p:.3f}")
    ax.plot(rec_q, prec_q, lw=1.5, ls="--", label=f"QAT  AP={ap_q:.3f}")
    ax.axhline(labels.mean(), ls=":", color="grey", lw=1,
               label=f"Baseline={labels.mean():.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall - PTQ vs QAT"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); _save(fig, out_dir / "cmp_pr.png")
    return float(ap_p), float(ap_q)


def plot_threshold_comparison(labels, ptq_scores, qat_scores,
                               theta: float, out_dir: Path):
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    rows_p = [_scalar_metrics(labels, (ptq_scores >= t).astype(int)) for t in thresholds]
    rows_q = [_scalar_metrics(labels, (qat_scores >= t).astype(int)) for t in thresholds]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, rows, title in zip(axes, [rows_p, rows_q], ["PTQ", "QAT"]):
        ax.plot(thresholds, [r["f1"]        for r in rows], lw=1.5, label="F1")
        ax.plot(thresholds, [r["precision"] for r in rows], "--", label="Precision", lw=1.2)
        ax.plot(thresholds, [r["recall"]    for r in rows], "--", label="Recall",    lw=1.2)
        ax.plot(thresholds, [r["FAR"]       for r in rows], ":", color="tomato",
                lw=1.2, label="FAR")
        ax.axvline(theta, color="steelblue", ls="--", lw=0.9,
                   label=f"theta={theta} (fw)")
        ax.set_title(f"Threshold sweep - {title}"); ax.set_xlabel("theta")
        ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Score")
    fig.tight_layout(); _save(fig, out_dir / "cmp_threshold_sweep.png")


def plot_confusion_comparison(labels, ptq_scores, qat_scores,
                               theta: float, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, scores, title in zip(axes, [ptq_scores, qat_scores],
                                  [f"PTQ  th={theta:.2f}", f"QAT  th={theta:.2f}"]):
        pred = (scores >= theta).astype(int)
        m = _scalar_metrics(labels, pred)
        cm = confusion_matrix(labels, pred, labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["no_person", "person"]).plot(
            ax=ax, colorbar=False, cmap="Blues", values_format='d')
        ax.set_title(f"{title}\nF1={m['f1']:.3f}  FAR={m['FAR']:.3f}", fontsize=9)
    fig.suptitle("Confusion Matrices - PTQ vs QAT", fontsize=10)
    fig.tight_layout(); _save(fig, out_dir / "cmp_confusion.png")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_ptq = Path(__file__).parent / "models/model_ptq_int8.tflite"
    default_qat = Path(__file__).parent / "models/model_qat_int8.tflite"
    p.add_argument("--ptq",      type=Path, default=default_ptq)
    p.add_argument("--qat",      type=Path, default=default_qat)
    p.add_argument("--load-ptq", type=Path, metavar="FILE",
                   help="Load PTQ scores from .npz (skip inference)")
    p.add_argument("--load-qat", type=Path, metavar="FILE",
                   help="Load QAT scores from .npz (skip inference)")
    p.add_argument("--save-scores", action="store_true",
                   help="Save scores to <out-dir>/ptq_scores.npz and qat_scores.npz")
    default_coco = Path(__file__).parent / "coco_cache"
    p.add_argument("--coco-dir",    type=Path, default=default_coco,
                   help=f"COCO root with train2017/ and annotations/ (default: {default_coco})")
    p.add_argument("--coco-split",  default="train")
    p.add_argument("--coco-offset", type=int, default=0)
    p.add_argument("--coco-img-dir", type=Path, default=None,
                   help="Override image directory (for non-standard layouts)")
    p.add_argument("--coco-ann-file", type=Path, default=None,
                   help="Override annotation JSON file (for non-standard layouts)")
    p.add_argument("--dataset-dir", type=Path,
                   help="Local VWW root (person/ non_person/ layout)")
    p.add_argument("--split",       default="test")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--theta",       type=float, default=0.60,
                   help="Firmware detection threshold (default: 0.60)")
    p.add_argument("--no-bootstrap", action="store_true")
    p.add_argument("--efficiency",  action="store_true",
                   help="Compute TinyML efficiency metrics and save to efficiency.csv")
    p.add_argument("--latency-ptq", type=float, default=351.3,
                   help="Measured PTQ latency in ms (default: 351.3, used with --efficiency)")
    p.add_argument("--latency-qat", type=float, default=351.3,
                   help="Measured QAT latency in ms (default: 351.3, used with --efficiency)")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).parent / "results/comparison")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _setup_ieee_style()

    need_inference = not (args.load_ptq and args.load_qat)

    interp_ptq = interp_qat = None
    if not args.load_ptq:
        print(f"[model]  PTQ: {args.ptq}")
        interp_ptq = build_interpreter(_bytes_from_tflite(args.ptq))
    if not args.load_qat:
        print(f"[model]  QAT: {args.qat}")
        interp_qat = build_interpreter(_bytes_from_tflite(args.qat))

    images = labels = None
    if need_inference:
        if args.coco_dir or args.coco_img_dir:
            images, labels = _load_coco(
                args.coco_dir, args.coco_split,
                offset=args.coco_offset, max_samples=args.max_samples,
                img_dir=args.coco_img_dir, ann_file=args.coco_ann_file,
            )
        elif args.dataset_dir:
            result = _load_dir(args.dataset_dir)
            if result:
                images, labels = result
        if images is None:
            result = _load_tfds(args.split)
            if result:
                images, labels = result
        if images is None:
            sys.exit(
                "\n[dataset] No dataset found. Options:\n"
                "  default: test/coco_cache/  (run model/download_coco.sh to populate)\n"
                "  --coco-dir /path/to/coco   standard layout: train2017/ + annotations/\n"
                "  --coco-img-dir <dir> --coco-ann-file <json>  non-standard layout\n"
                "  --dataset-dir ./dir        person/ non_person/ layout\n"
                "  TFDS visual_wake_words     needs COCO pre-downloaded"
            )
        if not args.coco_dir and args.max_samples:
            images = images[:args.max_samples]
            labels = labels[:args.max_samples]

    if args.load_ptq:
        d = np.load(args.load_ptq)
        ptq_scores, labels = d["scores"], d["labels"]
        print(f"[scores] PTQ: loaded {len(labels)} scores from {args.load_ptq}")
    else:
        ptq_scores = run_inference(interp_ptq, images, len(images), "PTQ")

    if args.load_qat:
        d = np.load(args.load_qat)
        qat_scores, labels = d["scores"], d["labels"]
        print(f"[scores] QAT: loaded {len(labels)} scores from {args.load_qat}")
    else:
        qat_scores = run_inference(interp_qat, images, len(images), "QAT")

    labels = np.asarray(labels, dtype=int)
    n = len(labels)

    if args.save_scores:
        np.savez_compressed(args.out_dir / "ptq_scores.npz",
                            scores=ptq_scores, labels=labels)
        np.savez_compressed(args.out_dir / "qat_scores.npz",
                            scores=qat_scores, labels=labels)
        print(f"[scores] Saved to {args.out_dir}/")

    ptq_pred = (ptq_scores >= args.theta).astype(int)
    qat_pred = (qat_scores >= args.theta).astype(int)
    mp = _scalar_metrics(labels, ptq_pred)
    mq = _scalar_metrics(labels, qat_pred)

    fpr_p, tpr_p, _ = roc_curve(labels, ptq_scores)
    fpr_q, tpr_q, _ = roc_curve(labels, qat_scores)
    auc_p = float(auc(fpr_p, tpr_p)); auc_q = float(auc(fpr_q, tpr_q))
    ap_p = float(average_precision_score(labels, ptq_scores))
    ap_q = float(average_precision_score(labels, qat_scores))

    ci_p = ci_q = {}
    if not args.no_bootstrap:
        print("[boot]   Computing 95 % CIs …")
        ci_p = _bootstrap_ci(labels, ptq_scores, args.theta)
        ci_q = _bootstrap_ci(labels, qat_scores, args.theta)

    BAR = "=" * 72
    print(f"\n{BAR}")
    print(f"  COMPARISON: PTQ vs QAT — MobileNetV1 int8  ({n} images, theta={args.theta:.2f})")
    print(BAR)
    print(f"  {'Metric':<18} {'PTQ':>10} {'QAT':>10} {'Delta (QAT-PTQ)':>16}")
    print(f"  {'-'*58}")

    def _row(label, pval, qval, fmt=".4f"):
        delta = qval - pval
        print(f"  {label:<18} {format(pval, fmt):>10} {format(qval, fmt):>10}"
              f"  {('+' if delta >= 0 else '') + format(delta, fmt):>15}")

    _row("Accuracy",  mp["accuracy"],  mq["accuracy"])
    _row("Precision", mp["precision"], mq["precision"])
    _row("Recall",    mp["recall"],    mq["recall"])
    _row("F1-score",  mp["f1"],        mq["f1"])
    _row("FAR",       mp["FAR"],       mq["FAR"])
    _row("ROC-AUC",   auc_p,           auc_q)
    _row("Avg Prec",  ap_p,            ap_q)
    print(BAR)

    sweep_p = find_optimal_thresholds(labels, ptq_scores)
    sweep_q = find_optimal_thresholds(labels, qat_scores)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_roc_comparison(labels, ptq_scores, qat_scores, args.out_dir)
    plot_pr_comparison(labels, ptq_scores, qat_scores, args.out_dir)
    plot_threshold_comparison(labels, ptq_scores, qat_scores, args.theta, args.out_dir)
    plot_confusion_comparison(labels, ptq_scores, qat_scores, args.theta, args.out_dir)

    # ── Efficiency metrics ────────────────────────────────────────────────────
    if args.efficiency:
        print("[eff]    Computing TinyML efficiency metrics …")
        ptq_bytes = _bytes_from_tflite(args.ptq) if args.ptq.exists() else None
        qat_bytes = _bytes_from_tflite(args.qat) if args.qat.exists() else None
        if ptq_bytes and qat_bytes:
            ptq_eff = analyze_efficiency(ptq_bytes, "PTQ_int8", args.latency_ptq)
            qat_eff = analyze_efficiency(qat_bytes, "QAT_int8", args.latency_qat)
            save_efficiency_csv(ptq_eff, qat_eff, args.out_dir / "efficiency.csv")
            print(f"  PTQ: size={ptq_eff['model_size_int8_kb']:.1f} KB  "
                  f"params={ptq_eff['n_params']:,}  fps={ptq_eff['fps']:.2f}")
            print(f"  QAT: size={qat_eff['model_size_int8_kb']:.1f} KB  "
                  f"params={qat_eff['n_params']:,}  fps={qat_eff['fps']:.2f}")
        else:
            print("[eff]    Skipped: --ptq / --qat .tflite files not found")

    # ── JSON ──────────────────────────────────────────────────────────────────
    def _j(v):
        if isinstance(v, (np.floating, np.integer)): return v.item()
        return v

    report = {
        "n_samples": n, "theta": float(args.theta),
        "ptq": {
            "model": str(args.ptq),
            "metrics": {k: _j(v) for k, v in mp.items()},
            "roc_auc": auc_p, "avg_precision": ap_p,
            "bootstrap_ci": ci_p,
            "optimal": {
                "theta_max_f1":   sweep_p["theta_f1"],
                "metrics_max_f1": {k: _j(v) for k, v in sweep_p["metrics_f1"].items()},
                "theta_recall80": sweep_p["theta_recall80"],
                "theta_recall90": sweep_p["theta_recall90"],
            },
        },
        "qat": {
            "model": str(args.qat),
            "metrics": {k: _j(v) for k, v in mq.items()},
            "roc_auc": auc_q, "avg_precision": ap_q,
            "bootstrap_ci": ci_q,
            "optimal": {
                "theta_max_f1":   sweep_q["theta_f1"],
                "metrics_max_f1": {k: _j(v) for k, v in sweep_q["metrics_f1"].items()},
                "theta_recall80": sweep_q["theta_recall80"],
                "theta_recall90": sweep_q["theta_recall90"],
            },
        },
        "delta_qat_minus_ptq": {
            k: _j(mq[k]) - _j(mp[k])
            for k in ("accuracy", "precision", "recall", "f1", "FAR")
        } | {"roc_auc": auc_q - auc_p, "avg_precision": ap_q - ap_p},
    }

    json_path = args.out_dir / "comparison.json"
    json_path.write_text(json.dumps(report, indent=2))
    print(f"[out]    {json_path.name}")
    print(f"[out]    plots → {args.out_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
