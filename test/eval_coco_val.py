#!/usr/bin/env python3
"""
eval_coco_val.py — Person detection evaluation on COCO val2017

Evaluates the TFLite int8 MobileNetV2 model used by Smart Visual Alarm
firmware.  Preprocessing matches image_provider.cc exactly:
  RGB uint8 → nearest-neighbour resize to 160×160 → subtract 128 → int8

Dataset
───────
  COCO 2017 val set (~5 000 images), read directly from coco_cache/
  without copying.  Run download_coco_vww.py first if not yet cached.

Model
─────
  Auto-detected from ../main/person_detect_model_data.cc, or supply:
    --model       path/to/model.tflite
    --model-cc    path/to/person_detect_model_data.cc

Usage examples
──────────────
  python eval_coco_val.py
  python eval_coco_val.py --theta 0.65
  python eval_coco_val.py --max-samples 500 --no-bootstrap
  python eval_coco_val.py --model my_model.tflite
"""

import argparse
import json
import queue
import re
import sys
import threading
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
)

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
                "  pip install ai-edge-litert"
            )

from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
IMG_SIZE    = 160
PERSON_CLS  = 1     # COCO category id for "person"

HERE        = Path(__file__).parent
COCO_CACHE  = HERE / "coco_cache"
VAL_IMAGES  = COCO_CACHE / "val_images_extracted" / "val2017"
VAL_ANN     = COCO_CACHE / "annotations_extracted" / "annotations" / "instances_val2017.json"


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing  (must match image_provider.cc exactly)
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 H×W×3  →  int8 160×160×3  (firmware pipeline)."""
    pil = Image.fromarray(img_rgb).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    arr = np.asarray(pil, dtype=np.int16)        # 160×160×3
    return (arr - 128).astype(np.int8)           # int8, zero_point = -128


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


def _auto_find_model() -> tuple[bytes, str]:
    candidates_cc  = [HERE / "../main/person_detect_model_data.cc"]
    candidates_tfl = [HERE / "model.tflite", HERE / "../build/model.tflite"]
    for p in candidates_cc:
        if p.resolve().exists():
            return _bytes_from_cc(p.resolve()), str(p.resolve())
    for p in candidates_tfl:
        if p.resolve().exists():
            return _bytes_from_tflite(p.resolve()), str(p.resolve())
    sys.exit(
        "[model] Cannot auto-locate model.\n"
        "  Use --model <file.tflite>  or  --model-cc <person_detect_model_data.cc>"
    )


def build_interpreter(model_bytes: bytes):
    interp = _Interpreter(model_content=model_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"[model] input  shape={inp['shape']}  dtype={inp['dtype']}")
    print(f"[model] output shape={out['shape']}  dtype={out['dtype']}  "
          f"quant=({out['quantization'][0]:.6f}, {out['quantization'][1]})")
    return interp


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
def infer(interp, img_int8: np.ndarray) -> float:
    """Return P(person) ∈ [0, 1]."""
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    interp.set_tensor(inp_det["index"],
                      img_int8.reshape(inp_det["shape"]).astype(np.int8))
    interp.invoke()

    raw   = interp.get_tensor(out_det["index"])[0]          # shape [2]
    zp    = out_det["quantization_parameters"]["zero_points"][0]
    scale = out_det["quantization_parameters"]["scales"][0]

    if scale != 0.0:
        probs = (raw.astype(np.float32) - zp) * scale       # dequantize
    else:
        probs = raw.astype(np.float32) / 128.0

    return float(probs[1])   # P(person) — last op is Softmax


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ──────────────────────────────────────────────────────────────────────────────
def load_dir(root: Path) -> tuple:
    """Load from  root/person/  and  root/non_person/  (no file copy)."""
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
        sys.exit(f"[dataset] No images found under {root}")
    labels = np.array(labels, dtype=int)
    n, pos = len(labels), int(labels.sum())
    print(f"[dataset] {n} images from {root}  (person={pos}, no_person={n-pos})")
    return paths, labels


def load_coco_val(ann_json: Path, images_dir: Path) -> tuple:
    """
    Returns (paths, labels) from COCO val2017 without copying any files.
    label=1 if image contains ≥1 person annotation, else 0.
    """
    if not ann_json.exists():
        sys.exit(
            f"[dataset] Annotations not found: {ann_json}\n"
            f"  Run:  python download_coco_vww.py"
        )
    if not images_dir.exists():
        sys.exit(
            f"[dataset] Images not found: {images_dir}\n"
            f"  Run:  python download_coco_vww.py"
        )

    print(f"[dataset] Loading COCO val2017 annotations …")
    with open(ann_json) as f:
        coco = json.load(f)

    person_ids = {a["image_id"] for a in coco["annotations"]
                  if a["category_id"] == PERSON_CLS}
    id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}

    paths, labels = [], []
    missing = 0
    for img_id, fname in sorted(id_to_name.items()):
        p = images_dir / fname
        if not p.exists():
            missing += 1
            continue
        paths.append(p)
        labels.append(1 if img_id in person_ids else 0)

    if missing:
        print(f"[dataset] Warning: {missing} images listed in annotations not found on disk")

    labels = np.array(labels, dtype=int)
    n, pos = len(labels), int(labels.sum())
    print(f"[dataset] {n} images  (person={pos}, no_person={n-pos})")
    return paths, labels


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def scalar_metrics(y_true, y_pred) -> dict:
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


def bootstrap_ci(y_true, y_scores, theta, n_boot=1000, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    acc_b, f1_b, far_b = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m   = scalar_metrics(y_true[idx], (y_scores[idx] >= theta).astype(int))
        acc_b.append(m["accuracy"]); f1_b.append(m["f1"]); far_b.append(m["FAR"])
    lo, hi = 0.025, 0.975
    return {
        "accuracy_ci": (float(np.quantile(acc_b, lo)), float(np.quantile(acc_b, hi))),
        "f1_ci":       (float(np.quantile(f1_b,  lo)), float(np.quantile(f1_b,  hi))),
        "FAR_ci":      (float(np.quantile(far_b,  lo)), float(np.quantile(far_b,  hi))),
    }


def threshold_sweep(y_true, y_scores):
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    rows       = [scalar_metrics(y_true, (y_scores >= t).astype(int)) for t in thresholds]
    f1s        = np.array([r["f1"]     for r in rows])
    recalls    = np.array([r["recall"] for r in rows])

    best_f1_idx = int(np.argmax(f1s))

    def _last_recall_above(target):
        rt, rm = None, None
        for t, r, row in zip(thresholds, recalls, rows):
            if r >= target:
                rt, rm = float(t), row
        return rt, rm

    r80 = _last_recall_above(0.80)
    r90 = _last_recall_above(0.90)
    return {
        "thresholds":      thresholds,
        "rows":            rows,
        "theta_f1":        float(thresholds[best_f1_idx]),
        "metrics_f1":      rows[best_f1_idx],
        "theta_recall80":  r80,
        "theta_recall90":  r90,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot]   {path.name}")


TITLE_SUFFIX = "MobileNetV2 Q8 · COCO val2017"


def plot_roc(y_true, y_scores, roc_auc, out_dir):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=1.5, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC Curve — {TITLE_SUFFIX}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "roc_curve.png")


def plot_pr(y_true, y_scores, ap, out_dir):
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, lw=1.5, label=f"AP = {ap:.3f}")
    ax.axhline(y_true.mean(), linestyle="--", color="grey", lw=1,
               label=f"Baseline = {y_true.mean():.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {TITLE_SUFFIX}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "pr_curve.png")


def plot_confusion(y_true, y_scores, thetas, out_dir):
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7, 5.5))
    fig.subplots_adjust(wspace=0.12, hspace=0.38)
    for idx, (ax, theta) in enumerate(zip(axes.flat, thetas)):
        row, col = divmod(idx, ncols)
        pred = (y_scores >= theta).astype(int)
        m    = scalar_metrics(y_true, pred)
        cm   = confusion_matrix(y_true, pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["no_person", "person"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format='d')
        ax.set_title(
            f"θ={theta:.2f}\nF1={m['f1']:.3f}  FAR={m['FAR']:.3f}", fontsize=9)
        ax.tick_params(labelsize=8)
        if row < nrows - 1:
            ax.set_xlabel("")
        if col > 0:
            ax.set_ylabel("")
    fig.suptitle(f"Confusion Matrices — {TITLE_SUFFIX}", fontsize=10)
    _save(fig, out_dir / "confusion_matrix.png")


def plot_threshold_sweep(sweep, firmware_theta, out_dir):
    thresholds = sweep["thresholds"]
    rows       = sweep["rows"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(thresholds, [r["f1"]        for r in rows], lw=1.5, label="F1")
    ax.plot(thresholds, [r["precision"] for r in rows], "--",   label="Precision", lw=1.2)
    ax.plot(thresholds, [r["recall"]    for r in rows], "--",   label="Recall",    lw=1.2)
    ax.plot(thresholds, [r["FAR"]       for r in rows], ":",
            color="tomato", lw=1.2, label="FAR")
    ax.axvline(firmware_theta, color="steelblue", linestyle="--", lw=0.9,
               label=f"θ={firmware_theta:.2f} (firmware)")
    ax.set_xlabel("Detection threshold θ")
    ax.set_ylabel("Score")
    ax.set_title(f"Metrics vs. Threshold — {TITLE_SUFFIX}")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
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
    p.add_argument("--model",     type=Path, help=".tflite binary")
    p.add_argument("--model-cc",  type=Path, help="C source with model byte array")
    p.add_argument("--dataset-dir", type=Path, default=None,
                   help="Directory with person/ and non_person/ sub-folders "
                        "(overrides --ann-json / --images-dir)")
    p.add_argument("--ann-json",  type=Path, default=VAL_ANN,
                   help=f"COCO annotations JSON  (default: {VAL_ANN})")
    p.add_argument("--images-dir", type=Path, default=VAL_IMAGES,
                   help=f"COCO val images dir  (default: {VAL_IMAGES})")
    p.add_argument("--theta",       type=float, default=0.50,
                   help="Detection threshold (default: 0.50)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit number of images (for quick runs)")
    p.add_argument("--out-dir",     type=Path,
                   default=HERE / "results" / "mobilenetv2_coco_val",
                   help="Output directory for plots and JSON")
    p.add_argument("--no-bootstrap", action="store_true",
                   help="Skip bootstrap CI (faster)")
    p.add_argument("--save-scores", type=Path, default=None,
                   help="Save scores+labels to .npz after inference")
    p.add_argument("--load-scores", type=Path, default=None,
                   help="Skip inference, load scores from .npz")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    interp     = None
    model_src  = str(args.load_scores) if args.load_scores else None
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

    # ── Load scores or run inference ──────────────────────────────────────────
    if args.load_scores:
        data   = np.load(args.load_scores)
        scores = data["scores"]
        labels = data["labels"]
        print(f"[scores] Loaded {len(labels)} scores from {args.load_scores}")
    else:
        if args.dataset_dir:
            paths, labels = load_dir(args.dataset_dir)
        else:
            paths, labels = load_coco_val(args.ann_json, args.images_dir)
        if args.max_samples:
            paths  = paths[:args.max_samples]
            labels = labels[:args.max_samples]

        n = len(paths)

        def _prefetch(items, buffer_size=64):
            q        = queue.Queue(maxsize=buffer_size)
            sentinel = object()
            def producer():
                for item in items:
                    q.put(np.asarray(Image.open(item).convert("RGB")))
                q.put(sentinel)
            threading.Thread(target=producer, daemon=True).start()
            while True:
                img = q.get()
                if img is sentinel:
                    break
                yield img

        print(f"[eval]   Running inference on {n} images …")
        scores = np.empty(n, dtype=np.float32)
        for i, img in enumerate(_prefetch(paths)):
            if i % 200 == 0:
                print(f"         {i:>4}/{n}", end="\r", flush=True)
            scores[i] = infer(interp, preprocess(img))
        print(f"         {n}/{n}  done.    ")

        if args.save_scores:
            np.savez_compressed(args.save_scores, scores=scores, labels=labels)
            print(f"[scores] Saved to {args.save_scores}")

    y_pred  = (scores >= args.theta).astype(int)
    m       = scalar_metrics(labels, y_pred)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    ap      = average_precision_score(labels, scores)
    n       = len(labels)
    dataset_label = (str(args.dataset_dir) if args.dataset_dir
                     else "COCO val2017")

    bar = "=" * 56
    print(f"\n{bar}")
    print(f"  Model   : {Path(model_src).name}")
    print(f"  Dataset : {dataset_label}  ({n} images)")
    print(f"  θ       : {args.theta:.2f}")
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
        ci = bootstrap_ci(labels, scores, args.theta)
        print(f"  Accuracy  95% CI : [{ci['accuracy_ci'][0]:.4f}, {ci['accuracy_ci'][1]:.4f}]")
        print(f"  F1        95% CI : [{ci['f1_ci'][0]:.4f},       {ci['f1_ci'][1]:.4f}]")
        print(f"  FAR       95% CI : [{ci['FAR_ci'][0]:.4f},      {ci['FAR_ci'][1]:.4f}]")

    # ── Threshold sweep ───────────────────────────────────────────────────────
    print("[sweep]  Computing threshold sweep …")
    sweep = threshold_sweep(labels, scores)

    r80_t, r80_m = sweep["theta_recall80"]
    r90_t, r90_m = sweep["theta_recall90"]

    print(f"\n  Optimal operating points:")
    print(f"  {'Point':<22} {'θ':>5}  {'F1':>6}  {'Recall':>7}  {'Prec':>6}  {'FAR':>6}")
    print(f"  {'-'*58}")
    def _row(label, theta, mets):
        if theta is None:
            print(f"  {label:<22}   n/a   (not achievable)")
            return
        print(f"  {label:<22} {theta:>5.2f}  {mets['f1']:>6.4f}  "
              f"{mets['recall']:>7.4f}  {mets['precision']:>6.4f}  {mets['FAR']:>6.4f}")
    _row("max F1",         sweep["theta_f1"],  sweep["metrics_f1"])
    _row("Recall ≥ 0.80",  r80_t, r80_m)
    _row("Recall ≥ 0.90",  r90_t, r90_m)
    _row(f"firmware (θ={args.theta:.2f})", args.theta, m)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    def _j(v):
        return v.item() if isinstance(v, (np.floating, np.integer)) else v
    def _mj(mets):
        return {k: _j(v) for k, v in mets.items()} if mets else None

    output = {
        "model":      model_src,
        "dataset":    dataset_label,
        "n_samples":  n,
        "theta":      float(args.theta),
        "metrics":    {k: _j(v) for k, v in m.items()},
        "roc_auc":    float(roc_auc),
        "avg_precision": float(ap),
        "bootstrap_ci":  ci,
        "optimal_thresholds": {
            "theta_max_f1":    sweep["theta_f1"],
            "metrics_max_f1":  _mj(sweep["metrics_f1"]),
            "theta_recall80":  r80_t,
            "metrics_recall80": _mj(r80_m),
            "theta_recall90":  r90_t,
            "metrics_recall90": _mj(r90_m),
        },
    }
    json_path = args.out_dir / "metrics.json"
    json_path.write_text(json.dumps(output, indent=2))
    print(f"\n[out]    {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_roc(labels, scores, roc_auc, args.out_dir)
    plot_pr(labels, scores, ap, args.out_dir)
    plot_confusion(labels, scores, [0.25, 0.50, 0.70, 0.90], args.out_dir)
    plot_threshold_sweep(sweep, args.theta, args.out_dir)
    print(f"[out]    plots → {args.out_dir}/\n")
    print("Done.")


if __name__ == "__main__":
    main()
