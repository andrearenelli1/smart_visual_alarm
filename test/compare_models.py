#!/usr/bin/env python3
"""
compare_models.py — Side-by-side comparison of PTQ vs QAT MobileNetV1 int8 models.

Runs the same evaluation pipeline as eval_vww.py on both models and produces:
  - Console table with all metrics (Accuracy, Precision, Recall, F1, FAR, AUC, AP)
  - Δ column (QAT − PTQ) for each metric
  - Side-by-side comparison plots: ROC, PR, threshold sweep, confusion matrices
  - JSON report saved to <out-dir>/comparison.json

Usage
─────
  # Quick demo (10 static images):
  python compare_models.py --demo

  # Full VWW validation set (needs TFDS / local dataset):
  python compare_models.py

  # Explicit paths:
  python compare_models.py \\
      --ptq models/model_ptq_int8.tflite \\
      --qat models/model_qat_int8.tflite \\
      --dataset-dir ~/datasets/vww

  # Reuse cached scores (skip slow inference):
  python compare_models.py \\
      --load-ptq ptq_scores.npz \\
      --load-qat qat_scores.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Import helpers from eval_vww.py (same directory) ─────────────────────────
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from eval_vww import (
    _setup_ieee_style,
    _bytes_from_tflite,
    _bytes_from_cc,
    _auto_find_model,
    build_interpreter,
    preprocess,
    infer,
    _load_static_samples,
    _load_tfds,
    _load_dir,
    _scalar_metrics,
    _bootstrap_ci,
    find_optimal_thresholds,
)
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from PIL import Image
import queue, threading


# ──────────────────────────────────────────────────────────────────────────────
# COCO loader with offset support
# ──────────────────────────────────────────────────────────────────────────────
def _load_coco(coco_dir: Path, split: str = "train",
               offset: int = 0, max_samples: int | None = None):
    """
    Load COCO 2017 images as (list[Path], np.ndarray[int]) using pycocotools.
    Images are returned as Path objects so the inference runner opens them lazily.

    Layout expected:
      <coco_dir>/train2017/         or val2017/
      <coco_dir>/annotations/instances_train2017.json
    """
    try:
        from pycocotools.coco import COCO
    except ImportError:
        sys.exit("[coco] pycocotools not installed: pip install pycocotools")

    ann_file = coco_dir / "annotations" / f"instances_{split}2017.json"
    img_dir  = coco_dir / f"{split}2017"
    if not ann_file.exists():
        sys.exit(f"[coco] Annotation file not found: {ann_file}")
    if not img_dir.exists():
        sys.exit(f"[coco] Image directory not found: {img_dir}")

    print(f"[coco]   Loading {ann_file.name} …")
    coco = COCO(str(ann_file))

    PERSON_CAT_ID = 1
    person_img_ids = {
        ann["image_id"]
        for ann in coco.dataset["annotations"]
        if ann["category_id"] == PERSON_CAT_ID and not ann.get("iscrowd", 0)
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


def run_inference(interp, images, labels, name: str) -> np.ndarray:
    n = len(images)
    scores = np.empty(n, dtype=np.float32)
    print(f"[eval]   {name}: running inference on {n} images …")
    for i, img in enumerate(_prefetch(images)):
        if i % 500 == 0:
            print(f"         {i:>5}/{n}", end="\r", flush=True)
        scores[i] = infer(interp, preprocess(img))
    print(f"         {n}/{n}  done.    ")
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot]   {path.name}")


def plot_roc_comparison(labels, ptq_scores, qat_scores, out_dir: Path):
    fpr_p, tpr_p, _ = roc_curve(labels, ptq_scores)
    fpr_q, tpr_q, _ = roc_curve(labels, qat_scores)
    auc_p = auc(fpr_p, tpr_p)
    auc_q = auc(fpr_q, tpr_q)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr_p, tpr_p, lw=1.5, label=f"PTQ  AUC={auc_p:.3f}")
    ax.plot(fpr_q, tpr_q, lw=1.5, linestyle="--", label=f"QAT  AUC={auc_q:.3f}")
    ax.plot([0, 1], [0, 1], ":", color="grey", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve — PTQ vs QAT")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "cmp_roc.png")
    return auc_p, auc_q


def plot_pr_comparison(labels, ptq_scores, qat_scores, out_dir: Path):
    prec_p, rec_p, _ = precision_recall_curve(labels, ptq_scores)
    prec_q, rec_q, _ = precision_recall_curve(labels, qat_scores)
    ap_p = average_precision_score(labels, ptq_scores)
    ap_q = average_precision_score(labels, qat_scores)
    baseline = labels.mean()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec_p, prec_p, lw=1.5, label=f"PTQ  AP={ap_p:.3f}")
    ax.plot(rec_q, prec_q, lw=1.5, linestyle="--", label=f"QAT  AP={ap_q:.3f}")
    ax.axhline(baseline, linestyle=":", color="grey", lw=1,
               label=f"Baseline={baseline:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — PTQ vs QAT")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "cmp_pr.png")
    return ap_p, ap_q


def plot_threshold_comparison(labels, ptq_scores, qat_scores, out_dir: Path):
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)

    def _sweep(scores):
        return [_scalar_metrics(labels, (scores >= t).astype(int))
                for t in thresholds]

    rows_p = _sweep(ptq_scores)
    rows_q = _sweep(qat_scores)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, rows, title in zip(axes, [rows_p, rows_q], ["PTQ", "QAT"]):
        ax.plot(thresholds, [r["f1"]        for r in rows], lw=1.5, label="F1")
        ax.plot(thresholds, [r["precision"] for r in rows], "--",   label="Precision", lw=1.2)
        ax.plot(thresholds, [r["recall"]    for r in rows], "--",   label="Recall",    lw=1.2)
        ax.plot(thresholds, [r["FAR"]       for r in rows], ":",
                color="tomato", lw=1.2, label="FAR")
        ax.axvline(0.70, color="steelblue", linestyle="--", lw=0.9,
                   label=r"$\theta=0.70$ (fw)")
        ax.set_title(f"Threshold sweep — {title}")
        ax.set_xlabel(r"$\theta$")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Score")
    fig.tight_layout()
    _save(fig, out_dir / "cmp_threshold_sweep.png")


def plot_confusion_comparison(labels, ptq_scores, qat_scores, theta: float,
                              out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, scores, title in zip(axes,
                                  [ptq_scores, qat_scores],
                                  [f"PTQ  θ={theta:.2f}", f"QAT  θ={theta:.2f}"]):
        pred = (scores >= theta).astype(int)
        m = _scalar_metrics(labels, pred)
        cm = confusion_matrix(labels, pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["no_person", "person"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format='d')
        ax.set_title(f"{title}\nF1={m['f1']:.3f}  FAR={m['FAR']:.3f}", fontsize=9)
    fig.suptitle("Confusion Matrices — PTQ vs QAT", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / "cmp_confusion.png")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Model inputs
    default_ptq = HERE / "models/model_ptq_int8.tflite"
    default_qat = HERE / "models/model_qat_int8.tflite"
    p.add_argument("--ptq",      type=Path, default=default_ptq,
                   help=f"PTQ .tflite (default: {default_ptq})")
    p.add_argument("--qat",      type=Path, default=default_qat,
                   help=f"QAT .tflite (default: {default_qat})")
    # Score cache
    p.add_argument("--load-ptq", type=Path, metavar="FILE",
                   help="Load PTQ scores from .npz (skip inference)")
    p.add_argument("--load-qat", type=Path, metavar="FILE",
                   help="Load QAT scores from .npz (skip inference)")
    p.add_argument("--save-scores", action="store_true",
                   help="Save scores to <out-dir>/ptq_scores.npz and qat_scores.npz")
    # Dataset
    p.add_argument("--dataset-dir", type=Path,
                   help="Local VWW root (person/ non_person/ layout)")
    p.add_argument("--coco-dir", type=Path,
                   help="COCO 2017 root (contains train2017/ and annotations/)")
    p.add_argument("--coco-split", default="train",
                   help="COCO split to use: train or val (default: train)")
    p.add_argument("--coco-offset", type=int, default=0,
                   help="Skip first N images from the COCO split (default: 0)")
    p.add_argument("--split", default="test",
                   help="TFDS split (default: test)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit number of images (applied after --coco-offset)")
    p.add_argument("--demo", action="store_true",
                   help="Use the 10 static project images (pipeline smoke-test)")
    # Eval options
    p.add_argument("--theta", type=float, default=0.70,
                   help="Firmware detection threshold (default: 0.70)")
    p.add_argument("--no-bootstrap", action="store_true",
                   help="Skip bootstrap CI (faster)")
    p.add_argument("--out-dir", type=Path, default=HERE / "results_comparison",
                   help="Output directory (default: test/results_comparison/)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _setup_ieee_style()

    need_inference = not (args.load_ptq and args.load_qat)

    # ── Build interpreters (only if needed) ───────────────────────────────────
    interp_ptq = interp_qat = None
    if not args.load_ptq:
        print(f"[model]  PTQ: {args.ptq}")
        interp_ptq = build_interpreter(_bytes_from_tflite(args.ptq))
    if not args.load_qat:
        print(f"[model]  QAT: {args.qat}")
        interp_qat = build_interpreter(_bytes_from_tflite(args.qat))

    # ── Load dataset (only if inference needed) ───────────────────────────────
    images = labels = None
    if need_inference:
        dataset = None
        if args.demo:
            dataset = _load_static_samples()
            if dataset is None:
                sys.exit("[dataset] --demo: static_images/sample_images/ not found")
        elif args.coco_dir:
            dataset = _load_coco(args.coco_dir, args.coco_split,
                                 offset=args.coco_offset,
                                 max_samples=args.max_samples)
        elif args.dataset_dir:
            dataset = _load_dir(args.dataset_dir)
        if dataset is None:
            dataset = _load_tfds(args.split)
        if dataset is None:
            sys.exit(
                "\n[dataset] No dataset available. Options:\n"
                "  --demo                           10 project images (smoke-test)\n"
                "  --coco-dir /path/to/coco         COCO 2017 with pycocotools\n"
                "  --dataset-dir ./vww_data         person/ non_person/ layout\n"
                "  TFDS visual_wake_words           needs COCO pre-downloaded"
            )
        images, labels = dataset
        if not args.coco_dir and args.max_samples:
            images = images[: args.max_samples]
            labels = labels[: args.max_samples]

    # ── Run / load inference ──────────────────────────────────────────────────
    if args.load_ptq:
        d = np.load(args.load_ptq)
        ptq_scores, labels = d["scores"], d["labels"]
        print(f"[scores] PTQ: loaded {len(labels)} scores from {args.load_ptq}")
    else:
        ptq_scores = run_inference(interp_ptq, images, labels, "PTQ")

    if args.load_qat:
        d = np.load(args.load_qat)
        qat_scores, labels = d["scores"], d["labels"]
        print(f"[scores] QAT: loaded {len(labels)} scores from {args.load_qat}")
    else:
        qat_scores = run_inference(interp_qat, images, labels, "QAT")

    labels = np.asarray(labels, dtype=int)

    if args.save_scores:
        np.savez_compressed(args.out_dir / "ptq_scores.npz",
                            scores=ptq_scores, labels=labels)
        np.savez_compressed(args.out_dir / "qat_scores.npz",
                            scores=qat_scores, labels=labels)
        print(f"[scores] Saved to {args.out_dir}/")

    n = len(labels)

    # ── Metrics ───────────────────────────────────────────────────────────────
    ptq_pred = (ptq_scores >= args.theta).astype(int)
    qat_pred = (qat_scores >= args.theta).astype(int)
    mp = _scalar_metrics(labels, ptq_pred)
    mq = _scalar_metrics(labels, qat_pred)

    auc_p, auc_q = (
        (lambda fpr, tpr: auc(fpr, tpr))(*roc_curve(labels, ptq_scores)[:2]),
        (lambda fpr, tpr: auc(fpr, tpr))(*roc_curve(labels, qat_scores)[:2]),
    )
    ap_p = average_precision_score(labels, ptq_scores)
    ap_q = average_precision_score(labels, qat_scores)

    # ── Bootstrap CI ──────────────────────────────────────────────────────────
    ci_p = ci_q = {}
    if not args.no_bootstrap:
        print("[boot]   Computing 95 % CIs …")
        ci_p = _bootstrap_ci(labels, ptq_scores, args.theta)
        ci_q = _bootstrap_ci(labels, qat_scores, args.theta)

    # ── Console report ────────────────────────────────────────────────────────
    BAR = "=" * 72
    print(f"\n{BAR}")
    print(f"  COMPARISON: PTQ vs QAT — MobileNetV1 int8 on VWW ({n} images)")
    print(f"  θ = {args.theta:.2f} (firmware threshold)")
    print(BAR)
    print(f"  {'Metric':<18} {'PTQ':>10} {'QAT':>10} {'Δ (QAT−PTQ)':>14}")
    print(f"  {'-'*56}")

    def _row(label, key, pval, qval, fmt=".4f"):
        delta = qval - pval
        ps = format(pval, fmt)
        qs = format(qval, fmt)
        ds = ("+" if delta >= 0 else "") + format(delta, fmt)
        print(f"  {label:<18} {ps:>10} {qs:>10}   {ds:>13}")

    _row("Accuracy",  "accuracy",  mp["accuracy"],  mq["accuracy"])
    _row("Precision", "precision", mp["precision"], mq["precision"])
    _row("Recall",    "recall",    mp["recall"],    mq["recall"])
    _row("F1-score",  "f1",        mp["f1"],        mq["f1"])
    _row("FAR",       "FAR",       mp["FAR"],       mq["FAR"])
    _row("ROC-AUC",   "roc_auc",   auc_p,           auc_q)
    _row("Avg Prec",  "ap",        ap_p,            ap_q)
    print(f"  {'-'*56}")
    print(f"  {'TP':<18} {mp['TP']:>10} {mq['TP']:>10} "
          f"  {mq['TP']-mp['TP']:>+14}")
    print(f"  {'FP':<18} {mp['FP']:>10} {mq['FP']:>10} "
          f"  {mq['FP']-mp['FP']:>+14}")
    print(f"  {'TN':<18} {mp['TN']:>10} {mq['TN']:>10} "
          f"  {mq['TN']-mp['TN']:>+14}")
    print(f"  {'FN':<18} {mp['FN']:>10} {mq['FN']:>10} "
          f"  {mq['FN']-mp['FN']:>+14}")
    print(BAR)

    if ci_p and ci_q:
        print(f"\n  95% Bootstrap CIs  (n=1000):")
        print(f"  {'Metric':<18} {'PTQ CI':>22} {'QAT CI':>22}")
        print(f"  {'-'*62}")
        for label, key in [("Accuracy", "accuracy_ci"), ("F1", "f1_ci"), ("FAR", "FAR_ci")]:
            lo_p, hi_p = ci_p[key]
            lo_q, hi_q = ci_q[key]
            print(f"  {label:<18} [{lo_p:.4f}, {hi_p:.4f}]      [{lo_q:.4f}, {hi_q:.4f}]")
        print()

    # ── Optimal thresholds ────────────────────────────────────────────────────
    sweep_p = find_optimal_thresholds(labels, ptq_scores)
    sweep_q = find_optimal_thresholds(labels, qat_scores)
    print(f"  Optimal operating points:")
    print(f"  {'Point':<22} {'PTQ θ':>6} {'PTQ F1':>7} {'QAT θ':>7} {'QAT F1':>7}")
    print(f"  {'-'*52}")
    for label, k_t, k_m in [
        ("max F1",      "theta_f1",      "metrics_f1"),
        ("Recall ≥ 0.80", "theta_recall80", "metrics_recall80"),
        ("Recall ≥ 0.90", "theta_recall90", "metrics_recall90"),
    ]:
        tp = sweep_p[k_t]; mp2 = sweep_p[k_m]
        tq = sweep_q[k_t]; mq2 = sweep_q[k_m]
        tp_s = f"{tp:.2f}" if tp is not None else " n/a"
        tq_s = f"{tq:.2f}" if tq is not None else " n/a"
        fp_s = f"{mp2['f1']:.4f}" if mp2 else "  n/a"
        fq_s = f"{mq2['f1']:.4f}" if mq2 else "  n/a"
        print(f"  {label:<22} {tp_s:>6} {fp_s:>7} {tq_s:>7} {fq_s:>7}")
    print(BAR)

    # ── Plots ─────────────────────────────────────────────────────────────────
    auc_p2, auc_q2 = plot_roc_comparison(labels, ptq_scores, qat_scores, args.out_dir)
    ap_p2, ap_q2   = plot_pr_comparison(labels, ptq_scores, qat_scores, args.out_dir)
    plot_threshold_comparison(labels, ptq_scores, qat_scores, args.out_dir)
    plot_confusion_comparison(labels, ptq_scores, qat_scores, args.theta, args.out_dir)
    print(f"[out]    plots → {args.out_dir}/")

    # ── JSON ──────────────────────────────────────────────────────────────────
    def _j(v):
        if isinstance(v, (np.floating, np.integer)): return v.item()
        return v

    report = {
        "n_samples":   n,
        "theta":       float(args.theta),
        "ptq": {
            "model": str(args.ptq),
            "metrics": {k: _j(v) for k, v in mp.items()},
            "roc_auc": float(auc_p), "avg_precision": float(ap_p),
            "bootstrap_ci": ci_p,
            "optimal": {
                "theta_max_f1":      sweep_p["theta_f1"],
                "metrics_max_f1":    {k: _j(v) for k, v in sweep_p["metrics_f1"].items()},
                "theta_recall80":    sweep_p["theta_recall80"],
                "theta_recall90":    sweep_p["theta_recall90"],
            },
        },
        "qat": {
            "model": str(args.qat),
            "metrics": {k: _j(v) for k, v in mq.items()},
            "roc_auc": float(auc_q), "avg_precision": float(ap_q),
            "bootstrap_ci": ci_q,
            "optimal": {
                "theta_max_f1":      sweep_q["theta_f1"],
                "metrics_max_f1":    {k: _j(v) for k, v in sweep_q["metrics_f1"].items()},
                "theta_recall80":    sweep_q["theta_recall80"],
                "theta_recall90":    sweep_q["theta_recall90"],
            },
        },
        "delta_qat_minus_ptq": {
            k: _j(mq[k]) - _j(mp[k])
            for k in ("accuracy", "precision", "recall", "f1", "FAR")
        } | {"roc_auc": float(auc_q - auc_p), "avg_precision": float(ap_q - ap_p)},
    }

    json_path = args.out_dir / "comparison.json"
    json_path.write_text(json.dumps(report, indent=2))
    print(f"[out]    {json_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
