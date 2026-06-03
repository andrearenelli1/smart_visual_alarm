"""
Full pipeline
=============

  python pipeline.py [--smoke]

Steps
-----
  1. Load COCO 2017 (train + val), filter person / no-person
  2. Train float32 MobileNetV1 (alpha=0.25) – 2-phase
  3. Evaluate float32 model  → stats_float
  4. PTQ  → INT8 TFLite model
  5. Evaluate PTQ model       → stats_ptq
  6. QAT fine-tune
  7. Evaluate QAT (Keras)     → stats_qat_keras
  8. Export QAT → INT8 TFLite
  9. Evaluate QAT TFLite      → stats_qat_tflite
 10. Export C/C++ array
 11. Print comparison table & save JSON
"""

import argparse
import pathlib
import sys

import tensorflow as tf

from config import (
    BATCH_SIZE, EPOCHS_FLOAT, EPOCHS_QAT,
    NUM_CALIB_SAMPLES,
    FLOAT_MODEL_PATH,
    PTQ_TFLITE_PATH, QAT_TFLITE_PATH,
    C_ARRAY_NAME, C_ARRAY_PATH, C_ARRAY_H_PATH,
    STATS_PATH,
)
from dataset import load_train_val, load_split, make_calibration_generator
from train   import train_float, load_best_float_model
from quantize import (
    post_training_quantize,
    quantization_aware_training,
    qat_to_tflite,
)
from evaluate import (
    evaluate_keras,
    evaluate_tflite,
    print_comparison_table,
    save_all_stats,
)
from c_array import tflite_to_c_array


def parse_args():
    p = argparse.ArgumentParser(description="Person-detector training pipeline")
    p.add_argument(
        "--smoke", action="store_true",
        help="Use tiny dataset subset for quick end-to-end smoke test",
    )
    p.add_argument(
        "--skip-train", action="store_true",
        help="Skip training and load existing checkpoint",
    )
    p.add_argument(
        "--backend", choices=["coco", "synthetic"], default=None,
        help="Dataset backend: 'coco' (real COCO 2017) or 'synthetic' (no download)",
    )
    p.add_argument(
        "--epochs-float", type=int, default=EPOCHS_FLOAT,
    )
    p.add_argument(
        "--epochs-qat", type=int, default=EPOCHS_QAT,
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Backend override ──────────────────────────────────────────────────
    if args.backend:
        import os
        os.environ["DATASET_BACKEND"] = args.backend
        import dataset
        dataset.DATASET_BACKEND = args.backend

    # ── Smoke-test overrides ───────────────────────────────────────────────
    if args.smoke:
        print("\n[smoke] Using tiny dataset subset for quick test")
        import os
        if not args.backend:
            os.environ["DATASET_BACKEND"] = "synthetic"
            import dataset
            dataset.DATASET_BACKEND = "synthetic"
        import config
        config.MAX_TRAIN_SAMPLES = 500
        config.MAX_VAL_SAMPLES   = 200
        config.NUM_CALIB_SAMPLES = 30
        args.epochs_float = 2
        args.epochs_qat   = 1

    # ── GPU check ─────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n[env] GPUs available: {gpus or 'none (using CPU)'}")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # ─────────────────────────────────────────────────────────────────────
    # 1. Dataset
    # ─────────────────────────────────────────────────────────────────────
    print("\n[step 1] Loading COCO 2017 …")
    train_ds, val_ds = load_train_val(batch_size=BATCH_SIZE)

    # ─────────────────────────────────────────────────────────────────────
    # 2. Train float32 model
    # ─────────────────────────────────────────────────────────────────────
    if args.skip_train and pathlib.Path(FLOAT_MODEL_PATH).exists():
        print("\n[step 2] Loading existing float32 checkpoint …")
        float_model = load_best_float_model()
    else:
        print("\n[step 2] Training float32 model …")
        float_model = train_float(
            train_ds, val_ds,
            epochs=args.epochs_float,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 3. Evaluate float32
    # ─────────────────────────────────────────────────────────────────────
    print("\n[step 3] Evaluating float32 model …")
    stats_float = evaluate_keras(float_model, val_ds, label="float32")

    # ─────────────────────────────────────────────────────────────────────
    # 4. PTQ
    # ─────────────────────────────────────────────────────────────────────
    print("\n[step 4] Post-training quantisation (INT8) …")
    calib_gen = make_calibration_generator(NUM_CALIB_SAMPLES)
    post_training_quantize(float_model, calib_gen, PTQ_TFLITE_PATH)

    # ─────────────────────────────────────────────────────────────────────
    # 5. Evaluate PTQ TFLite
    # ─────────────────────────────────────────────────────────────────────
    print("\n[step 5] Evaluating PTQ model …")
    # Use a small validation subset for TFLite (image-by-image is slow on CPU)
    val_ds_small = load_split(
        "validation",
        batch_size=32,
        max_samples=500,
        augment=False,
        balance=False,
    )
    stats_ptq = evaluate_tflite(PTQ_TFLITE_PATH, val_ds_small, label="PTQ_int8")

    # ─────────────────────────────────────────────────────────────────────
    # 6. QAT fine-tune
    # ─────────────────────────────────────────────────────────────────────
    print("\n[step 6] Quantisation-aware training …")
    # Reload best float checkpoint (using tf_keras loader, not Keras-3)
    base_for_qat = load_best_float_model()
    qat_model = quantization_aware_training(
        base_for_qat,
        train_ds,
        val_ds,
        epochs=args.epochs_qat,
    )

    # ─────────────────────────────────────────────────────────────────────
    # 7. Evaluate QAT Keras model
    # ─────────────────────────────────────────────────────────────────────
    print("\n[step 7] Evaluating QAT Keras model …")
    stats_qat_keras = evaluate_keras(qat_model, val_ds, label="QAT_keras")

    # ─────────────────────────────────────────────────────────────────────
    # 8. Export QAT → TFLite
    # ─────────────────────────────────────────────────────────────────────
    print("\n[step 8] Exporting QAT model to TFLite …")
    calib_gen = make_calibration_generator(NUM_CALIB_SAMPLES)
    qat_to_tflite(qat_model, calib_gen, QAT_TFLITE_PATH)

    # ─────────────────────────────────────────────────────────────────────
    # 9. Evaluate QAT TFLite
    # ─────────────────────────────────────────────────────────────────────
    print("\n[step 9] Evaluating QAT TFLite model …")
    stats_qat_tflite = evaluate_tflite(
        QAT_TFLITE_PATH, val_ds_small, label="QAT_int8_tflite"
    )

    # ─────────────────────────────────────────────────────────────────────
    # 10. Generate C array
    # ─────────────────────────────────────────────────────────────────────
    print("\n[step 10] Generating C/C++ array …")
    tflite_to_c_array(
        tflite_path=QAT_TFLITE_PATH,
        output_cc=C_ARRAY_PATH,
        output_h=C_ARRAY_H_PATH,
        array_name=C_ARRAY_NAME,
    )

    # ─────────────────────────────────────────────────────────────────────
    # 11. Summary
    # ─────────────────────────────────────────────────────────────────────
    all_stats = [stats_float, stats_ptq, stats_qat_keras, stats_qat_tflite]
    save_all_stats(all_stats, STATS_PATH)
    print_comparison_table(all_stats)

    print("\n[done] Output files:")
    print(f"  Float checkpoint : {FLOAT_MODEL_PATH}")
    print(f"  PTQ TFLite       : {PTQ_TFLITE_PATH}")
    print(f"  QAT TFLite       : {QAT_TFLITE_PATH}")
    print(f"  C source         : {C_ARRAY_PATH}")
    print(f"  C header         : {C_ARRAY_H_PATH}")
    print(f"  Stats JSON       : {STATS_PATH}")


if __name__ == "__main__":
    main()
