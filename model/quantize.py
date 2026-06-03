"""
Quantisation pipeline:

  1. PTQ  – post-training INT8 quantisation → TFLite
  2. QAT  – quantisation-aware training (fine-tune) → TFLite

Compatibility note (TF 2.16+ / Keras 3):
  tfmot ≤ 0.8.x uses `isinstance(model, tf_keras.Model)` AND the
  `_is_graph_network` attribute to detect Functional models.
  tf_keras ≥ 2.16 (used by TF 2.21) no longer sets `_is_graph_network`,
  and `load_model('.keras')` returns a Keras-3 object (not tf_keras).
  Fix: rebuild the model via the tf_keras `build_model()` factory,
  copy weights, and set `_is_graph_network = True`.
"""

import pathlib
import numpy as np
import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot
from config import (
    EPOCHS_QAT, LR_QAT, BATCH_SIZE,
    NUM_CALIB_SAMPLES,
    PTQ_TFLITE_PATH, QAT_TFLITE_PATH,
    IMAGE_SIZE, ALPHA,
)
from model import build_model, recompile


# ── PTQ ───────────────────────────────────────────────────────────────────────

def post_training_quantize(
    model: tf.keras.Model,
    calib_gen,
    output_path: str = PTQ_TFLITE_PATH,
) -> bytes:
    """
    Full INT8 post-training quantisation.

    Parameters
    ----------
    model       : trained float32 Keras model
    calib_gen   : callable → generator yielding [np.ndarray] batches
    output_path : where to save the .tflite file
    """
    print("\n[PTQ] converting to INT8 TFLite …")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations          = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = calib_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type   = tf.int8
    converter.inference_output_type  = tf.int8

    tflite_model = converter.convert()

    _save_tflite(tflite_model, output_path)
    print(f"[PTQ] saved → {output_path}  "
          f"({len(tflite_model)/1e3:.1f} kB)")
    return tflite_model


# ── QAT ───────────────────────────────────────────────────────────────────────

def quantization_aware_training(
    float_model: tf.keras.Model,
    train_ds:    tf.data.Dataset,
    val_ds:      tf.data.Dataset,
    epochs:      int   = EPOCHS_QAT,
    lr:          float = LR_QAT,
) -> tf.keras.Model:
    """
    Apply fake-quantisation nodes to the model, then fine-tune.

    Returns the QAT model (ready for TFLite conversion via qat_to_tflite).
    """
    print("\n[QAT] building tf_keras-compatible model for QAT …")

    # Rebuild via the tf_keras factory so tfmot's isinstance checks pass.
    # (loading a .keras file returns a Keras-3 object, not tf_keras.)
    fresh = build_model(trainable_base=True, learning_rate=lr)
    _transfer_weights(float_model, fresh)

    # tfmot ≤ 0.8.x needs _is_graph_network=True to detect Functional models
    _ensure_graph_network(fresh)

    print("[QAT] wrapping model with fake-quantisation nodes …")
    qat_model = tfmot.quantization.keras.quantize_model(fresh)
    recompile(qat_model, lr=lr)
    qat_model.summary(line_length=80)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=3,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1,
            min_lr=1e-8, verbose=1,
        ),
    ]

    print(f"[QAT] fine-tuning for up to {epochs} epochs …")
    history = qat_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Attach history so callers can plot QAT training curves
    qat_model._qat_history = history.history
    return qat_model


def qat_to_tflite(
    qat_model: tf.keras.Model,
    calib_gen,
    output_path: str = QAT_TFLITE_PATH,
) -> bytes:
    """
    Convert a QAT-trained Keras model to a fully INT8 TFLite model.

    The representative dataset is still used so that activation scales
    are collected (even though weights are already quantised).
    """
    print("\n[QAT→TFLite] stripping fake-Q nodes and converting …")

    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations          = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = calib_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type   = tf.int8
    converter.inference_output_type  = tf.int8

    tflite_model = converter.convert()
    _save_tflite(tflite_model, output_path)
    print(f"[QAT→TFLite] saved → {output_path}  "
          f"({len(tflite_model)/1e3:.1f} kB)")
    return tflite_model


# ── Helpers ───────────────────────────────────────────────────────────────────

def _transfer_weights(src: tf.keras.Model, dst: tf.keras.Model) -> None:
    """
    Copy weights from src → dst by matching variable names.

    Falls back to positional matching if name-based lookup fails,
    which handles the case where src is a Keras-3 model (different
    variable naming) and dst is a tf_keras model.
    """
    src_weights = {w.name: w.numpy() for w in src.weights}
    dst_weights = {w.name: w for w in dst.weights}

    matched, fallback = 0, 0

    # Try name-based matching first; skip layers whose shapes changed
    for name, var in dst_weights.items():
        if name in src_weights:
            src_val = src_weights[name]
            if var.shape != src_val.shape:
                print(f"  [weights] skipping {name}: "
                      f"shape {src_val.shape} → {var.shape} (incompatible)")
                continue
            var.assign(src_val)
            matched += 1

    if matched < len(dst.weights) // 2:
        # Name mismatch (Keras3 vs tf_keras naming) — fall back to positional
        print(f"  [weights] name match: {matched}/{len(dst.weights)} "
              "— falling back to positional copy")
        src_vals = [w.numpy() for w in src.weights]
        for var, val in zip(dst.weights, src_vals):
            var.assign(val)
        fallback = len(dst.weights)
    else:
        print(f"  [weights] transferred {matched}/{len(dst.weights)} by name")


def _ensure_graph_network(model: tf.keras.Model) -> None:
    """
    tfmot ≤ 0.8.x checks `_is_graph_network` to detect Functional models,
    but tf_keras ≥ 2.16 dropped that attribute.
    """
    if not getattr(model, "_is_graph_network", False):
        model._is_graph_network = True


def _save_tflite(data: bytes, path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
