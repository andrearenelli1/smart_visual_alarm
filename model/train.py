"""Training loop for the float32 MobileNetV1 person-detector."""

import pathlib
import tensorflow as tf
import tf_keras as keras
from config import (
    EPOCHS_FLOAT, LR_FLOAT, LR_WARMUP,
    BATCH_SIZE, FLOAT_MODEL_PATH,
)
from model import build_model, unfreeze_base, load_keras_model


def get_callbacks(checkpoint_path: str) -> list:
    pathlib.Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def train_float(
    train_ds: tf.data.Dataset,
    val_ds:   tf.data.Dataset,
    epochs:   int = EPOCHS_FLOAT,
) -> tf.keras.Model:
    """
    Two-phase training:
      Phase 1 – frozen backbone, train only the head.
      Phase 2 – unfreeze backbone and fine-tune at lower LR.
    """
    print("\n[train] Phase 1 – head only (backbone frozen)")
    model = build_model(trainable_base=False, learning_rate=LR_FLOAT)
    model.summary(line_length=80)

    phase1_epochs = max(1, epochs // 2)
    phase2_epochs = epochs - phase1_epochs

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=get_callbacks(FLOAT_MODEL_PATH),
        verbose=1,
    )

    print("\n[train] Phase 2 – fine-tune full network (backbone unfrozen)")
    unfreeze_base(model, lr=LR_WARMUP)

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs + phase2_epochs,
        initial_epoch=phase1_epochs,
        callbacks=get_callbacks(FLOAT_MODEL_PATH),
        verbose=1,
    )

    return model


def load_best_float_model() -> keras.Model:
    return load_keras_model(FLOAT_MODEL_PATH)
