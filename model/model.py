"""
MobileNetV1 (alpha=0.25) binary person-detector.

Uses tf_keras explicitly (instead of tf.keras) so that
tensorflow_model_optimization isinstance checks always pass,
even when tensorflow_datasets or other packages import Keras 3
and hijack tf.keras.
"""

import tensorflow as tf
import tf_keras as keras          # tf_keras = legacy Keras 2, used by tfmot
from config import IMAGE_SIZE, CHANNELS, ALPHA, LR_FLOAT

_HEAD_LAYER_NAMES = {"gray_to_rgb", "gap", "dropout", "logits"}


def build_model(
    alpha: float = ALPHA,
    image_size: int = IMAGE_SIZE,
    channels: int = CHANNELS,
    trainable_base: bool = False,
    learning_rate: float = LR_FLOAT,
) -> keras.Model:
    """
    Build a flat tf_keras Functional model: grayscale input → MobileNetV1 backbone → 2-class head.

    Input shape : (image_size, image_size, 1)  grayscale normalized to [-1, 1].
    Output shape: (2,) softmax — index 0 = no_person, index 1 = person.
    The grayscale channel is repeated 3× via Concatenate so ImageNet-pretrained
    weights apply. All layers are direct children (no nested sub-model) so
    tfmot.quantize_model works correctly.
    """
    inputs = keras.Input(
        shape=(image_size, image_size, channels),
        name="input_grayscale",
    )
    # Repeat grayscale channel 3× so ImageNet-pretrained MobileNet weights apply
    x = keras.layers.Concatenate(axis=-1, name="gray_to_rgb")([inputs, inputs, inputs])

    base = keras.applications.MobileNet(
        input_shape=(image_size, image_size, 3),
        alpha=alpha,
        include_top=False,
        weights="imagenet",
    )
    for layer in base.layers:
        layer.trainable = trainable_base

    # Inline base layers as direct children — skip the base InputLayer
    for layer in base.layers[1:]:
        x = layer(x)

    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = keras.layers.Dense(2, activation="softmax", name="logits")(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"MobileNetV1_a{alpha}_fw96_person",
    )
    _compile(model, learning_rate)
    return model


def _compile(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )


def recompile(model: keras.Model, lr: float) -> None:
    _compile(model, lr)


def unfreeze_base(model: keras.Model, lr: float = 1e-4) -> None:
    for layer in model.layers:
        if layer.name not in _HEAD_LAYER_NAMES:
            layer.trainable = True
    _compile(model, lr)


def load_keras_model(path: str) -> keras.Model:
    """Load a saved model as a tf_keras model (not Keras 3)."""
    return keras.models.load_model(path)


def model_summary_str(model: keras.Model) -> str:
    lines = []
    model.summary(print_fn=lambda s: lines.append(s))
    return "\n".join(lines)
