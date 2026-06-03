"""
COCO 2017 → binary person / no-person dataset builder.

Three backends (set DATASET_BACKEND env var or config):
  "local"     : fast local loader via pycocotools + tf.data (recommended)
  "coco"      : tensorflow_datasets auto-download (huge, ~25 GB)
  "synthetic" : random data, no download (unit-tests / smoke-test)

For the "local" backend run first:
    bash download_coco.sh          # full (~19 GB)
    bash download_coco.sh --val-only  # val only (~1 GB, for quick test)
"""

import os
import pathlib
import tensorflow as tf
import tf_keras as keras
import numpy as np

try:
    import tensorflow_datasets as tfds
    _TFDS_AVAILABLE = True
except ImportError:
    _TFDS_AVAILABLE = False

try:
    from pycocotools.coco import COCO
    _COCO_AVAILABLE = True
except ImportError:
    _COCO_AVAILABLE = False

from config import (
    IMAGE_SIZE, CHANNELS, PERSON_LABEL, BATCH_SIZE,
    MAX_TRAIN_SAMPLES, MAX_VAL_SAMPLES, MAX_TEST_SAMPLES,
    COCO_DATA_DIR,
)

DATASET_BACKEND = os.environ.get("DATASET_BACKEND", "local").lower()
COCO_PERSON_CATEGORY_ID = 1   # COCO category id (1-indexed, not 0-indexed)

# BT.601 luma weights — identical to image_provider.cc:
#   gray = (305*R + 600*G + 119*B) >> 10
_BT601 = tf.constant([305.0 / 1024.0, 600.0 / 1024.0, 119.0 / 1024.0],
                     dtype=tf.float32)


def _rgb_to_gray_bt601(image: tf.Tensor) -> tf.Tensor:
    """float32 [H, W, 3] in [0,255] → float32 [H, W, 1] in [0,255], BT.601 luma."""
    gray = tf.tensordot(image, _BT601, axes=[[2], [0]])  # [H, W]
    return tf.expand_dims(tf.clip_by_value(gray, 0.0, 255.0), axis=-1)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess_image(path: tf.Tensor, label: tf.Tensor, augment: bool = False):
    """Load JPEG → BT.601 grayscale → resize → normalize [-1, 1] → (image, label)."""
    raw   = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.cast(image, tf.float32)
    image = _rgb_to_gray_bt601(image)                  # [H, W, 1]
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])

    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        image = tf.clip_by_value(image, 0.0, 255.0)

    image = image / 127.5 - 1.0    # [-1, 1]
    return image, label


# ── LOCAL backend (pycocotools) ───────────────────────────────────────────────

def _build_local_lists(split: str):
    """
    Parse COCO JSON annotations and return two lists:
      paths  – absolute path to each image file (str)
      labels – 1.0 if image contains a person, else 0.0 (float32)
    """
    if not _COCO_AVAILABLE:
        raise RuntimeError(
            "pycocotools not installed. Run: pip install pycocotools"
        )

    coco_dir = pathlib.Path(COCO_DATA_DIR)
    ann_file = coco_dir / "annotations" / f"instances_{split}2017.json"
    img_dir  = coco_dir / f"{split}2017"

    if not ann_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {ann_file}\n"
            f"Run:  bash download_coco.sh"
        )
    if not img_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {img_dir}\n"
            f"Run:  bash download_coco.sh"
        )

    print(f"  [local] loading {ann_file.name} …", flush=True)
    coco = COCO(str(ann_file))

    # image IDs that contain at least one person
    person_img_ids = set(
        ann["image_id"]
        for ann in coco.dataset["annotations"]
        if ann["category_id"] == COCO_PERSON_CATEGORY_ID
        and not ann.get("iscrowd", 0)
    )

    paths, labels = [], []
    for img_info in coco.dataset["images"]:
        img_id   = img_info["id"]
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            continue                        # skip missing files
        paths.append(str(img_path))
        labels.append(1 if img_id in person_img_ids else 0)

    paths  = np.array(paths,  dtype=object)
    labels = np.array(labels, dtype=np.int32)

    n_person    = int(labels.sum())
    n_no_person = len(labels) - n_person
    print(f"  [local] {split}: {len(paths)} images  "
          f"(person={n_person}, no-person={n_no_person})", flush=True)
    return paths, labels


def _local_split(
    split: str,
    batch_size: int,
    max_samples: int | None,
    augment: bool,
    balance: bool,
    offset: int = 0,
) -> tf.data.Dataset:
    paths, labels = _build_local_lists(split)

    if offset > 0:
        paths  = paths[offset:]
        labels = labels[offset:]

    if max_samples is not None:
        paths  = paths[:max_samples]
        labels = labels[:max_samples]

    if balance and split == "train":
        paths, labels = _balance_arrays(paths, labels)

    ds = tf.data.Dataset.from_tensor_slices(
        (paths.astype(str), labels)
    )

    if split == "train":
        ds = ds.shuffle(buffer_size=min(len(paths), 10_000),
                        reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, l: _preprocess_image(p, l, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _balance_arrays(paths: np.ndarray, labels: np.ndarray):
    """Oversample the minority class to equalise class counts."""
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    n = max(len(pos_idx), len(neg_idx))
    if len(pos_idx) < n:
        pos_idx = np.resize(pos_idx, n)       # repeat positives
    elif len(neg_idx) < n:
        neg_idx = np.resize(neg_idx, n)

    idx = np.concatenate([pos_idx, neg_idx])
    np.random.shuffle(idx)
    return paths[idx], labels[idx]


# ── TFDS backend ──────────────────────────────────────────────────────────────

def _label_from_objects(objects) -> tf.Tensor:
    labels     = objects["label"]          # 0-indexed in TFDS (person = 0)
    has_person = tf.reduce_any(tf.equal(labels, PERSON_LABEL))
    return tf.cast(has_person, tf.int32)


def _preprocess_coco_tfds(example, augment: bool = False):
    image = tf.cast(example["image"], tf.float32)
    image = _rgb_to_gray_bt601(image)                  # [H, W, 1]
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        image = tf.clip_by_value(image, 0.0, 255.0)
    image = image / 127.5 - 1.0
    label = _label_from_objects(example["objects"])
    return image, label


def _tfds_balance(ds: tf.data.Dataset, seed: int = 42) -> tf.data.Dataset:
    person_ds    = ds.filter(lambda x, y: tf.equal(y, 1))
    no_person_ds = ds.filter(lambda x, y: tf.equal(y, 0))
    return tf.data.experimental.sample_from_datasets(
        [person_ds, no_person_ds], weights=[0.5, 0.5], seed=seed
    )


def _tfds_split(
    split: str,
    batch_size: int,
    max_samples: int | None,
    augment: bool,
    balance: bool,
    cache: bool,
    offset: int = 0,
) -> tf.data.Dataset:
    if not _TFDS_AVAILABLE:
        raise RuntimeError("tensorflow_datasets not installed.")

    shuffle = (split == "train")
    raw_ds  = tfds.load("coco/2017", split=split,
                        shuffle_files=shuffle, as_supervised=False)

    if offset > 0:
        raw_ds = raw_ds.skip(offset)

    if max_samples is not None:
        raw_ds = raw_ds.take(max_samples)

    ds = raw_ds.map(
        lambda ex: _preprocess_coco_tfds(ex, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if balance and split == "train":
        ds = _tfds_balance(ds)
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=2048, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ── SYNTHETIC backend ─────────────────────────────────────────────────────────

def _synthetic_split(
    split: str,
    n_samples: int,
    batch_size: int,
    augment: bool,
) -> tf.data.Dataset:
    rng    = np.random.default_rng(0 if split == "train" else 1)
    images = rng.standard_normal((n_samples, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)).astype(np.float32)
    labels = np.array([i % 2 for i in range(n_samples)], dtype=np.int32)
    print(f"  [dataset] synthetic mode: {n_samples} samples ({split})")
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ── Public API ────────────────────────────────────────────────────────────────

def load_split(
    split: str = "train",
    batch_size: int = BATCH_SIZE,
    max_samples: int | None = None,
    augment: bool = False,
    balance: bool = True,
    cache: bool = False,
    offset: int = 0,
) -> tf.data.Dataset:
    """
    Load a split as a batched tf.data.Dataset.

    Backend selected by DATASET_BACKEND env var:
      'local'     -> pycocotools + local JPEG files (recommended)
      'coco'      -> tensorflow_datasets (auto-download ~25 GB)
      'synthetic' -> random data (no download, testing only)

    offset: skip the first N images before applying max_samples.
            Used by load_test_set() to avoid overlap with training data.
    """
    backend = DATASET_BACKEND

    if backend == "synthetic":
        n = max_samples if max_samples is not None else (1000 if split == "train" else 200)
        return _synthetic_split(split, n, batch_size, augment)

    if backend == "coco":
        return _tfds_split(split, batch_size, max_samples, augment, balance, cache,
                           offset=offset)

    # default: local
    coco_split = "val" if split == "validation" else split
    return _local_split(coco_split, batch_size, max_samples, augment, balance,
                        offset=offset)


def load_train_val(
    batch_size: int = BATCH_SIZE,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train_ds = load_split(
        "train",
        batch_size=batch_size,
        max_samples=MAX_TRAIN_SAMPLES,
        augment=True,
        balance=True,
    )
    val_ds = load_split(
        "validation",
        batch_size=batch_size,
        max_samples=MAX_VAL_SAMPLES,
        augment=False,
        balance=False,
    )
    return train_ds, val_ds


def load_test_set(batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    """
    Load MAX_TEST_SAMPLES images from the COCO train split that were
    never seen during training.

    Training consumes paths[0 : MAX_TRAIN_SAMPLES].
    This function loads paths[MAX_TRAIN_SAMPLES : MAX_TRAIN_SAMPLES + MAX_TEST_SAMPLES],
    so there is zero overlap with the training data.
    No shuffle, no augmentation, no class balancing.
    """
    offset = MAX_TRAIN_SAMPLES if MAX_TRAIN_SAMPLES is not None else 0
    return load_split(
        "train",
        batch_size=batch_size,
        offset=offset,
        max_samples=MAX_TEST_SAMPLES,
        augment=False,
        balance=False,
    )


def make_calibration_generator(num_samples: int = 200):
    """Generator for TFLite representative dataset (PTQ calibration)."""
    ds = load_split(
        "validation",
        batch_size=1,
        max_samples=num_samples,
        augment=False,
        balance=False,
    )

    def generator():
        for images, _ in ds.take(num_samples):
            yield [images.numpy()]

    return generator
