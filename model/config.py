"""Central configuration for the MobileNetV1 person-detector project."""

# ── Model ────────────────────────────────────────────────────────────────────
IMAGE_SIZE      = 96           # firmware input (H×W)
CHANNELS        = 1            # grayscale (firmware: [1, 96, 96, 1] int8)
ALPHA           = 0.25         # width-multiplier
NUM_CLASSES     = 2            # softmax: [no_person, person]

# ── Dataset ──────────────────────────────────────────────────────────────────
# Local COCO directory (created by download_coco.sh)
import os as _os
COCO_DATA_DIR = _os.path.join(_os.path.dirname(__file__), "coco_data")

# TFDS backend: person (COCO id 1) is label 0 (0-indexed)
PERSON_LABEL    = 0
BATCH_SIZE      = 32

# Set to None to use the full split; set an int for a quick smoke-test
MAX_TRAIN_SAMPLES = 70000
MAX_VAL_SAMPLES   = None
MAX_TEST_SAMPLES  = 40000   # held-out slice of train split, after MAX_TRAIN_SAMPLES

# ── Training ─────────────────────────────────────────────────────────────────
EPOCHS_FLOAT    = 10

LR_FLOAT        = 1e-3
LR_WARMUP       = 1e-4        # first epoch LR

# ── QAT (quantization-aware training) ────────────────────────────────────────
EPOCHS_QAT      = 3
LR_QAT          = 1e-5

# ── Quantization / export ────────────────────────────────────────────────────
NUM_CALIB_SAMPLES = 200       # for representative dataset (PTQ)

# ── Paths ────────────────────────────────────────────────────────────────────
FLOAT_MODEL_PATH   = "checkpoints/float_model.keras"
PTQ_TFLITE_PATH    = "output/person_detect_ptq_int8.tflite"
QAT_TFLITE_PATH    = "output/person_detect_qat_int8.tflite"
C_ARRAY_NAME       = "g_person_detect_model_data"
C_ARRAY_PATH       = "output/g_person_detect_model_data.cc"
C_ARRAY_H_PATH     = "output/g_person_detect_model_data.h"
STATS_PATH         = "output/stats.json"
