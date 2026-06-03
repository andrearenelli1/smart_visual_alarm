# MobileNetV1 α=0.25 — Person Detector (COCO 2017)

Binary classifier: **person** / **no-person** — trained from scratch on COCO 2017,
PTQ-quantised, QAT-fine-tuned, and exported as a fully INT8 TFLite model ready
for deployment on ESP32-S3 via TensorFlow Lite for Microcontrollers.

---

## Architecture

```
Input (96×96×1, int8 — grayscale)
  └─ gray_to_rgb  (Concatenate ×3 → 96×96×3)
       └─ MobileNetV1 backbone  α=0.25
            └─ GlobalAveragePooling2D
                 └─ Dropout 0.3
                      └─ Dense(2, softmax)
                              │
                    index 0 = P(no_person)
                    index 1 = P(person)
```

Parameters: ~213 k (α=0.25) — INT8 size: ~311 KB

---

## Pipeline Overview

```
COCO 2017 train split
  └─ first 70,000 images → balanced person / no-person batches

Phase 1 — Float training  (train.py)
  1a. Frozen backbone, train head only        (~5 epochs)
  1b. Unfreeze all layers, fine-tune at low LR (~5 epochs)
  → checkpoint: checkpoints/float_model.keras

Phase 2 — PTQ  (quantize.py)
  Representative dataset calibration (200 images)
  → output/person_detect_ptq_int8.tflite

Phase 3 — QAT  (quantize.py)
  Fine-tune with fake-quantisation nodes       (~3 epochs, LR=1e-5)
  → output/person_detect_qat_int8.tflite  ← deployed in firmware

Export  (c_array.py)
  → output/g_person_detect_model_data.cc / .h  ← copy to main/
  → output/stats.json                          ← all evaluation metrics
```

---

## Quick Start

### 1 — Install dependencies (Python 3.10+)

```bash
pip install -r requirements.txt
```

### 2 — Download COCO 2017

```bash
bash download_coco.sh        # downloads to coco_data/ (~25 GB)
```

### 3 — Run the full pipeline

```bash
# Full training + PTQ + QAT + export
python pipeline.py

# Smoke-test (tiny subset, 2 epochs) — runs in minutes on CPU
python pipeline.py --smoke

# Skip float training if checkpoint already exists
python pipeline.py --skip-train
```

All outputs are written to `output/`.

---

## Output Files

| File | Description | Tracked in git |
|------|-------------|---------------|
| `output/person_detect_ptq_int8.tflite` | INT8 PTQ model | ✓ |
| `output/person_detect_qat_int8.tflite` | INT8 QAT model (deployed) | ✓ |
| `output/g_person_detect_model_data.cc` | C array for firmware | ✓ |
| `output/g_person_detect_model_data.h`  | C header for firmware | ✓ |
| `output/stats.json` | All evaluation metrics | ✓ |
| `checkpoints/float_model.keras` | Float32 Keras checkpoint | ✗ |
| `coco_data/` | COCO 2017 images (~25 GB) | ✗ |
| `logs/` | TensorBoard training logs | ✗ |

---

## Deploying to Firmware

After training, copy the generated C array into the firmware source:

```bash
# From the repo root
cp model/output/g_person_detect_model_data.cc main/person_detect_model_data.cc
```

The file may need two small adaptations to match the existing firmware header:

```c
// Change the include from:
#include "g_person_detect_model_data.h"
// to:
#include "person_detect_model_data.h"

// Change the length declaration from:
const unsigned int g_person_detect_model_data_len = ...;
// to:
const int g_person_detect_model_data_len = ...;
```

Then rebuild and flash the firmware:

```bash
idf.py build && idf.py -p /dev/ttyUSB0 flash
```

---

## Evaluation Metrics

Metrics are collected at every stage (float, PTQ, QAT-Keras, QAT-TFLite) and
written to `output/stats.json`.

| Metric | Description |
|--------|-------------|
| accuracy | Binary, at Youden-optimal threshold |
| precision / recall | Per-class (person, no-person) |
| F1 macro | Unweighted mean of both classes |
| AUC-ROC | Area under ROC curve |
| confusion matrix | TN / FP / FN / TP |
| threshold_optimal | Youden's J: argmax(TPR − FPR) |
| inference_ms_per_img | Wall-clock time per image |
| model_size_mb | File size on disk |

To re-evaluate an existing model without retraining:

```bash
python evaluate.py
```

---

## Performance on Held-Out Set

Evaluated on COCO 2017 train[70,000:] — **48,287 images never seen during training**.

| Model | θ | Precision | Recall | F1 | FAR | AUC |
|-------|---|-----------|--------|----|-----|-----|
| Original PTQ (reference) | 0.50 | 81.4% | 66.4% | 0.731 | 17.8% | 0.823 |
| **QAT (deployed)** | **0.55** | **86.6%** | **62.3%** | **0.725** | **11.4%** | **0.856** |
| QAT | 0.45 | 83.3% | 70.3% | 0.763 | 16.6% | 0.856 |

At the same F1 (0.725), the QAT model produces **39% fewer false alarms** compared
to the reference PTQ model.

---

## Project Structure

```
model/
├── config.py          — hyperparameters and all path constants
├── dataset.py         — COCO loader, class balancing, calibration generator
├── model.py           — MobileNetV1 builder and compile helpers
├── train.py           — two-phase float training loop
├── quantize.py        — PTQ and QAT conversion
├── evaluate.py        — Keras + TFLite evaluation, stats.json writer
├── pipeline.py        — end-to-end orchestration (entry point)
├── run_training.py    — thin wrapper around pipeline.py
├── c_array.py         — TFLite → C/H source generator
├── plot_style.py      — shared Matplotlib/IEEE style helpers
├── replot.py          — regenerate plots from existing stats.json
├── test_external.py   — evaluation on an external test dataset
├── download_coco.sh   — COCO 2017 downloader script
├── requirements.txt
└── output/            — generated artifacts (see table above)
```
