# MobileNetV1 α=0.25 – Person Detector (COCO 2017)

Binary classifier: **person** / **no-person** — trained, PTQ-quantised,
QAT-fine-tuned and exported as a fully INT8 TFLite model with a C array.

---

## Architecture

```
Input (224×224×3, float32 / int8)
  └─ MobileNetV1 backbone  α=0.25, pretrained ImageNet
       └─ GlobalAveragePooling2D
            └─ Dropout 0.3
                 └─ Dense(1, sigmoid)   ← person probability
```

Parameters: ~470 k (α=0.25)  
Float32 size: ~1.9 MB  
INT8 size:    ~480 kB  

---

## Pipeline

```
COCO 2017
  ├─ train2017 (118k imgs)  →  balanced person/no-person
  └─ val2017   (  5k imgs)  →  evaluation

Phase 1 – Float training (2 phases)
  Phase 1a: frozen backbone, train head
  Phase 1b: unfreeze all, fine-tune at low LR
  → evaluate → stats_float

Phase 2 – PTQ (post-training INT8 quantisation)
  representative dataset calibration (200 val images)
  → evaluate TFLite → stats_ptq

Phase 3 – QAT (quantisation-aware training)
  fine-tune 5 epochs with fake-Q nodes
  → evaluate Keras QAT → stats_qat_keras
  → export INT8 TFLite
  → evaluate TFLite    → stats_qat_tflite

Export
  → model_qat_int8.tflite
  → model_qat_int8.cc  (C array)
  → model_qat_int8.h   (header)
  → stats.json         (all metrics)
```

---

## Quick start

```bash
# 1 – Install deps (Python 3.10+)
pip install -r requirements.txt

# 2 – Smoke test (tiny subset, 2 epochs) – runs in minutes on CPU
python pipeline.py --smoke

# 3 – Full training
python pipeline.py

# Skip re-training if checkpoint exists
python pipeline.py --skip-train
```

COCO 2017 is downloaded automatically by `tensorflow_datasets` (~25 GB for
train, ~1 GB for val).  
Set `MAX_TRAIN_SAMPLES` / `MAX_VAL_SAMPLES` in `config.py` to cap the size.

---

## Metrics collected at every stage

| Metric                | Description                              |
|-----------------------|------------------------------------------|
| accuracy              | binary (threshold 0.5)                   |
| precision / recall    | per class (person, no-person)            |
| F1 macro              | unweighted mean of both classes          |
| AUC-ROC               | area under ROC curve                     |
| confusion matrix      | TN / FP / FN / TP                        |
| inference_ms_per_img  | wall-clock time per image                |
| model_size_mb         | file size or param-count × 4 bytes       |

---

## Output files

| File                           | Description                      |
|--------------------------------|----------------------------------|
| `checkpoints/float_model.keras`| best float32 Keras checkpoint    |
| `output/model_ptq_int8.tflite` | INT8 PTQ TFLite model            |
| `output/model_qat_int8.tflite` | INT8 QAT TFLite model (best)     |
| `output/model_qat_int8.cc`     | C source with model bytes        |
| `output/model_qat_int8.h`      | C header                         |
| `output/stats.json`            | all evaluation stats (JSON)      |

---

## Using the C array (TFLM / embedded)

```c
#include "model_qat_int8.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

// The model data is in model_qat_int8.cc:
//   const unsigned char person_detect_model_data[];
//   const unsigned int  person_detect_model_data_len;

const tflite::Model* model =
    tflite::GetModel(person_detect_model_data);
```

Input tensor: `int8[1, 224, 224, 3]`  
Output tensor: `int8[1, 1]`  — dequantise with the tensor's scale/zero-point
to get the sigmoid probability ∈ [0, 1].

---

## Project structure

```
person_detector/
├── config.py          – all hyper-parameters & paths
├── dataset.py         – COCO loader, balancing, calibration gen
├── model.py           – MobileNetV1 builder + compile helpers
├── train.py           – two-phase float training loop
├── quantize.py        – PTQ and QAT conversion
├── evaluate.py        – Keras + TFLite evaluation, comparison table
├── c_array.py         – TFLite → C/H source generator
├── pipeline.py        – end-to-end orchestration script
└── requirements.txt
```
