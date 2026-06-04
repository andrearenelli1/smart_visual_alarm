# Smart Visual Alarm System

TinyML person-detection alarm on the **ESP32-S3-EYE**: fully on-device inference with MobileNetV1 via TensorFlow Lite for Microcontrollers, MQTT event publishing, and real-time Telegram notifications.

Raw video frames **never leave the device** — only compact JSON alarm descriptors are transmitted over the network.

---

## System Architecture

```
┌──────────────────────────── ESP32-S3-EYE Firmware ────────────────────────────┐
│  OV2640 Camera → TFLite Inference → Moving-Average Filter → Detection         │
│                                                              Responder ────┐   │
└───────────────────────────────────────────────────────────────────────────┼───┘
                                                                            │ JSON (MQTT QoS1)
                                                                            ▼
                                                               Mosquitto MQTT Broker
                                                                            │
                                                                            ▼
                                                               Python Backend → Telegram Bot
```

| Tier | Component | Role |
|------|-----------|------|
| Firmware | ESP32-S3-EYE | Capture → infer → filter → publish |
| Broker | Mosquitto | MQTT message bus |
| Backend | Python script | Subscribe → notify → play audio alert |

---

## Repository Structure

```
smart_visual_alarm/
├── main/                          # ESP-IDF firmware (C/C++)
│   ├── main.cc                    # Entry point, FreeRTOS task setup
│   ├── main_functions.cc/.h       # Inference loop (init + run)
│   ├── image_provider.cc/.h       # OV2640 frame capture + preprocessing
│   ├── detection_responder.cc/.h  # Alarm logic, LED blink, MQTT publish
│   ├── mqtt_publisher.c/.h        # MQTT QoS1 client (esp-mqtt)
│   ├── wifi_manager.c/.h          # Wi-Fi connection management
│   ├── lcd_display.c/.h           # ST7789 viewfinder (optional)
│   ├── model_settings.h           # Input resolution, tensor arena size
│   ├── person_detect_model_data.cc # MobileNetV1 QAT int8 model (C array)
│   ├── arch_profiler.h            # Per-layer latency profiling
│   ├── esp_cli.c/.h               # UART CLI for on-demand inference
│   ├── app_camera_esp.c/.h        # Low-level camera driver
│   └── Kconfig.projbuild          # menuconfig options (threshold, cooldown, Wi-Fi…)
├── model/                         # Training pipeline (Python)
│   ├── config.py                  # Hyperparameters and paths
│   ├── dataset.py                 # COCO 2017 loader and class balancing
│   ├── model.py                   # MobileNetV1 builder
│   ├── train.py                   # Two-phase float training
│   ├── quantize.py                # PTQ and QAT conversion
│   ├── evaluate.py                # Keras + TFLite evaluation
│   ├── pipeline.py                # End-to-end orchestration
│   ├── run_training.py            # Standalone training entry point
│   ├── c_array.py                 # TFLite → C array generator
│   ├── replot.py                  # Re-generate evaluation plots from saved scores
│   ├── plot_style.py              # Shared matplotlib style
│   ├── test_external.py           # Quick sanity-check on an external image
│   ├── output/                    # Trained models and stats (tracked in git)
│   ├── download_coco.sh           # COCO 2017 downloader
│   └── requirements.txt
├── backend/                       # Python notification backend
│   ├── alarm_backend.py           # MQTT subscriber + Telegram + audio
│   ├── live_score_plot.py         # Real-time person-score visualizer (MQTT)
│   ├── live_latency_plot.py       # Real-time inference latency visualizer (MQTT)
│   ├── plot_threshold_sweep_stacked.py  # Threshold-sweep figure generator
│   ├── split_latency_plots.py     # Per-block latency figure generator
│   └── requirements.txt
├── test/                          # Offline evaluation scripts
│   ├── compare_models.py          # PTQ vs QAT side-by-side comparison
│   ├── models/                    # TFLite models (original PTQ, QAT)
│   ├── results/                   # Plots and metrics output
│   │   └── comparison/            # PTQ/QAT overlay plots (ROC, PR, confusion)
│   ├── hardware_log/              # On-device latency and score CSV/PNG logs
│   └── requirements.txt
├── CMakeLists.txt
├── partitions.csv
├── sdkconfig.defaults             # Safe build defaults (no credentials)
└── report_ieee.tex                # IEEE 2-column paper source
```

---

## Hardware

- **Board**: [ESP32-S3-EYE](https://github.com/espressif/esp-who) (Espressif)
- **SoC**: ESP32-S3, dual-core Xtensa LX7 @ 240 MHz
- **SRAM**: 512 KB on-chip + 8 MB OPI PSRAM
- **Camera**: OV2640 (up to 2 MP)
- **Display**: 1.3-inch ST7789 LCD (240×240)

---

## Firmware Setup

### Prerequisites

- [ESP-IDF v5.x](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/)
- A Mosquitto broker reachable from the board's Wi-Fi network

### Configure

```bash
idf.py menuconfig
```

Under **Application Configuration**:

| Option | Default | Description |
|--------|---------|-------------|
| WiFi SSID / Password | — | Your Wi-Fi network |
| MQTT Broker IP | — | IP of the Mosquitto host |
| MQTT Broker Port | 1883 | |
| Person detection threshold (%) | 55 | Alarm trigger threshold θ |
| Alarm cooldown (seconds) | 10 | Minimum interval between alarms |
| Moving-average window (frames) | 5 | Smoothing window size |

### Build and Flash

```bash
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

---

## Backend Setup

```bash
cd backend
pip install -r requirements.txt

export TELEGRAM_BOT_TOKEN=<your_bot_token>
export TELEGRAM_CHAT_ID=<your_chat_id>
export MQTT_BROKER=localhost        # or the broker's IP
export MQTT_PORT=1883
export MQTT_TOPIC=alarm/person

python alarm_backend.py
```

The backend subscribes to `alarm/person`, plays a local WAV alert, and sends a Telegram message with event ID, confidence score, and timestamp.

Live plots (score and latency) can be launched independently:

```bash
python backend/live_score_plot.py
python backend/live_latency_plot.py
```

---

## MQTT Alarm Payload

```json
{"event_id": 42, "confidence": 0.87, "timestamp_ms": 123456789}
```

Published on `alarm/person` with MQTT QoS 1 (at-least-once delivery).

---

## Detection Performance

Model: **MobileNetV1 α=0.25 QAT int8**, evaluated on **48,287 held-out COCO 2017
images** (train split offset 70,000 — never seen during training).

| Model | AUC-ROC | AP | Precision | Recall | F1 | FAR |
|-------|---------|----|-----------|--------|----|-----|
| PTQ int8 (stock) | 0.823 | 0.858 | 85.6% | 58.4% | 0.695 | 11.6% |
| **QAT int8 (ours)** | **0.856** | **0.877** | **88.4%** | 57.5% | **0.697** | **8.9%** |

All values at **θ = 0.60** (evaluation threshold used in the paper;
firmware default is θ = 0.55).

The 3-frame moving-average filter applied in firmware further reduces the live
false-alarm rate by smoothing isolated single-frame spikes.

---

## Inference Latency

Measured on-device over n = 104 consecutive frames (`CONFIG_NN_OPTIMIZED` enabled,
Xtensa LX7 @ 240 MHz, tensor arena in PSRAM):

| Layer group | Mean | Std |
|-------------|------|-----|
| Conv2D (3×3 opening + all 1×1 PW) | 367.8 ms | 1.04 ms |
| DepthwiseConv2D | 22.3 ms | 0.38 ms |
| Other (GAP, QUANTIZE, FC, Softmax) | 13.4 ms | 0.21 ms |
| **Total Invoke()** | **403.5 ms** | **1.12 ms** |

**~2.48 FPS** effective throughput. Conv2D accounts for ≈91% of invoke time;
early depthwise-separable blocks (DS1–3) on larger feature maps are up to 3×
slower than later ones (DS6–13).

---

## Model: MobileNetV1 QAT int8

| Property | Value |
|----------|-------|
| Input | 96×96 px, grayscale, int8 |
| Depth multiplier α | 0.25 |
| Parameters | 213,272 |
| Model size (int8) | 304.2 KB |
| Estimated MACs | ≈7.5 MMAC |
| Tensor arena (PSRAM) | 100 KB |
| Quantization | Quantization-aware training (QAT) |
| Training set | COCO 2017 train, first 70,000 images |

---

## Training

The full training pipeline lives in `model/`. See [`model/README.md`](model/README.md) for detailed instructions.

```bash
cd model
pip install -r requirements.txt

# Download COCO 2017 train split (~25 GB)
bash download_coco.sh

# Run the full pipeline: float training → PTQ → QAT → export C array
python pipeline.py
```

After training, copy the generated C array into the firmware:

```bash
cp model/output/g_person_detect_model_data.cc main/person_detect_model_data.cc
cp model/output/g_person_detect_model_data.h  main/person_detect_model_data.h
```

---

## Offline Evaluation

```bash
cd test
pip install -r requirements.txt

# Compare PTQ vs QAT on held-out COCO images (offset 70,000)
python compare_models.py \
    --coco-dir ../test/coco_cache \
    --coco-split train \
    --coco-offset 70000 \
    --theta 0.60

# Reuse cached scores (skip slow inference)
python compare_models.py \
    --load-ptq results/comparison/ptq_scores.npz \
    --load-qat results/comparison/qat_scores.npz
```

Results (plots + `comparison.json`) are saved to `test/results/`.

---

## Report

`report_ieee.tex` — IEEE 2-column paper covering architecture, model analysis,
hardware latency profiling, and detection quality evaluation on COCO 2017.

```bash
pdflatex report_ieee.tex
```
