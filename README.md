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

The system is split into three tiers:

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
│   ├── person_detect_model_data.cc # MobileNetV1 int8 model (C array)
│   ├── arch_profiler.h            # Per-layer latency profiling
│   ├── esp_cli.c/.h               # UART CLI for on-demand inference
│   ├── app_camera_esp.c/.h        # Low-level camera driver
│   └── Kconfig.projbuild          # menuconfig options
├── model/                         # Training pipeline (Python)
│   ├── config.py                  # Hyperparameters and paths
│   ├── dataset.py                 # COCO 2017 loader and balancing
│   ├── model.py                   # MobileNetV1 builder
│   ├── train.py                   # Two-phase float training
│   ├── quantize.py                # PTQ and QAT conversion
│   ├── evaluate.py                # Keras + TFLite evaluation
│   ├── pipeline.py                # End-to-end orchestration
│   ├── c_array.py                 # TFLite → C array generator
│   ├── output/                    # Trained models and stats (tracked in git)
│   ├── download_coco.sh           # COCO 2017 downloader
│   └── requirements.txt
├── backend/                       # Python notification backend
│   ├── alarm_backend.py           # MQTT subscriber + Telegram + audio
│   ├── live_score_plot.py         # Real-time score visualizer
│   └── requirements.txt
├── test/                          # Offline evaluation scripts
│   ├── eval_vww.py                # Model evaluation on COCO 2017 VWW
│   ├── compare_models.py          # PTQ vs QAT side-by-side comparison
│   ├── download_coco_vww.py       # Dataset downloader
│   ├── models/                    # TFLite models (original, PTQ, QAT)
│   └── requirements.txt
├── static_images/                 # Sample 96×96 grayscale test images
├── CMakeLists.txt
├── partitions.csv
├── sdkconfig.defaults             # Safe build defaults (no credentials)
└── report_ieee.tex                # IEEE 2-column paper
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
- A Mosquitto broker running on the same network as the board

### Configure

```bash
idf.py menuconfig
```

Under **Application Configuration**, set:

| Option | Description |
|--------|-------------|
| WiFi SSID / Password | Your Wi-Fi network |
| MQTT Broker IP | IP of the machine running Mosquitto |
| MQTT Broker Port | Default: 1883 |
| Person detection threshold (%) | Default: 55 |
| Alarm cooldown (seconds) | Default: 10 |
| Moving-average window (frames) | Default: 3 |

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

---

## MQTT Alarm Payload

Each alarm event is published as a JSON message on `alarm/person`:

```json
{"event_id": 42, "confidence": 0.87, "timestamp_ms": 123456789}
```

---

## Detection Performance

Model: **MobileNetV1 α=0.25 QAT int8**, evaluated on **48,287 held-out COCO 2017
images** (train split offset 70,000 — never seen during training).

| θ | Precision | Recall | F1 | FAR | Notes |
|---|-----------|--------|----|-----|-------|
| 0.45 | 83.3% | 70.3% | 0.763 | 16.6% | Youden-optimal region |
| **0.55** | **86.6%** | **62.3%** | **0.725** | **11.4%** | **Firmware default** |
| 0.65 | 89.7% | 53.5% | 0.670 | 7.3% | Low false-alarm profile |

**ROC-AUC = 0.856 — AP = 0.877**

Compared to the reference pre-trained PTQ model (AUC 0.823) evaluated on the
same held-out set, the QAT model at θ = 0.55 achieves the same F1 (0.725) with
**39% fewer false alarms** (FAR 11.4% vs 17.8%).

The 3-frame moving-average filter applied in firmware further reduces the live
false-alarm rate by smoothing out isolated single-frame spikes.

---

## Inference Latency

Measured on-device over 1,479 consecutive frames (`CONFIG_NN_OPTIMIZED` enabled):

| Layer group | Mean | P99 |
|-------------|------|-----|
| Conv2D (3×3 + all 1×1 PW) | 323.4 ms | 325 ms |
| DepthwiseConv2D | 26.6 ms | 28 ms |
| Other | < 1 ms | — |
| **Total Invoke()** | **351.3 ms** | **353 ms** |

**~2.85 FPS** — compatible with the 1-second alarm window (3-frame average ≈ 1.05 s).

---

## Model: MobileNetV1 (QAT int8)

| Property | Value |
|----------|-------|
| Input | 96×96 px, grayscale, int8 |
| Depth multiplier α | 0.25 |
| Parameters | 213,272 |
| Model size (int8) | ~311 KB |
| Estimated MACs | ~7.2 MMAC |
| Tensor arena (PSRAM) | 100 KB |
| Quantization | Quantization-aware training (QAT) |
| Training set | COCO 2017 train, first 70,000 images |

---

## Training

The full training pipeline lives in `model/`. See [`model/README.md`](model/README.md) for detailed instructions.

```bash
cd model
pip install -r requirements.txt

# Download COCO 2017 (~25 GB)
bash download_coco.sh

# Run the full pipeline: float training → PTQ → QAT → export
python pipeline.py
```

After training, update the firmware model:

```bash
# Copy the generated C array into the firmware source
cp model/output/g_person_detect_model_data.cc main/person_detect_model_data.cc
# Update the #include and length type to match the firmware header if needed
```

---

## Offline Evaluation

```bash
cd test
pip install -r requirements.txt

# Evaluate the firmware model (auto-detected from main/)
python eval_vww.py --coco-dir ../model/coco_data

# Compare PTQ vs QAT side by side
python compare_models.py --coco-dir ../model/coco_data --coco-offset 70000
```

Results (plots + `metrics.json`) are saved to `test/results/`.

---

## Branches

| Branch | Model | Notes |
|--------|-------|-------|
| `main` | MobileNetV1 α=0.25 QAT int8, θ=55% | Production firmware |
| `test/qat-mobilenetv1` | QAT model evaluation and threshold analysis | Development |
| `mobilenet-v2` | MobileNetV2 160×160 RGB | Latency benchmark only (~0.52 FPS) |

---

## Report

`report_ieee.tex` — IEEE 2-column paper covering architecture, model analysis, hardware latency profiling, and detection quality evaluation on COCO 2017.

```bash
pdflatex report_ieee.tex
```
