# Smart Visual Alarm System — `experiment/sram-arena` branch

> **Branch closed — results below.**
> Moves the TFLite tensor arena from PSRAM to internal SRAM and measures the effect on inference latency.

---

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
├── backend/                       # Python notification backend
│   ├── alarm_backend.py           # MQTT subscriber + Telegram + audio
│   ├── live_score_plot.py         # Real-time score visualizer
│   ├── requirements.txt
│   ├── arch_log.csv               # Per-layer latency log (from device)
│   └── stats_log.csv              # Aggregate inference stats log
├── test/                          # Offline evaluation scripts
│   ├── eval_vww.py                # Model evaluation on COCO 2017 VWW
│   ├── download_coco_vww.py       # Dataset downloader
│   ├── plot_stats.py              # Latency plots from CSV logs
│   ├── model_efficiency.py        # TinyML metrics (params, MACs, size)
│   ├── requirements.txt
│   └── results/                   # Evaluation plots and metrics.json
├── static_images/                 # Sample JPEG images for testing
├── CMakeLists.txt
├── partitions.csv
├── sdkconfig.defaults             # Safe build defaults (no credentials)
├── sdkconfig.defaults.esp32s3     # ESP32-S3 specific defaults
├── dependencies.lock
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
| Person detection threshold (%) | Default: 70 |
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

Model evaluated on **123,287 COCO 2017 images** (Visual Wake Words layout).

| θ | Precision | Recall | F1 | FAR | Profile |
|---|-----------|--------|----|-----|---------|
| 0.26 | 68.2% | 86.1% | 0.761 | 47.5% | Max F1 |
| **0.70** | **88.9%** | **49.3%** | **0.634** | **7.3%** | **Firmware default** |
| 0.92 | 98.3% | 20.3% | 0.337 | 0.41% | High precision |

**ROC-AUC = 0.814 — AP = 0.852**

The firmware default θ = 0.70 was chosen empirically: thresholds below ~0.65 produce false alarms on low-texture scenes (blank walls, ceiling). The 3-frame moving-average filter further reduces live false-alarm rate by averaging down isolated spikes.

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

## Model: MobileNetV1 (int8)

| Property | Value |
|----------|-------|
| Input | 96×96 px, grayscale, int8 |
| Depth multiplier α | 0.25 |
| Parameters | 213,272 |
| Model size (int8) | 293.5 KB |
| Estimated MACs | ~7.2 MMAC |
| Tensor arena (PSRAM) | 100 KB |
| Quantization | Post-training linear (per-channel) |

---

## Offline Evaluation

```bash
cd test
pip install -r requirements.txt

# Download COCO 2017 VWW split (~60 GB — takes a while)
python download_coco_vww.py

# Run evaluation
python eval_vww.py

# Regenerate latency plots from CSV logs
python plot_stats.py
```

Results (plots + `metrics.json`) are saved to `test/results/`.

---

## Branches

| Branch | Model | Input | Latency | Notes |
|--------|-------|-------|---------|-------|
| `main` | MobileNetV1 α=0.25 | 96×96 gray | ~351 ms (2.85 FPS) | Production firmware |
| `mobilenet-v2` | MobileNetV2 | 160×160 RGB | ~1919 ms (0.52 FPS) | Latency benchmark only — below alarm-window requirement |

---

## Report

`report_ieee.tex` — IEEE 2-column paper covering architecture, model analysis, hardware latency profiling, and detection quality evaluation on COCO 2017.

Compile with:
```bash
pdflatex report_ieee.tex
```
(Requires `graphicspath` pointing to `test/results/` for embedded figures.)
