"""
Smart Visual Alarm - Python backend
Subscribes to MQTT alarm and scores topics.
- Alarm events  → plays sound + Telegram notification + logged to hardware_log/alarms_log.csv
- Scores events → logged to console + hardware_log/scores_log.csv
                  (live plot: live_score_plot.py)

Usage:
    export TELEGRAM_BOT_TOKEN="your_token_here"
    export TELEGRAM_CHAT_ID="your_chat_id_here"
    python alarm_backend.py

To get your chat_id: send any message to your bot, then open:
    https://api.telegram.org/bot<TOKEN>/getUpdates
"""

import csv
import json
import math
import os
import struct
import subprocess
import tempfile
import threading
import wave
import datetime
from pathlib import Path

import requests
import paho.mqtt.client as mqtt

# --- Configuration (override via environment variables) ---
TELEGRAM_BOT_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN",  "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID    = os.environ.get("TELEGRAM_CHAT_ID",    "YOUR_CHAT_ID")
MQTT_BROKER         = os.environ.get("MQTT_BROKER",         "localhost")
MQTT_PORT           = int(os.environ.get("MQTT_PORT",       "1883"))
MQTT_TOPIC          = os.environ.get("MQTT_TOPIC",          "alarm/person")
MQTT_SCORES_TOPIC   = os.environ.get("MQTT_SCORES_TOPIC",   "alarm/scores")

LOG_DIR = Path(__file__).parent.parent / "test" / "hardware_log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_SCORES_CSV = LOG_DIR / "scores_log.csv"
_ALARMS_CSV = LOG_DIR / "alarms_log.csv"

def _append_csv(path: Path, row: dict):
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Sound
# ---------------------------------------------------------------------------

def _play_sound_worker():
    sample_rate = 44100
    freq = 880
    beeps = [(0.15, 1.0), (0.15, 0.0), (0.15, 1.0), (0.15, 0.0), (0.4, 1.0)]

    samples = []
    for duration, amplitude in beeps:
        n = int(sample_rate * duration)
        for i in range(n):
            v = int(32767 * amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
            samples.append(struct.pack('<h', v))

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        with wave.open(tmp.name, 'w') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(b''.join(samples))

        for cmd in [["aplay", "-q", tmp.name], ["paplay", tmp.name]]:
            try:
                subprocess.run(cmd, timeout=5, check=True, capture_output=True)
                return
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue

        print('\a', end='', flush=True)
    except Exception as e:
        print(f"[Sound] Error: {e}")
        print('\a', end='', flush=True)
    finally:
        os.unlink(tmp.name)


def play_alarm_sound():
    threading.Thread(target=_play_sound_worker, daemon=True).start()


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def send_telegram(message: str) -> None:
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN":
        print(f"[Telegram] (not configured) would send: {message}")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=5)
        resp.raise_for_status()
        print(f"[Telegram] Sent: {message}")
    except Exception as e:
        print(f"[Telegram] Error: {e}")


# ---------------------------------------------------------------------------
# MQTT callbacks
# ---------------------------------------------------------------------------

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Connected to {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(MQTT_TOPIC)
        print(f"[MQTT] Subscribed to '{MQTT_TOPIC}'")
        client.subscribe(MQTT_SCORES_TOPIC)
        print(f"[MQTT] Subscribed to '{MQTT_SCORES_TOPIC}'")
    else:
        print(f"[MQTT] Connection failed (rc={rc})")


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())

        ts_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if msg.topic == MQTT_SCORES_TOPIC:
            raw  = data.get("raw",      0)
            filt = data.get("filtered", 0)
            print(f"[Score] raw={raw}%  filtered={filt}%")
            _append_csv(_SCORES_CSV, {"timestamp": ts_str, "raw": raw, "filtered": filt})
            return

        # alarm/person
        event_id   = data.get("event_id", "?")
        confidence = data.get("confidence", 0.0)

        print(f"[ALARM] event={event_id}  confidence={confidence*100:.0f}%  time={ts_str}")
        _append_csv(_ALARMS_CSV, {"timestamp": ts_str, "event_id": event_id,
                                   "confidence": f"{confidence:.4f}"})
        play_alarm_sound()

        text = (
            f"ALARM: Person detected!\n"
            f"Event #{event_id} | Confidence: {confidence*100:.0f}%\n"
            f"Time: {ts_str}"
        )
        send_telegram(text)

    except Exception as e:
        print(f"[Error] Bad message on {msg.topic}: {e}  payload={msg.payload}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"[MQTT] Connecting to {MQTT_BROKER}:{MQTT_PORT} ...")
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    main()
