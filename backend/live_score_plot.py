"""
Live person-score plot: raw inference output vs. 3-frame moving average.

Saves the final plot to test/hardware_log/ on exit (Ctrl+C or window close).

Usage:
    python live_score_plot.py
    MQTT_BROKER=192.168.1.10 python live_score_plot.py
"""

import collections
import json
import os
import shutil
import threading
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import paho.mqtt.client as mqtt

MQTT_BROKER       = os.environ.get("MQTT_BROKER",           "localhost")
MQTT_PORT         = int(os.environ.get("MQTT_PORT",         "1883"))
MQTT_SCORES_TOPIC = os.environ.get("MQTT_SCORES_TOPIC",     "alarm/scores")
ALARM_THRESHOLD   = int(os.environ.get("ALARM_THRESHOLD_PCT", "60"))

LOG_DIR = Path(__file__).parent.parent / "test" / "hardware_log"
LOG_DIR.mkdir(parents=True, exist_ok=True)
WINDOW            = 120   # frames to display at once

_lock     = threading.Lock()
_raw      = collections.deque(maxlen=WINDOW)
_filtered = collections.deque(maxlen=WINDOW)
_frame    = [0]   # total frames received (mutable int)


def _ieee_style():
    use_latex = shutil.which("latex") is not None
    base = {
        "font.family":      "serif",
        "axes.labelsize":   9,
        "font.size":        9,
        "legend.fontsize":  8,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "axes.linewidth":   0.6,
        "grid.linewidth":   0.4,
        "lines.linewidth":  1.4,
        "figure.dpi":       110,
    }
    if use_latex:
        base.update({"text.usetex": True,
                     "text.latex.preamble": r"\usepackage{amsmath}"})
    else:
        base.update({"text.usetex": False,
                     "font.serif":       ["STIX Two Text", "STIXGeneral",
                                          "DejaVu Serif"],
                     "mathtext.fontset": "cm"})
    plt.rcParams.update(base)


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        raw  = int(data.get("raw",      0))
        filt = int(data.get("filtered", 0))
        with _lock:
            _raw.append(raw)
            _filtered.append(filt)
            _frame[0] += 1
    except Exception as e:
        print(f"[Error] {e}")


def main():
    _ieee_style()

    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.14)

    (line_raw,)  = ax.plot([], [], color="#4e79a7", lw=1.0, alpha=0.55,
                            label="Raw score")
    (line_filt,) = ax.plot([], [], color="#e15759", lw=1.8,
                            label=r"Moving avg ($w{=}3$)")
    ax.axhline(ALARM_THRESHOLD, color="#59a14f", lw=0.9, ls="--",
               label=f"Threshold ({ALARM_THRESHOLD}\\%)")

    ax.set_ylim(-2, 102)
    ax.set_xlim(0, WINDOW)
    ax.set_ylabel("Person score (\\%)")
    ax.set_xlabel("Frame (last 120)")
    ax.set_title("Live person-detection score --- ESP32-S3-EYE")
    ax.legend(loc="upper left", framealpha=0.85)
    ax.grid(True, alpha=0.28)
    ax.set_axisbelow(True)

    # shaded alarm zone
    ax.axhspan(ALARM_THRESHOLD, 102, color="#59a14f", alpha=0.06, zorder=0)

    status_text = ax.text(
        0.99, 0.97, "waiting...",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7, color="gray", style="italic",
    )

    def update(_):
        with _lock:
            r    = list(_raw)
            f    = list(_filtered)
            total = _frame[0]

        n  = len(r)
        xs = list(range(n))
        line_raw.set_data(xs, r)
        line_filt.set_data(xs[:len(f)], f)

        if n > 0:
            last_raw  = r[-1]
            last_filt = f[-1] if f else 0
            label = (f"frame {total}  |  "
                     f"raw {last_raw}\\%  filtered {last_filt}\\%")
            status_text.set_text(label)
            status_text.set_color("#e15759" if last_filt >= ALARM_THRESHOLD
                                  else "gray")

        return line_raw, line_filt, status_text

    client = mqtt.Client()
    client.on_message = on_message
    print(f"[MQTT] Connecting to {MQTT_BROKER}:{MQTT_PORT} ...")
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.subscribe(MQTT_SCORES_TOPIC)
    print(f"[MQTT] Subscribed to '{MQTT_SCORES_TOPIC}'")
    client.loop_start()

    def on_close(_):
        client.loop_stop()
        if _frame[0] > 0:
            ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
            out = LOG_DIR / f"live_score_{ts}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"[plot] Saved → {out}")

    fig.canvas.mpl_connect("close_event", on_close)

    ani = animation.FuncAnimation(   # noqa: F841
        fig, update, interval=380, blit=True, cache_frame_data=False
    )
    try:
        plt.show()
    except KeyboardInterrupt:
        on_close(None)
    client.loop_stop()


if __name__ == "__main__":
    main()
