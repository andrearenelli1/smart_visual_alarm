"""
Live inference-latency plot: two-column bar chart updated in real time.

Subscribes to alarm/stats (published by the firmware when ARCH_PROFILER
is enabled). Each message is a JSON with per-block DW/PW timing:
  {"total":351.3,"conv":5.1,"dw":[...],"pw":[...],"gap":0.1}

Left column  : Conv2D total (opening conv + all PW) vs DW Conv total
Right column : per-block mean latency (Conv3x3 + DS1-DS13 + GAP)

Saves a timestamped PNG and a CSV to test/hardware_log/ on exit.

Usage:
    python live_latency_plot.py
    MQTT_BROKER=192.168.1.10 python live_latency_plot.py
"""

import collections
import csv
import json
import os
import shutil
import threading
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import paho.mqtt.client as mqtt

MQTT_BROKER       = os.environ.get("MQTT_BROKER",       "localhost")
MQTT_PORT         = int(os.environ.get("MQTT_PORT",     "1883"))
MQTT_STATS_TOPIC  = os.environ.get("MQTT_STATS_TOPIC",  "alarm/stats")
MAX_SAMPLES       = int(os.environ.get("LATENCY_WINDOW", "200"))

LOG_DIR = Path(__file__).parent.parent / "test" / "hardware_log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "conv":  "#4e79a7",
    "dw":    "#f28e2b",
    "other": "#bab0ac",
}

_lock    = threading.Lock()
_samples = collections.deque(maxlen=MAX_SAMPLES)
_count   = [0]


# ── Data model ────────────────────────────────────────────────────────────────
def _empty_sample():
    return {"total": 0, "conv": 0,
            "dw": [0] * 13, "pw": [0] * 13, "gap": 0, "other": 0}


def _mean_sample(samples):
    if not samples:
        return _empty_sample()
    n = len(samples)
    m = _empty_sample()
    for s in samples:
        m["total"] += s["total"]
        m["conv"]  += s["conv"]
        m["gap"]   += s["gap"]
        m["other"] += s["other"]
        for i in range(13):
            m["dw"][i] += s["dw"][i]
            m["pw"][i] += s["pw"][i]
    for k in ("total", "conv", "gap", "other"):
        m[k] /= n
    for i in range(13):
        m["dw"][i] /= n
        m["pw"][i] /= n
    return m


# ── MQTT ──────────────────────────────────────────────────────────────────────
def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        sample = {
            "total": float(data.get("total", 0)),
            "conv":  float(data.get("conv",  0)),
            "dw":    [float(x) for x in data.get("dw", [0] * 13)],
            "pw":    [float(x) for x in data.get("pw", [0] * 13)],
            "gap":   float(data.get("gap",   0)),
            "other": float(data.get("other", 0)),
        }
        with _lock:
            _samples.append(sample)
            _count[0] += 1
    except Exception as e:
        print(f"[Error] {e}")


# ── Style ─────────────────────────────────────────────────────────────────────
def _ieee_style():
    use_latex = shutil.which("latex") is not None
    base = {
        "font.family": "serif", "axes.labelsize": 9, "font.size": 9,
        "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.6, "grid.linewidth": 0.4,
        "lines.linewidth": 1.5, "figure.dpi": 110,
        "text.usetex": False,
    }
    if use_latex:
        base.update({"text.usetex": True,
                     "text.latex.preamble": r"\usepackage{amsmath}"})
    else:
        base.update({"font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "cm"})
    plt.rcParams.update(base)


# ── Plots ─────────────────────────────────────────────────────────────────────
def _draw_left(ax, m):
    """Left: Conv2D total vs DW Conv total."""
    ax.clear()
    conv_total = m["conv"] + sum(m["pw"])
    dc_total   = sum(m["dw"])
    values = [conv_total, dc_total]
    labels = ["Conv2D\n(3x3 + all 1x1)", "DW Conv\n(3x3)"]
    colors = [COLORS["conv"], COLORS["dw"]]
    bars = ax.barh(labels, values, color=colors, alpha=0.85, height=0.5)
    for bar, v in zip(bars, values):
        if v > 0:
            ax.text(v + 1, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f} ms", va="center", ha="left", fontsize=7.5)
    ax.set_xlabel("Mean latency (ms)")
    ax.set_title("Layer-type breakdown")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    if max(values) > 0:
        ax.set_xlim(0, max(values) * 1.25)


def _draw_right(ax, m, n):
    """Right: per-block bar chart."""
    ax.clear()
    values = [m["conv"]]
    for i in range(13):
        values.append(m["dw"][i] + m["pw"][i])
    values.append(m["gap"])
    values.append(m["other"])

    labels = ["Conv 3x3"] + [f"DS{i}" for i in range(1, 14)] + ["GAP", "Other"]
    bar_colors = [COLORS["conv"]] + [COLORS["dw"]] * 13 + [COLORS["other"], COLORS["other"]]

    y = range(len(values))
    ax.barh(list(y), values, color=bar_colors, alpha=0.85, height=0.7)
    for yi, v in zip(y, values):
        if v >= 0.5:
            ax.text(v + 0.3, yi, f"{v:.1f}", va="center", ha="left", fontsize=6.5)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean latency (ms)")
    ax.set_title(f"Per-block latency (n={n})")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    if max(values) > 0:
        ax.set_xlim(0, max(values) * 1.2)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    _ieee_style()

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9, 5))
    fig.suptitle("Live Inference Latency — ESP32-S3-EYE", fontsize=10)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=0.12, wspace=0.35)

    status = fig.text(0.5, 0.01, "Waiting for data...", ha="center",
                      fontsize=7, color="gray", style="italic")

    def update(_):
        with _lock:
            snap = list(_samples)
            n    = _count[0]
        if not snap:
            return
        m = _mean_sample(snap)
        _draw_left(ax_left, m)
        _draw_right(ax_right, m, n)
        status.set_text(f"total mean: {m['total']:.1f} ms  |  samples: {n}")
        status.set_color("#4e79a7")
        fig.canvas.draw_idle()

    def on_close(_):
        client.loop_stop()
        with _lock:
            snap = list(_samples)
            n    = _count[0]
        if not snap:
            print("[exit] No data received, nothing saved.")
            return
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save PNG
        out_png = LOG_DIR / f"latency_{ts}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved -> {out_png}")

        # Save CSV (one row per sample)
        out_csv = LOG_DIR / f"latency_{ts}.csv"
        fields = (["total_ms", "conv_opening_ms"] +
                  [f"dw{i+1}_ms" for i in range(13)] +
                  [f"pw{i+1}_ms" for i in range(13)] +
                  ["gap_ms", "other_ms"])
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for s in snap:
                row = {"total_ms": s["total"], "conv_opening_ms": s["conv"],
                       "gap_ms": s["gap"], "other_ms": s["other"]}
                for i in range(13):
                    row[f"dw{i+1}_ms"] = s["dw"][i]
                    row[f"pw{i+1}_ms"] = s["pw"][i]
                w.writerow(row)
        print(f"[csv]  Saved -> {out_csv}  ({n} rows)")

    fig.canvas.mpl_connect("close_event", on_close)

    client = mqtt.Client()
    client.on_message = on_message
    print(f"[MQTT] Connecting to {MQTT_BROKER}:{MQTT_PORT} ...")
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.subscribe(MQTT_STATS_TOPIC)
    print(f"[MQTT] Subscribed to '{MQTT_STATS_TOPIC}'")
    client.loop_start()

    ani = animation.FuncAnimation(  # noqa: F841
        fig, update, interval=500, cache_frame_data=False
    )
    try:
        plt.show()
    except KeyboardInterrupt:
        on_close(None)
    client.loop_stop()


if __name__ == "__main__":
    main()
