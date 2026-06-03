#pragma once

#include <esp_timer.h>
#include <cstdio>
#include <cstring>
#include "tensorflow/lite/micro/micro_profiler_interface.h"

// Keep in sync with arch_stats_t in mqtt_publisher.h
struct ArchStats {
    float total_ms;
    float conv_opening_ms;   // Event 0: Conv2D 3×3 stride-2
    float ds_ms[13];         // Events 1-26: DS block i → ds_ms[i-1] = DW+PW sum
    int   n_ds;              // always 13 for MobileNetV1
    float gap_ms;            // AVERAGE_POOL_2D
    float other_ms;          // RESHAPE + SOFTMAX
};

// Per-op profiler that maps TFLite Micro events to the MobileNetV1 architecture
// groups shown in the TikZ diagram:
//
//   Event  0        : CONV_2D            <- opening 3×3 Conv, stride 2
//   Events 1-2      : DW + PW            <- DS Block 1
//   Events 3-4      : DW + PW            <- DS Block 2
//   ...
//   Events 25-26    : DW + PW            <- DS Block 13
//   Event 27        : AVERAGE_POOL_2D    <- Global Average Pool
//   Event 28        : RESHAPE            <- negligible
//   Event 29        : SOFTMAX            <- classifier head
//
// Enable at build time with -DARCH_PROFILER.

class ArchProfiler : public tflite::MicroProfilerInterface {
 public:
  static constexpr int kMaxEvents = 64;

  uint32_t BeginEvent(const char* tag) override {
    if (num_events_ >= kMaxEvents) return 0;
    tags_[num_events_] = tag;
    start_us_[num_events_] = esp_timer_get_time();
    dur_us_[num_events_] = 0;
    return num_events_++;
  }

  void EndEvent(uint32_t handle) override {
    if (handle < kMaxEvents)
      dur_us_[handle] =
          (uint32_t)(esp_timer_get_time() - start_us_[handle]);
  }

  void ClearEvents() { num_events_ = 0; }

  ArchStats GetStats() const {
    ArchStats s{};
    uint64_t total_us = 0;
    for (int i = 0; i < num_events_; i++) total_us += dur_us_[i];
    s.total_ms = total_us / 1000.0f;

    if (num_events_ > 0)
      s.conv_opening_ms = dur_us_[0] / 1000.0f;

    s.n_ds = 0;
    int ev = 1;
    for (int blk = 0; blk < 13 && ev + 1 < num_events_; blk++, ev += 2) {
      s.ds_ms[blk] = (dur_us_[ev] + dur_us_[ev + 1]) / 1000.0f;
      s.n_ds++;
    }

    if (ev < num_events_)
      s.gap_ms = dur_us_[ev++] / 1000.0f;   // AVERAGE_POOL_2D

    float other_us = 0;
    while (ev < num_events_) other_us += dur_us_[ev++];
    s.other_ms = other_us / 1000.0f;

    return s;
  }

  void PrintArchBreakdown() const {
    if (num_events_ == 0) return;

    uint64_t total_us = 0;
    for (int i = 0; i < num_events_; i++) total_us += dur_us_[i];
    float total_ms = total_us / 1000.0f;

    printf("\n--- MobileNetV1 Arch Breakdown (%d ops) ---\n", num_events_);

    // Event 0: opening Conv2D 3×3
    _print_row("Conv2D 3x3 s2 (opening)", dur_us_[0], total_us);

    // Events 1..26: 13 depthwise separable blocks
    int ev = 1;
    for (int blk = 1; blk <= 13 && ev + 1 < num_events_; blk++, ev += 2) {
      char label[32];
      snprintf(label, sizeof(label), "DS Block %2d  (DW + PW)", blk);
      _print_row_2(label, dur_us_[ev], dur_us_[ev + 1], total_us);
    }

    // Remaining events: GAP, Reshape, Softmax
    while (ev < num_events_) {
      _print_row(tags_[ev], dur_us_[ev], total_us);
      ev++;
    }

    printf("  ------------------------------------------\n");
    printf("  Total Invoke()          : %6.2f ms\n\n", total_ms);
  }

 private:
  const char* tags_[kMaxEvents] = {};
  int64_t     start_us_[kMaxEvents] = {};
  uint32_t    dur_us_[kMaxEvents] = {};
  int         num_events_ = 0;

  static void _print_row(const char* label, uint32_t us, uint64_t total_us) {
    float ms  = us / 1000.0f;
    float pct = total_us > 0 ? 100.0f * us / total_us : 0.0f;
    printf("  %-28s: %6.2f ms  (%4.1f%%)\n", label, ms, pct);
  }

  static void _print_row_2(const char* label,
                            uint32_t dw_us, uint32_t pw_us,
                            uint64_t total_us) {
    float dw  = dw_us / 1000.0f;
    float pw  = pw_us / 1000.0f;
    float tot = dw + pw;
    float pct = total_us > 0 ? 100.0f * (dw_us + pw_us) / total_us : 0.0f;
    printf("  %-28s: %6.2f ms  (%4.1f%%)  [DW=%.2f PW=%.2f]\n",
           label, tot, pct, dw, pw);
  }
};
