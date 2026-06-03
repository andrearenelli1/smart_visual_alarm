/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Adapted for MobileNetV2: input 160x160x3 RGB int8.
// Camera captures 240x240 RGB565; pixels are nearest-neighbour downscaled to
// 160x160 and converted to signed 8-bit RGB (uint8 - 128).

#include "string.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#if (CONFIG_TFLITE_USE_BSP)
#include "bsp/esp-bsp.h"
#endif

#include "esp_heap_caps.h"
#include "esp_log.h"

#include "app_camera_esp.h"
#include "esp_camera.h"
#include "model_settings.h"
#include "image_provider.h"
#include "esp_main.h"

#if LCD_DISPLAY
#include "lcd_display.h"
#endif

static const char* TAG = "app_camera";
static uint16_t* display_buf;

// Decode one RGB565 pixel (little-endian as returned by ESP camera driver)
// into separate R, G, B uint8 components.
static inline void rgb565_to_rgb888(uint16_t px,
                                    uint8_t* r, uint8_t* g, uint8_t* b) {
  uint8_t hb = px & 0xFF;
  uint8_t lb = px >> 8;
  *r = (lb & 0x1F) << 3;
  *g = ((hb & 0x07) << 5) | ((lb & 0xE0) >> 3);
  *b = (hb & 0xF8);
}

// Downscale src (src_w x src_h, RGB565) to kNumRows x kNumCols and write
// channel-last RGB int8 into image_data[row * kNumCols * 3 + col * 3 + ch].
static void downscale_rgb565_to_int8(const uint16_t* src,
                                     int src_w, int src_h,
                                     int8_t* image_data) {
  for (int mi = 0; mi < kNumRows; mi++) {
    int si = (mi * src_h) / kNumRows;
    for (int mj = 0; mj < kNumCols; mj++) {
      int sj = (mj * src_w) / kNumCols;
      uint8_t r, g, b;
      rgb565_to_rgb888(src[si * src_w + sj], &r, &g, &b);
      int base = (mi * kNumCols + mj) * 3;
      image_data[base + 0] = (int8_t)(r - 128);
      image_data[base + 1] = (int8_t)(g - 128);
      image_data[base + 2] = (int8_t)(b - 128);
    }
  }
}

TfLiteStatus InitCamera() {
#if CLI_ONLY_INFERENCE
  ESP_LOGI(TAG, "CLI_ONLY_INFERENCE enabled, skipping camera init");
  return kTfLiteOk;
#endif

#if DISPLAY_SUPPORT || LCD_DISPLAY
  if (display_buf == NULL) {
    display_buf = (uint16_t*) heap_caps_malloc(
        240 * 240 * sizeof(uint16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  }
  if (display_buf == NULL) {
    ESP_LOGE(TAG, "Couldn't allocate display buffer");
    return kTfLiteError;
  }
#endif

#if LCD_DISPLAY
  if (lcd_display_init() != ESP_OK) {
    ESP_LOGE(TAG, "LCD init failed");
    return kTfLiteError;
  }
#endif

#if ESP_CAMERA_SUPPORTED
  int ret = app_camera_init();
  if (ret != 0) {
    MicroPrintf("Camera init failed\n");
    return kTfLiteError;
  }
  MicroPrintf("Camera Initialized\n");
#else
  ESP_LOGE(TAG, "Camera not supported for this device");
#endif
  return kTfLiteOk;
}

void* image_provider_get_display_buf() {
  return (void*) display_buf;
}

TfLiteStatus GetImage(int image_width, int image_height, int channels,
                      int8_t* image_data) {
#if ESP_CAMERA_SUPPORTED
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    ESP_LOGE(TAG, "Camera capture failed");
    return kTfLiteError;
  }

  const uint16_t* src = (const uint16_t*) fb->buf;
  // Camera always captures 240x240 RGB565 (CAMERA_FRAME_SIZE FRAMESIZE_240X240)
  const int src_w = 240;
  const int src_h = 240;

#if LCD_DISPLAY
  // Inference: downscale 240x240 → 160x160 RGB int8
  downscale_rgb565_to_int8(src, src_w, src_h, image_data);

  // Display: copy full 240x240 frame to display_buf.
  // Camera outputs RGB565 in the byte order the ST7789 expects — no swap needed.
  // detection_responder.cc calls lcd_display_draw_frame on this buffer.
  memcpy(display_buf, src, src_w * src_h * sizeof(uint16_t));
  esp_camera_fb_return(fb);

#elif DISPLAY_SUPPORT
  // BSP display path: downscale for inference, hand full frame to BSP
  downscale_rgb565_to_int8(src, src_w, src_h, image_data);
  memcpy(display_buf, src, src_w * src_h * sizeof(uint16_t));
  esp_camera_fb_return(fb);

#else
  // No display: just downscale and convert
  MicroPrintf("Image Captured\n");
  downscale_rgb565_to_int8(src, src_w, src_h, image_data);
  esp_camera_fb_return(fb);
#endif

  return kTfLiteOk;
#else
  return kTfLiteError;
#endif
}
