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

// Get the camera module ready
TfLiteStatus InitCamera() {
#if CLI_ONLY_INFERENCE
  ESP_LOGI(TAG, "CLI_ONLY_INFERENCE enabled, skipping camera init");
  return kTfLiteOk;
#endif
// If any display path is active, allocate the upscaled RGB565 frame buffer.
// Layout (uint16_t elements, total = 240×240 + 96×96 = 66816):
//   [0    .. 57599] upscaled 240×240 output written to LCD
//   [57600.. 66815] temporary raw 96×96 camera frame
#if DISPLAY_SUPPORT || LCD_DISPLAY
  if (display_buf == NULL) {
#if DISPLAY_SUPPORT
    display_buf = (uint16_t *) heap_caps_malloc((240 * 240 + 96 * 96) * sizeof(uint16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
#else
    display_buf = (uint16_t *) heap_caps_malloc(240 * 240 * sizeof(uint16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
#endif
  }
  if (display_buf == NULL) {
    ESP_LOGE(TAG, "Couldn't allocate display buffer");
    return kTfLiteError;
  }
#endif // DISPLAY_SUPPORT || LCD_DISPLAY

#if LCD_DISPLAY
  if (lcd_display_init() != ESP_OK) {
    ESP_LOGE(TAG, "LCD init failed");
    return kTfLiteError;
  }
#endif // LCD_DISPLAY

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

void *image_provider_get_display_buf()
{
  return (void *) display_buf;
}

// Get an image from the camera module
TfLiteStatus GetImage(int image_width, int image_height, int channels, int8_t* image_data) {
#if ESP_CAMERA_SUPPORTED
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    ESP_LOGE(TAG, "Camera capture failed");
    return kTfLiteError;
  }

#if DISPLAY_SUPPORT
  // Camera at 96×96: copy raw frame, extract grayscale for inference,
  // byte-swap, then upscale 2.5× to 240×240 for display.
  uint16_t* cam_buf = display_buf + (240 * 240);
  memcpy((uint8_t*)cam_buf, fb->buf, fb->len);
  esp_camera_fb_return(fb);

  for (int i = 0; i < kNumRows; i++) {
    for (int j = 0; j < kNumCols; j++) {
      uint16_t inference_pixel = cam_buf[i * kNumCols + j];
      uint8_t hb = inference_pixel & 0xFF;
      uint8_t lb = inference_pixel >> 8;
      uint8_t r = (lb & 0x1F) << 3;
      uint8_t g = ((hb & 0x07) << 5) | ((lb & 0xE0) >> 3);
      uint8_t b = (hb & 0xF8);
      image_data[i * kNumCols + j] = (int8_t)(((305 * r + 600 * g + 119 * b) >> 10) - 128);
    }
  }

  lv_draw_sw_rgb565_swap(cam_buf, 96 * 96);

  for (int i = 0; i < 240; i++) {
    int si = (i * 96) / 240;
    for (int j = 0; j < 240; j++) {
      display_buf[i * 240 + j] = cam_buf[si * kNumCols + (j * 96) / 240];
    }
  }
#elif LCD_DISPLAY
  // Camera at 240×240: downscale nearest-neighbor to 96×96 for inference,
  // byte-swap full frame into display_buf for LCD.
  uint16_t* src = (uint16_t*)fb->buf;

  for (int mi = 0; mi < kNumRows; mi++) {
    int si = (mi * 240) / kNumRows;
    for (int mj = 0; mj < kNumCols; mj++) {
      int sj = (mj * 240) / kNumCols;
      uint16_t px = src[si * 240 + sj];
      uint8_t hb = px & 0xFF;
      uint8_t lb = px >> 8;
      uint8_t r = (lb & 0x1F) << 3;
      uint8_t g = ((hb & 0x07) << 5) | ((lb & 0xE0) >> 3);
      uint8_t b = (hb & 0xF8);
      image_data[mi * kNumCols + mj] = (int8_t)(((305 * r + 600 * g + 119 * b) >> 10) - 128);
    }
  }

  memcpy(display_buf, src, 240 * 240 * sizeof(uint16_t));

  esp_camera_fb_return(fb);
#else // DISPLAY_SUPPORT || LCD_DISPLAY
  MicroPrintf("Image Captured\n");
  // We have initialised camera to grayscale
  // Just quantize to int8_t
  for (int i = 0; i < image_width * image_height; i++) {
    image_data[i] = ((uint8_t *) fb->buf)[i] ^ 0x80;
  }

  esp_camera_fb_return(fb);
#endif // DISPLAY_SUPPORT || LCD_DISPLAY
  /* here the esp camera can give you grayscale image directly */
  return kTfLiteOk;
#else
  return kTfLiteError;
#endif
}
