/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 */

#include "detection_responder.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "esp_main.h"
#include "esp_timer.h"
#include "sdkconfig.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define LED_GPIO        GPIO_NUM_3
#define LED_BLINK_COUNT 6        /* on+off cycles */
#define LED_BLINK_MS    150      /* half-period */

static bool s_led_initialized = false;

static void led_init()
{
    gpio_config_t cfg = {
        .pin_bit_mask = 1ULL << LED_GPIO,
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&cfg);
    gpio_set_level(LED_GPIO, 0);
    s_led_initialized = true;
}

static void blink_task(void *)
{
    for (int i = 0; i < LED_BLINK_COUNT; i++) {
        gpio_set_level(LED_GPIO, 1);
        vTaskDelay(pdMS_TO_TICKS(LED_BLINK_MS));
        gpio_set_level(LED_GPIO, 0);
        vTaskDelay(pdMS_TO_TICKS(LED_BLINK_MS));
    }
    vTaskDelete(NULL);
}

static void led_blink_alarm()
{
    xTaskCreate(blink_task, "led_blink", 1024, NULL, 5, NULL);
}

extern "C" {
#include "mqtt_publisher.h"
}

#if DISPLAY_SUPPORT
#include "image_provider.h"
#include "bsp/esp-bsp.h"

#define IMG_WD 240
#define IMG_HT 240

static lv_obj_t *camera_canvas = NULL;
static lv_obj_t *person_indicator = NULL;
static lv_obj_t *label = NULL;

void create_gui(void)
{
    bsp_display_cfg_t cfg = {
        .lvgl_port_cfg = {
            .task_priority   = CONFIG_BSP_DISPLAY_LVGL_TASK_PRIORITY,
            .task_stack      = 6144,
            .task_affinity   = 1,
            .task_max_sleep_ms = CONFIG_BSP_DISPLAY_LVGL_MAX_SLEEP,
            .timer_period_ms = CONFIG_BSP_DISPLAY_LVGL_TICK,
        },
        .buffer_size   = BSP_LCD_DRAW_BUFF_SIZE,
        .double_buffer = BSP_LCD_DRAW_BUFF_DOUBLE,
        .flags = { .buff_dma = false, .buff_spiram = true },
    };
    bsp_display_start_with_config(&cfg);
    bsp_display_backlight_on();

    bsp_display_lock(0);
    camera_canvas = lv_canvas_create(lv_scr_act());
    assert(camera_canvas);
    lv_obj_align(camera_canvas, LV_ALIGN_TOP_MID, 0, 0);

    person_indicator = lv_led_create(lv_scr_act());
    assert(person_indicator);
    lv_obj_align(person_indicator, LV_ALIGN_BOTTOM_MID, -70, 0);
    lv_led_set_color(person_indicator, lv_palette_main(LV_PALETTE_RED));

    label = lv_label_create(lv_scr_act());
    assert(label);
    lv_label_set_text_static(label, "Person detected");
    lv_obj_align_to(label, person_indicator, LV_ALIGN_OUT_RIGHT_MID, 20, 0);
    bsp_display_unlock();
}
#elif LCD_DISPLAY
#include "image_provider.h"
extern "C" {
#include "lcd_display.h"
}
#endif // DISPLAY_SUPPORT / LCD_DISPLAY

#ifndef CONFIG_SCORE_WINDOW
#define CONFIG_SCORE_WINDOW 5
#endif
#define SCORE_WINDOW CONFIG_SCORE_WINDOW

static int     s_event_id       = 0;
static int64_t s_last_alarm_ms  = 0;
static int     s_score_buf[SCORE_WINDOW] = {0};
static int     s_score_idx      = 0;

static int moving_average(int new_score)
{
    s_score_buf[s_score_idx] = new_score;
    s_score_idx = (s_score_idx + 1) % SCORE_WINDOW;
    int sum = 0;
    for (int i = 0; i < SCORE_WINDOW; i++) sum += s_score_buf[i];
    return sum / SCORE_WINDOW;
}

void RespondToDetection(float person_score, float no_person_score, int invoke_ms)
{
    if (!s_led_initialized) led_init();

    int raw_pct          = (int)(person_score * 100 + 0.5f);
    int person_score_int = moving_average(raw_pct);
    (void)no_person_score;
    mqtt_publisher_publish_score(raw_pct, person_score_int, invoke_ms);

#if DISPLAY_SUPPORT
    if (!camera_canvas) {
        create_gui();
    }
    uint16_t *buf = (uint16_t *)image_provider_get_display_buf();
    bsp_display_lock(0);
    if (person_score_int < CONFIG_ALARM_THRESHOLD_PCT) {
        lv_led_off(person_indicator);
    } else {
        lv_led_on(person_indicator);
    }
    lv_canvas_set_buffer(camera_canvas, buf, IMG_WD, IMG_HT, LV_COLOR_FORMAT_RGB565);
    bsp_display_unlock();
#elif LCD_DISPLAY && !defined(NO_LCD)
    {
        uint16_t *buf = (uint16_t *)image_provider_get_display_buf();
        lcd_display_draw_frame(buf, 0, 0, 240, 240);
    }
#endif

    MicroPrintf("person score:%d%%, no person score %d%%",
                person_score_int, 100 - person_score_int);

    if (person_score_int >= CONFIG_ALARM_THRESHOLD_PCT) {
        int64_t now_ms      = esp_timer_get_time() / 1000LL;
        int64_t cooldown_ms = (int64_t)CONFIG_ALARM_COOLDOWN_SEC * 1000LL;
        if (now_ms - s_last_alarm_ms >= cooldown_ms) {
            s_last_alarm_ms = now_ms;
            led_blink_alarm();
            mqtt_publisher_publish_alarm(++s_event_id, person_score);
        }
    }
}
