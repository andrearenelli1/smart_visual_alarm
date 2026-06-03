/*
 * ST7789 LCD driver for ESP32-S3-Eye
 *
 * Uses esp_lcd_panel (IDF 5.x) over SPI3, no LVGL/BSP dependency.
 * Asynchronous DMA transfers: a binary semaphore gates each draw call so
 * the caller never overwrites the buffer while SPI is still reading it.
 *
 * Pin mapping (ESP32-S3-Eye):
 *   SCLK 21 | MOSI 47 | DC 43 | CS 44 | BL 48 (active-low) | RST n/a
 */

#include "lcd_display.h"

#include "driver/spi_master.h"
#include "driver/gpio.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

static const char *TAG = "lcd_display";

#define LCD_HOST         SPI3_HOST
#define LCD_CLK_HZ       (40 * 1000 * 1000)
#define LCD_PIN_SCLK     21
#define LCD_PIN_MOSI     47
#define LCD_PIN_DC       43
#define LCD_PIN_CS       44
#define LCD_PIN_RST      (-1)
#define LCD_PIN_BL       48
#define LCD_BL_ON_LEVEL  0          /* backlight is active-low */
#define LCD_H_RES        240
#define LCD_V_RES        240
#define LCD_CMD_BITS     8
#define LCD_PARAM_BITS   8

static esp_lcd_panel_handle_t s_panel   = NULL;
static SemaphoreHandle_t      s_lcd_done = NULL;

static bool IRAM_ATTR on_trans_done(esp_lcd_panel_io_handle_t io,
                                     esp_lcd_panel_io_event_data_t *edata,
                                     void *ctx)
{
    BaseType_t woken = pdFALSE;
    xSemaphoreGiveFromISR((SemaphoreHandle_t)ctx, &woken);
    return woken == pdTRUE;
}

esp_err_t lcd_display_init(void)
{
    s_lcd_done = xSemaphoreCreateBinary();
    if (!s_lcd_done) {
        ESP_LOGE(TAG, "Failed to create semaphore");
        return ESP_ERR_NO_MEM;
    }
    /* Pre-give so the first draw doesn't block. */
    xSemaphoreGive(s_lcd_done);

    /* Backlight off during init sequence. */
    gpio_config_t bl_cfg = {
        .mode         = GPIO_MODE_OUTPUT,
        .pin_bit_mask = 1ULL << LCD_PIN_BL,
    };
    ESP_ERROR_CHECK(gpio_config(&bl_cfg));
    gpio_set_level(LCD_PIN_BL, !LCD_BL_ON_LEVEL);

    spi_bus_config_t bus = {
        .mosi_io_num     = LCD_PIN_MOSI,
        .miso_io_num     = -1,
        .sclk_io_num     = LCD_PIN_SCLK,
        .quadwp_io_num   = -1,
        .quadhd_io_num   = -1,
        .max_transfer_sz = LCD_H_RES * LCD_V_RES * 2,
    };
    ESP_ERROR_CHECK(spi_bus_initialize(LCD_HOST, &bus, SPI_DMA_CH_AUTO));

    esp_lcd_panel_io_handle_t io;
    esp_lcd_panel_io_spi_config_t io_cfg = {
        .dc_gpio_num            = LCD_PIN_DC,
        .cs_gpio_num            = LCD_PIN_CS,
        .pclk_hz                = LCD_CLK_HZ,
        .lcd_cmd_bits           = LCD_CMD_BITS,
        .lcd_param_bits         = LCD_PARAM_BITS,
        .spi_mode               = 0,
        .trans_queue_depth      = 10,
        .on_color_trans_done    = on_trans_done,
        .user_ctx               = s_lcd_done,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_HOST, &io_cfg, &io));

    esp_lcd_panel_dev_config_t panel_cfg = {
        .reset_gpio_num = LCD_PIN_RST,
        .rgb_ele_order  = LCD_RGB_ELEMENT_ORDER_RGB,
        .bits_per_pixel = 16,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_st7789(io, &panel_cfg, &s_panel));
    ESP_ERROR_CHECK(esp_lcd_panel_reset(s_panel));
    ESP_ERROR_CHECK(esp_lcd_panel_init(s_panel));
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(s_panel, true));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(s_panel, true));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(s_panel, false, false));

    gpio_set_level(LCD_PIN_BL, LCD_BL_ON_LEVEL);
    ESP_LOGI(TAG, "ST7789 240×240 ready");
    return ESP_OK;
}

void lcd_display_draw_frame(const void *buf, int x_start, int y_start,
                             int x_end, int y_end)
{
    if (!s_panel || !s_lcd_done) return;

    /* Wait for the previous SPI DMA transfer to complete. */
    xSemaphoreTake(s_lcd_done, portMAX_DELAY);

    /* Kick off the new transfer asynchronously (returns immediately). */
    ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(s_panel,
                                              x_start, y_start,
                                              x_end, y_end, buf));
}
