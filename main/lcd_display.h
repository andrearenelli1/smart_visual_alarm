#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t lcd_display_init(void);

/*
 * Draw a rectangle of pixels to the LCD.
 * buf must be RGB565 in the byte order the camera outputs — no swap needed.
 * x_end / y_end are exclusive (esp_lcd convention).
 */
void lcd_display_draw_frame(const void *buf, int x_start, int y_start,
                             int x_end, int y_end);

#ifdef __cplusplus
}
#endif
