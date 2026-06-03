#include "main_functions.h"
#include "esp_log.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_main.h"
#include "esp_cli.h"

extern "C" {
#include "wifi_manager.h"
#include "mqtt_publisher.h"
}

void tf_main(void)
{
    wifi_manager_init();
    mqtt_publisher_init();

    setup();
    esp_cli_start();

#if CLI_ONLY_INFERENCE
    esp_cli_register_inference_command();
    vTaskDelay(portMAX_DELAY);
#else
    while (true) {
        loop();
    }
#endif
}

extern "C" void app_main()
{
    xTaskCreate((TaskFunction_t)&tf_main, "tf_main", 4 * 1024, NULL, 8, NULL);
    vTaskDelete(NULL);
}
