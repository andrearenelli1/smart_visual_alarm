idf_component_register(
    SRCS
        "detection_responder.cc"
        "image_provider.cc"
        "main.cc"
        "main_functions.cc"
        "model_settings.cc"
        "person_detect_model_data.cc"
        "app_camera_esp.c"
        "esp_cli.c"
        "wifi_manager.c"
        "mqtt_publisher.c"
        "lcd_display.c"

    PRIV_REQUIRES
        console static_images spi_flash
        nvs_flash esp_wifi esp_event esp_netif mqtt
        esp_lcd driver

    INCLUDE_DIRS "")

if(NO_LCD)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE NO_LCD=1)
endif()

if(ARCH_PROFILER)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ARCH_PROFILER=1)
endif()
