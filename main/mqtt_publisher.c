#include "mqtt_publisher.h"
#include "sdkconfig.h"

#ifndef CONFIG_MQTT_SCORES_TOPIC
#define CONFIG_MQTT_SCORES_TOPIC "alarm/scores"
#endif
#ifndef CONFIG_MQTT_STATS_TOPIC
#define CONFIG_MQTT_STATS_TOPIC "alarm/stats"
#endif
#include "mqtt_client.h"
#include "esp_log.h"
#include "esp_timer.h"
#include <stdio.h>

static const char *TAG = "mqtt";
static esp_mqtt_client_handle_t s_client = NULL;

static void mqtt_event_handler(void *arg, esp_event_base_t base,
                                int32_t event_id, void *event_data)
{
    esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t)event_data;
    switch (event->event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "Connected to broker %s:%d",
                     CONFIG_MQTT_BROKER_IP, CONFIG_MQTT_BROKER_PORT);
            break;
        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGW(TAG, "Disconnected from broker");
            break;
        case MQTT_EVENT_ERROR:
            ESP_LOGE(TAG, "MQTT error");
            break;
        default:
            break;
    }
}

void mqtt_publisher_init(void)
{
    esp_mqtt_client_config_t cfg = {
        .broker.address.hostname = CONFIG_MQTT_BROKER_IP,
        .broker.address.port     = CONFIG_MQTT_BROKER_PORT,
        .broker.address.transport = MQTT_TRANSPORT_OVER_TCP,
    };
    s_client = esp_mqtt_client_init(&cfg);
    esp_mqtt_client_register_event(s_client, ESP_EVENT_ANY_ID,
                                   mqtt_event_handler, NULL);
    ESP_ERROR_CHECK(esp_mqtt_client_start(s_client));
}

void mqtt_publisher_publish_alarm(int event_id, float confidence)
{
    if (!s_client) return;

    char payload[128];
    snprintf(payload, sizeof(payload),
             "{\"event_id\":%d,\"confidence\":%.2f,\"timestamp_ms\":%lld}",
             event_id, confidence,
             (long long)(esp_timer_get_time() / 1000LL));

    int msg_id = esp_mqtt_client_publish(
        s_client, CONFIG_MQTT_TOPIC, payload, 0, /*qos=*/1, /*retain=*/0);

    if (msg_id >= 0) {
        ESP_LOGI(TAG, "Alarm published [event=%d conf=%.0f%%]: %s",
                 event_id, confidence * 100, payload);
    } else {
        ESP_LOGE(TAG, "Publish failed (not connected?)");
    }
}

void mqtt_publisher_publish_score(int raw_pct, int filtered_pct)
{
    if (!s_client) return;

    char payload[64];
    snprintf(payload, sizeof(payload),
             "{\"raw\":%d,\"filtered\":%d}", raw_pct, filtered_pct);

    esp_mqtt_client_publish(
        s_client, CONFIG_MQTT_SCORES_TOPIC, payload, 0, /*qos=*/0, /*retain=*/0);
}

void mqtt_publisher_publish_stats(const char *json)
{
    if (!s_client || !json) return;
    esp_mqtt_client_publish(
        s_client, CONFIG_MQTT_STATS_TOPIC, json, 0, /*qos=*/0, /*retain=*/0);
}
