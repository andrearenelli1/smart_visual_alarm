#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void mqtt_publisher_init(void);
void mqtt_publisher_publish_alarm(int event_id, float confidence);
void mqtt_publisher_publish_score(int raw_pct, int filtered_pct);
void mqtt_publisher_publish_stats(const char *json);

#ifdef __cplusplus
}
#endif
