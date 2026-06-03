#pragma once

void mqtt_publisher_init(void);
void mqtt_publisher_publish_alarm(int event_id, float confidence);
void mqtt_publisher_publish_score(int raw_pct, int filtered_pct);
