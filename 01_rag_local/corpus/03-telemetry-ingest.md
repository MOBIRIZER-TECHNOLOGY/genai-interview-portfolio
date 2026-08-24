# atlas-telemetry: ingest pipeline

`atlas-telemetry` accepts a 20 Hz sensor stream from every robot over gRPC and
writes it to a TimescaleDB hypertable partitioned by day.

## Backpressure

When the write buffer exceeds **80%** capacity, telemetry enters *shed mode*:

1. Non-critical channels (`imu_raw`, `wheel_slip`) are dropped first.
2. If the buffer still exceeds 80% after 5 seconds, sample rate for all channels
   drops from 20 Hz to 5 Hz.
3. Shed mode clears when the buffer falls under 55% for 30 consecutive seconds.

Shed mode raises the `TelemetryDegraded` alert but does **not** page on-call
unless it persists for more than 10 minutes.

## Error codes

| Code | Meaning | Action |
|---|---|---|
| `TLM-101` | Clock skew above 250 ms between robot and cell | Re-sync NTP on the robot |
| `TLM-204` | Duplicate sequence number in stream | Benign, dedup handles it |
| `TLM-330` | Hypertable chunk write failed | Page DBA, check disk on `tsdb-0` |
| `TLM-402` | Schema version mismatch | Robot firmware needs upgrade |
