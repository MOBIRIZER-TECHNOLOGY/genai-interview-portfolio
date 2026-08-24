# Atlas Fleet Platform - Overview

Atlas is the Northwind Robotics internal control plane for warehouse robot
fleets. It was first deployed at the Rotterdam DC in March 2023 and now runs at
eleven distribution centres.

## Components

| Component | Purpose | Language |
|---|---|---|
| `atlas-dispatch` | Assigns tasks to robots, runs the auction loop | Go |
| `atlas-telemetry` | Ingests robot sensor streams at 20 Hz | Rust |
| `atlas-vision` | Pallet + barcode recognition on edge GPUs | Python |
| `atlas-console` | Operator web UI | TypeScript |
| `atlas-sim` | Deterministic replay + simulation harness | Python |

## Key numbers

- A single Atlas cell supports a maximum of **240 concurrent robots**.
- The dispatch auction loop runs every **150 ms**.
- Telemetry retention is **90 days** hot, **7 years** cold.
- Target end-to-end task assignment latency is **p99 below 400 ms**.

## Deployment model

Each distribution centre runs its own Atlas cell. Cells are independent: there is
no cross-cell scheduling. A regional aggregator called `atlas-roll` pulls
read-only metrics from every cell for the executive dashboard.
