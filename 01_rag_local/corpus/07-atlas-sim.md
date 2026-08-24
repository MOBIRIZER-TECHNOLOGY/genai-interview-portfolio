# atlas-sim: deterministic replay

`atlas-sim` replays recorded telemetry against a candidate `atlas-dispatch` build
so you can measure a scheduling change before it touches a real warehouse.

## Determinism guarantees

Given the same **scenario file** and the same **seed**, atlas-sim produces
byte-identical output. This holds because:

- The auction loop is driven by simulated time, not wall-clock time.
- Random tie-breaks use a seeded PCG64 generator, one stream per robot.
- Floating point is pinned to `float64` and FMA is disabled in the sim build.

## Scenario files

Scenarios live in `sim/scenarios/*.yaml`. The three canonical ones:

- `peak-friday.yaml` - Rotterdam, 214 robots, Black Friday 2024 trace.
- `degraded-gantry.yaml` - Memphis with 2 of 9 vision gantries offline.
- `battery-cliff.yaml` - a fleet that all hits 15% charge inside 4 minutes.

## The regression gate

Any PR touching `atlas-dispatch` must run all three scenarios. The gate fails if
**throughput drops more than 2%** or **p99 assignment latency rises above 400 ms**
on any scenario. This gate has blocked 6 regressions since it was added.
