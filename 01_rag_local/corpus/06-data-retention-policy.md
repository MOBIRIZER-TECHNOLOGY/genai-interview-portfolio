# Data and privacy policy for Atlas

## Retention

| Data class | Hot | Cold | Deleted after |
|---|---|---|---|
| Robot telemetry | 90 days | 7 years | 7 years |
| Vision frames | 14 days | none | 14 days |
| Operator audit log | 400 days | 10 years | 10 years |
| Task/dispatch records | 2 years | 7 years | 7 years |

Vision frames are the shortest-lived class because gantry cameras can
incidentally capture warehouse staff. Frames are **blurred at the edge** before
leaving the gantry, and only the blurred version is ever written to storage.

## Access

- Telemetry and dispatch data: any Atlas engineer.
- Vision frames: requires membership of the `atlas-vision-review` group and an
  approved justification ticket. Access is logged and reviewed monthly.
- Operator audit log: security team only.

## Region rule

Data never leaves the region it was produced in. The Rotterdam and Hamburg cells
write to `eu-central-1`, the Memphis cell writes to `us-east-1`. `atlas-roll`
aggregates only **counts and percentiles**, never raw records, so it is exempt.
