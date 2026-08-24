# Atlas on-call runbook

On-call rotation is **one week**, handing over Tuesdays at 10:00 CET.
Primary carries the pager, secondary is the escalation after 15 minutes.

## Severity ladder

| Sev | Definition | Response time |
|---|---|---|
| SEV1 | Robots halted in a cell, or safety system offline | 5 min, page immediately |
| SEV2 | Throughput down more than 30%, or dispatch p99 above 2 s | 15 min |
| SEV3 | Degraded but throughput normal, e.g. shed mode | Next business day |

## Top three incidents by frequency

1. **`TLM-101` clock skew storms** - usually a failed NTP pod. Restart
   `ntp-relay` in the cell namespace. 41% of pages.
2. **Dispatch auction stall** - the bid RPC pool exhausts. Symptom: auction
   period climbs above 400 ms. Fix: scale `atlas-dispatch` replicas from 3 to 5.
   22% of pages.
3. **Vision gantry GPU fell off the bus** - requires a physical power cycle of
   the gantry. 14% of pages.

## The freeze window

No Atlas deploys between **1 November and 7 January**, which is peak season.
Exceptions need VP Engineering sign-off, filed as an `ATLAS-FREEZE-EXC` ticket.
