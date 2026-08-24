# atlas-dispatch: the auction loop

Task assignment in Atlas uses a **sealed-bid reverse auction**. Every 150 ms,
`atlas-dispatch` publishes the set of open tasks, and every idle robot submits a
bid representing its estimated cost to complete that task.

## Bid formula

```
bid = travel_time_s + (0.4 * battery_penalty) + (2.1 * congestion_score)
```

- `battery_penalty` is `0` above 60% charge, and rises linearly to `30` at 10%.
- `congestion_score` comes from the aisle occupancy grid, refreshed every 500 ms.
- The lowest bid wins. Ties are broken by robot serial number, ascending.

## Starvation guard

A task that loses **12 consecutive auctions** is escalated: its priority is
multiplied by 1.5 and it is pinned to the next auction round regardless of bids.
This is called the *Rotterdam rule*, added after a September 2023 incident where
17 pallets sat unassigned for 40 minutes.

## Tuning knobs

| Env var | Default | Notes |
|---|---|---|
| `ATLAS_AUCTION_PERIOD_MS` | `150` | Below 100 ms the bid RPC saturates |
| `ATLAS_STARVATION_ROUNDS` | `12` | The Rotterdam rule threshold |
| `ATLAS_MAX_BIDS_PER_TASK` | `64` | Bids beyond this are dropped, nearest-first |
