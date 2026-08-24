# atlas-vision: pallet and barcode recognition

`atlas-vision` runs on edge GPUs, one NVIDIA A2 per aisle gantry, and answers two
questions: *what pallet is this* and *is it safe to pick*.

## Models in production

| Model | Task | Input | p50 latency |
|---|---|---|---|
| `nw-pallet-detect-v4` | Pallet bounding boxes | 1280x720 RGB | 18 ms |
| `nw-barcode-ocr-v2` | Code-128 barcode read | 640x480 crop | 7 ms |
| `nw-damage-clf-v1` | Damaged/intact classifier | 224x224 crop | 4 ms |

## Confidence policy

- Barcode reads below **0.92** confidence are re-attempted up to 3 times with a
  different exposure.
- After 3 failures the pallet is routed to the **manual inspection lane**.
- The damage classifier is advisory only. It never blocks a pick, it files a
  `DamageSuspected` ticket.

## Known limitation

`nw-barcode-ocr-v2` degrades badly on shrink-wrapped pallets with specular glare.
Measured read rate drops from 99.1% to 84%. The v3 model, which adds polarised
capture, is scheduled for Q3.
