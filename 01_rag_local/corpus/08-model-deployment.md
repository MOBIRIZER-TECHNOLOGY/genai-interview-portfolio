# Deploying a model to atlas-vision

Vision models are shipped as TensorRT engines, built per GPU architecture.

## The pipeline

1. Train in PyTorch, export ONNX (opset 17).
2. `trtexec` builds the engine on a machine with the **same** GPU as the target
   gantry. Engines are not portable across architectures.
3. Engine plus metadata go to the `nw-models` S3 bucket under
   `vision/{model_name}/{version}/`.
4. A `ModelRelease` CRD in the cell Kubernetes namespace pins the version.
5. Rollout is **canary by aisle**: one gantry for 24 hours, then 25%, then all.

## Required metadata

Every release must ship a `card.yaml` with:

- `training_data_snapshot` - the dataset hash
- `eval_report` - precision/recall on the frozen `vision-eval-2024q2` set
- `fallback_version` - what to roll back to
- `owner` - a named engineer, not a team

A release without `fallback_version` is rejected by the admission webhook.

## Acceptance thresholds

| Model type | Metric | Minimum |
|---|---|---|
| Pallet detect | mAP@0.5 | 0.94 |
| Barcode OCR | Exact-match read rate | 0.985 |
| Damage classifier | Recall on damaged class | 0.80 |
