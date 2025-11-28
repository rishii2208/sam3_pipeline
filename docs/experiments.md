# Experiment & Evaluation Plan

## Data Splits

- Apply deterministic split script (`scripts/split_data.py`, to be implemented) using 70/15/15 stratified by fruit class.
- Maintain a `splits.json` artifact for reproducibility.

## Training Configurations

| Pipeline | Segmentation | Detector | Counting | Notes                       |
| -------- | ------------ | -------- | -------- | --------------------------- |
| P2       | S0           | SAM3+D2  | C1       | Baseline w/out segmentation |
| P3       | S1           | SAM3+D2  | C1       | Adds HSV mask prior         |

Shared hyper-parameters:

- Image size: 640 x 640 (consistent with SAM3 default).
- Batch size: 8 (adjust if GPU memory limited).
- Optimizer: AdamW (lr=3e-4, weight decay=1e-4).
- Scheduler: cosine decay with warmup (5 epochs).
- Epochs: 50 (early-stop on val mAP).

## Metrics

1. Detection: mAP@0.5, mAP@0.5:0.95, per-class precision/recall.
2. Counting: mean absolute error, root mean square error, per-class count accuracy.
3. Runtime: throughput (images/sec) recorded for fairness.

## Logging

- Use TensorBoard or WandB for loss curves.
- Save checkpoints under `results/<pipeline>/checkpoints/`.
- Store metrics in `results/<pipeline>/metrics.json`.

## Reporting

- Generate `reports/ablation.md` summarizing metrics.
- Include table + commentary on attention + segmentation impact.
- Add qualitative grid of predictions for both pipelines.
