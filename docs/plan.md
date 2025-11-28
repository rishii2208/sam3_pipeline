# Fruits Counting Project Plan

## Objectives

- Implement ablation pipelines **P2** and **P3** defined by your lead and report their counting metrics.
- Reuse the existing dataset (Kaggle: afsananadia/fruits-images-dataset-object-detection) and integrate Meta/Facebook's SAM3 model as the attention-enabled detector backbone.
- Produce quantitative results plus qualitative examples so the comparison can feed directly into the paper draft.

## Requirements Recap

| Stage        | Option | Description                                                                                                                                     | Pipelines |
| ------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| Segmentation | S0     | No preprocessing, raw RGB                                                                                                                       | P2        |
| Segmentation | S1     | HSV color-threshold mask; keep masked RGB                                                                                                       | P3        |
| Detection    | D2     | Detector with adaptive attention (SAM3 backbone + your existing detection head). Baseline D1 only used for reference, not part of current task. | P2, P3    |
| Counting     | C1     | Box counting: threshold + NMS, then count surviving detections per image                                                                        | P2, P3    |

Deliverables:

1. Training/inference scripts for the two pipelines that can be launched via CLI arguments.
2. Logged metrics (mAP, precision/recall per class, counting MAE/RMSE) plus aggregated table comparing P2 vs P3.
3. Short write-up with observations and sample visualizations.

## Dataset Notes

- Download from Kaggle (approx 1.2 GB). Verify license allows research use.
- Expected folder structure: `train/`, `test/`, `annotations` (Pascal VOC style XML). Confirm once downloaded.
- Augmentations: horizontal flip, color jitter; keep consistent between pipelines.

## SAM3 Integration

1. Clone `facebookresearch/sam3` and install according to README (PyTorch 2.3+, CUDA 12 recommended).
2. Use SAM3 encoder as backbone; attach existing detection head (similar to your current attention model). Freeze first few stages to reduce overfitting if GPU limits.
3. Export feature maps for ROI heads; ensure attention weights logged for ablation discussion.

## Pipeline Breakdown

### P2: S0 + D2 + C1

1. Preprocess: resize to detector resolution, normalize.
2. SAM3-based detector with attention block enabled (D2).
3. Apply confidence threshold (e.g., 0.25) + class-wise NMS.
4. Count remaining boxes per image for final metric.

### P3: S1 + D2 + C1

1. Convert image to HSV, create binary mask via tuned hue/saturation/value thresholds per fruit color.
2. Multiply mask with RGB image (or stack as additional channel) before feeding SAM3 detector.
3. Remainder identical to P2.

## Experiment/Result Plan

1. Split data into train/val/test (70/15/15) ensuring class balance.
2. Train detector once per pipeline (shared weights if feasible, but log differences).
3. Evaluate on test split; store metrics under `results/p2/metrics.json` etc.
4. Visualize 20 random samples with predicted boxes and counts.
5. Summarize findings in `reports/ablation.md`.

## Immediate Next Actions

1. Scaffold repo structure (`src/`, `data/`, `scripts/`, `results/`).
2. Set up Python env (PyTorch + SAM3 dependencies, Kaggle API).
3. Implement data loader to parse annotation format.
4. Add modular pipeline runner that toggles S0 vs S1 via config.
5. Schedule experiments once training code verified on a smoke subset.
