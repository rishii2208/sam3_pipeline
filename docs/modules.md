# Module Design Notes

## Segmentation Stage

- `NoSegmentation` (S0): identity transform; keeps compatibility with transforms pipeline.
- `HSVMaskSegmentation` (S1): converts RGB to HSV and applies a configurable mask. Mask can later be tuned per class by passing multiple intervals or using morphology operations.
- Implementation is stateless callables returning masked RGB for compatibility with SAM3 input expectations (3-channel tensors).

## Detection Stage (D2)

- `SAM3Detector` will wrap the Facebook SAM3 encoder/decoder stack and attach the existing detection head (classification + box regression).
- Steps:
  1. Load SAM3 weights via the official repo (either pip install or git submodule) inside `src/models/sam3_backbone.py` (to be created).
  2. Freeze early backbone blocks for stability; expose an attention block toggle for ablations.
  3. Build detection head using FPN + anchor generator similar to RetinaNet or adapt the team's current head.
  4. Provide `training_step`, `validation_step`, and `predict` APIs expected by `PipelineRunner`.

## Counting Stage (C1)

- Simple helper `count_boxes` counts predictions above threshold post-NMS. Later we can extend to class-conditional thresholds or per-image metrics.

## Data Flow

1. `PipelineRunner` instantiates segmentation transform based on pipeline config.
2. Loader fetches images/boxes, applies transforms, and feeds the detector.
3. Detector outputs logits + boxes -> standard loss functions during training.
4. During evaluation, outputs go through NMS, then `count_boxes` for final metrics.

## Pending Work

- Implement dataset parser (XML/JSON) to supply bounding boxes.
- Finish `PipelineRunner.train/evaluate` with PyTorch training loop (likely Lightning for clarity).
- Integrate SAM3 repo as git submodule or add instructions for manual install.
