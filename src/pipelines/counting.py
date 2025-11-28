"""Counting stage (C1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torchvision.ops import nms


def filter_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    score_threshold: float = 0.25,
    nms_iou: float = 0.5,
) -> torch.Tensor:
    if boxes.size == 0:
        return torch.empty(0, dtype=torch.long)
    torch_boxes = torch.as_tensor(boxes, dtype=torch.float32)
    torch_scores = torch.as_tensor(scores, dtype=torch.float32)
    keep = torch_scores >= score_threshold
    torch_boxes = torch_boxes[keep]
    torch_scores = torch_scores[keep]
    if torch_boxes.size(0) == 0:
        return torch.empty(0, dtype=torch.long)
    keep_idx = nms(torch_boxes, torch_scores, nms_iou)
    return keep_idx


def count_boxes(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.25, nms_iou: float = 0.5) -> int:
    keep_idx = filter_boxes(boxes, scores, score_threshold=threshold, nms_iou=nms_iou)
    return int(keep_idx.numel())


@dataclass
class CountingMetrics:
    """Simple MAE/RMSE tracker for predicted counts."""

    gt_counts: List[int]
    pred_counts: List[int]

    def __init__(self) -> None:
        self.gt_counts = []
        self.pred_counts = []

    def update(self, gt_count: int, pred_count: int) -> None:
        self.gt_counts.append(int(gt_count))
        self.pred_counts.append(int(pred_count))

    def summary(self) -> Dict[str, float]:
        if not self.gt_counts:
            return {"mae": 0.0, "rmse": 0.0}
        diff = np.array(self.pred_counts) - np.array(self.gt_counts)
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        return {"mae": mae, "rmse": rmse}
