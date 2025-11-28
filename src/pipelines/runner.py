"""Pipeline orchestration for P2/P3 experiments."""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import FruitsDataset
from src.models.sam3_detector import SAM3Detector
from src.pipelines.counting import CountingMetrics, count_boxes
from src.pipelines.segmentation import HSVMaskSegmentation, NoSegmentation


class PipelineRunner:
    def __init__(
        self,
        pipeline_id: str,
        pipeline_config_path: Path,
        dataset_config_path: Path,
        output_dir: Path,
        device: str = "cuda",
    ) -> None:
        self.pipeline_id = pipeline_id
        self.pipeline_config = self._read_yaml(pipeline_config_path)
        self.dataset_config = self._read_yaml(dataset_config_path)
        self.output_dir = output_dir
        self.device = device

        if pipeline_id not in self.pipeline_config:
            raise KeyError(f"Unknown pipeline id {pipeline_id}")
        self.pipeline_spec = self.pipeline_config[pipeline_id]

        dataset_section = self.dataset_config.get("dataset", {})
        self.dataset_root = Path(dataset_section.get("local_dir", "data/fruits"))
        self.dataset_name = dataset_section.get("name", "fruits")
        self.class_names = dataset_section.get("classes", [])

        self.calibration_path = self.output_dir / f"{self.pipeline_id}_calibration.json"
        self.calibration = self._load_calibration()
        self.count_threshold = self.calibration.get("score_threshold", 0.3)
        self.nms_iou = self.calibration.get("nms_iou", 0.5)

    def _read_yaml(self, path: Path):
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _build_segmentation(self):
        stage = self.pipeline_spec.get("segmentation", "S0")
        if stage == "S1":
            return HSVMaskSegmentation()
        return NoSegmentation()

    def _build_detector(self, class_names: List[str]):
        detector_stage = self.pipeline_spec.get("detector")
        if detector_stage != "D2":
            raise NotImplementedError(f"Only D2 detector is supported right now, got {detector_stage}")
        return SAM3Detector(class_names=class_names, device=self.device)

    def _collate_fn(self, batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        return images, list(targets)

    def evaluate(self, split: str = "val", limit: int | None = None) -> Dict[str, float]:
        segmentation = self._build_segmentation()
        dataset = FruitsDataset(
            root=self.dataset_root,
            split=split,
            segmentation=segmentation,
            class_names=self.class_names,
        )
        class_names = self._detector_class_names(dataset)
        detector = self._build_detector(class_names)
        summary = self._run_counting_epoch(
            dataset,
            detector,
            limit=limit,
            score_threshold=self.count_threshold,
            nms_iou=self.nms_iou,
        )
        self._persist_metrics(summary, split)
        return summary

    def _flatten_detections(self, detection_result: Dict) -> Tuple[np.ndarray, np.ndarray]:
        boxes_list: List[np.ndarray] = []  # type: ignore[name-defined]
        scores_list: List[np.ndarray] = []  # type: ignore[name-defined]
        for det in detection_result.get("detections", []):
            boxes_list.append(det["boxes"])
            scores_list.append(det["scores"])
        if not boxes_list:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)  # type: ignore[name-defined]
        return np.concatenate(boxes_list, axis=0), np.concatenate(scores_list, axis=0)

    def _persist_metrics(self, summary: Dict[str, float], split: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"{self.pipeline_id}_{split}_metrics.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    def train(self, resume_path: str | None = None, limit: int | None = None):
        segmentation = self._build_segmentation()
        dataset = FruitsDataset(
            root=self.dataset_root,
            split="train",
            segmentation=segmentation,
            class_names=self.class_names,
        )
        class_names = self._detector_class_names(dataset)
        detector = self._build_detector(class_names)

        thresholds = self.pipeline_spec.get("count_threshold_grid", [0.2, 0.25, 0.3, 0.35, 0.4])
        nms_values = self.pipeline_spec.get("nms_grid", [0.4, 0.5, 0.6])
        best = None

        for score_thr in thresholds:
            for nms_iou in nms_values:
                summary = self._run_counting_epoch(
                    dataset,
                    detector,
                    limit=limit or self.pipeline_spec.get("calibration_limit", 150),
                    score_threshold=score_thr,
                    nms_iou=nms_iou,
                )
                candidate = {
                    "score_threshold": score_thr,
                    "nms_iou": nms_iou,
                    "mae": summary["mae"],
                    "rmse": summary["rmse"],
                }
                if best is None or candidate["mae"] < best["mae"]:
                    best = candidate

        if best is None:
            raise RuntimeError("Calibration failed; dataset might be empty.")

        self.count_threshold = best["score_threshold"]
        self.nms_iou = best["nms_iou"]
        self._save_calibration(best)
        return best

    def _run_counting_epoch(
        self,
        dataset: FruitsDataset,
        detector: SAM3Detector,
        limit: int | None,
        score_threshold: float,
        nms_iou: float,
    ) -> Dict[str, float]:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=self._collate_fn)
        metrics = CountingMetrics()
        for idx, (images, targets) in enumerate(dataloader):
            if limit is not None and idx >= limit:
                break
            image_tensor = images[0]
            target = targets[0]
            gt_count = int(target["boxes"].shape[0])

            np_image = image_tensor.permute(1, 2, 0).cpu().numpy()
            np_image = (np_image * 255).astype("uint8")

            detection = detector.predict(np_image)
            boxes, scores = self._flatten_detections(detection)
            pred_count = count_boxes(boxes, scores, threshold=score_threshold, nms_iou=nms_iou)
            metrics.update(gt_count, pred_count)

        return metrics.summary()

    def _save_calibration(self, payload: Dict[str, float]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.calibration_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _load_calibration(self) -> Dict[str, float]:
        if self.calibration_path.exists():
            with self.calibration_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def _detector_class_names(self, dataset: FruitsDataset) -> List[str]:
        if dataset.class_to_idx:
            ordered = sorted(dataset.class_to_idx, key=dataset.class_to_idx.get)
            return [name.replace("_", " ") for name in ordered]
        return ["fruit"]
