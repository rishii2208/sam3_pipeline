"""Dataset utilities for the Kaggle fruits detection set."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    image_path: Path
    annotation_path: Path


def _normalize_class_name(name: str) -> str:
    return name.replace(" ", "").replace("-", "").replace("_", "").lower()


class FruitsDataset(Dataset):
    """Loader that supports Pascal VOC XML and YOLO TXT annotations."""

    def __init__(
        self,
        root: Path,
        split: str = "train",
        transforms: Callable | None = None,
        segmentation: Callable | None = None,
        class_names: Sequence[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.segmentation = segmentation
        self.samples = self._index_samples()

        if class_names:
            normalized = [_normalize_class_name(name) for name in class_names]
            self.class_to_idx = {name: idx + 1 for idx, name in enumerate(normalized)}
            self.yolo_idx_to_label = {idx: name for idx, name in enumerate(normalized)}
        else:
            self.class_to_idx = self._build_class_index()
            self.yolo_idx_to_label = {idx: name for name, idx in self.class_to_idx.items()}

    def _resolve_dir(self, split: str, kind: str) -> Path:
        candidates: Sequence[Path] = (
            self.root / kind / split,
            self.root / split / kind,
            self.root / kind,
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not locate {kind} directory for split '{split}' under {self.root}")

    def _index_samples(self) -> List[Sample]:
        images_dir = self._resolve_dir(self.split, "images")
        annotations_dir = self._resolve_dir(self.split, "annotations")
        samples: List[Sample] = []
        annotation_files = sorted(list(annotations_dir.glob("*.xml")) + list(annotations_dir.glob("*.txt")))
        for annotation_path in annotation_files:
            image_path = self._match_image(images_dir, annotation_path.stem)
            if image_path is None:
                continue
            samples.append(Sample(image_path=image_path, annotation_path=annotation_path))
        if not samples:
            raise RuntimeError(f"No annotation/image pairs found in {annotations_dir}")
        return samples

    def _match_image(self, images_dir: Path, stem: str) -> Path | None:
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _build_class_index(self) -> Dict[str, int]:
        labels = set()
        for sample in self.samples:
            if sample.annotation_path.suffix.lower() == ".xml":
                tree = ET.parse(sample.annotation_path)
                for obj in tree.findall("object"):
                    name = obj.findtext("name")
                    if name:
                        labels.add(_normalize_class_name(name))
            else:
                labels.add(self._class_from_filename(sample.annotation_path.stem))
        return {label: idx + 1 for idx, label in enumerate(sorted(labels))}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image at {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.segmentation is not None:
            image = self.segmentation(image)
        return image

    def _parse_annotation(self, path: Path, image_size: Tuple[int, int]):
        suffix = path.suffix.lower()
        if suffix == ".xml":
            return self._parse_voc_xml(path)
        if suffix == ".txt":
            return self._parse_yolo_txt(path, image_size)
        raise ValueError(f"Unsupported annotation format: {path}")

    def _parse_voc_xml(self, path: Path):
        tree = ET.parse(path)
        boxes: List[List[float]] = []
        labels: List[int] = []
        for obj in tree.findall("object"):
            name = obj.findtext("name", default="")
            bndbox = obj.find("bndbox")
            if not name or bndbox is None:
                continue
            xmin = float(bndbox.findtext("xmin", default="0"))
            ymin = float(bndbox.findtext("ymin", default="0"))
            xmax = float(bndbox.findtext("xmax", default="0"))
            ymax = float(bndbox.findtext("ymax", default="0"))
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx.get(_normalize_class_name(name), 0))
        if not boxes:
            boxes = [[0, 0, 1, 1]]
            labels = [0]
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def _parse_yolo_txt(self, path: Path, image_size: Tuple[int, int]):
        width, height = image_size
        boxes: List[List[float]] = []
        labels: List[int] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(float(parts[0]))
                x_center, y_center, box_w, box_h = map(float, parts[1:])
                x1 = max((x_center - box_w / 2) * width, 0.0)
                y1 = max((y_center - box_h / 2) * height, 0.0)
                x2 = min((x_center + box_w / 2) * width, width)
                y2 = min((y_center + box_h / 2) * height, height)
                boxes.append([x1, y1, x2, y2])
                label_name = self.yolo_idx_to_label.get(class_id, self._class_from_filename(path.stem))
                labels.append(self.class_to_idx.get(label_name, 0))
        if not boxes:
            label_name = self._class_from_filename(path.stem)
            labels = [self.class_to_idx.get(label_name, 0)]
            boxes = [[0, 0, 1, 1]]
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def _class_from_filename(self, stem: str) -> str:
        if "_" in stem:
            _, name = stem.split("_", 1)
        else:
            name = stem
        return _normalize_class_name(name)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self._load_image(sample.image_path)
        height, width = image.shape[:2]
        boxes, labels = self._parse_annotation(sample.annotation_path, (width, height))

        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "image_id": torch.tensor([index]),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image_tensor, target
