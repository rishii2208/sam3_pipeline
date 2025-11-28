"""Create reproducible train/val/test splits for the fruits dataset."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

SPLIT_NAMES = ["train", "val", "test"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


@dataclass
class Sample:
    image_path: Path
    annotation_path: Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Dataset root directory (after download)")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def collect_samples(root: Path) -> List[Sample]:
    label_files = sorted(root.rglob("*.txt"))
    samples: List[Sample] = []
    for label_path in label_files:
        if label_path.name.startswith("classes"):
            continue
        image_path = _match_image(label_path)
        if image_path is None:
            continue
        samples.append(Sample(image_path=image_path, annotation_path=label_path))
    if not samples:
        raise RuntimeError(f"No label files found under {root}")
    return samples


def _match_image(label_path: Path) -> Path | None:
    stem = label_path.stem
    for ext in IMAGE_EXTS:
        candidate = label_path.with_name(f"{stem}{ext}")
        if candidate.exists():
            return candidate
    return None


def assign_splits(samples: Sequence[Sample], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[Sample]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def ensure_split_dirs(root: Path) -> Dict[str, Dict[str, Path]]:
    layout = {}
    for split in SPLIT_NAMES:
        layout[split] = {
            "images": root / "images" / split,
            "annotations": root / "annotations" / split,
        }
        for path in layout[split].values():
            path.mkdir(parents=True, exist_ok=True)
    return layout


def sanitize_stem(stem: str) -> str:
    if "_" in stem:
        prefix, suffix = stem.split("_", 1)
    else:
        prefix, suffix = "", stem
    prefix = prefix.replace(" ", "")
    suffix = suffix.strip().replace(" ", "")
    if prefix:
        return f"{prefix}_{suffix}"
    return suffix


def move_sample(sample: Sample, split_dirs: Dict[str, Path], dry_run: bool = False) -> str:
    safe_stem = sanitize_stem(sample.annotation_path.stem)
    dst_ann = split_dirs["annotations"] / f"{safe_stem}.txt"
    dst_img = split_dirs["images"] / f"{safe_stem}{sample.image_path.suffix.lower()}"

    if dry_run:
        print(f"[DRY-RUN] Would move {sample.annotation_path} -> {dst_ann}")
        print(f"[DRY-RUN] Would move {sample.image_path} -> {dst_img}")
        return safe_stem

    shutil.copy2(sample.annotation_path, dst_ann)
    shutil.copy2(sample.image_path, dst_img)
    return safe_stem


def main():
    args = parse_args()
    split_index_path = args.root / "splits.json"
    if split_index_path.exists():
        print(f"Splits already exist at {split_index_path}. Delete it if you need to re-split.")
        return

    samples = collect_samples(args.root)
    splits = assign_splits(samples, args.train_ratio, args.val_ratio, args.seed)
    layout = ensure_split_dirs(args.root)

    split_index = {split: [] for split in SPLIT_NAMES}
    for split, records in splits.items():
        for sample in records:
            stem = move_sample(sample, layout[split], dry_run=args.dry_run)
            split_index[split].append(stem)

    with split_index_path.open("w", encoding="utf-8") as handle:
        json.dump(split_index, handle, indent=2)
    print(f"Wrote split metadata to {split_index_path}")


if __name__ == "__main__":
    main()
