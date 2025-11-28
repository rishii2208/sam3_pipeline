"""Evaluate a configured pipeline (P2/P3) on a dataset split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.runner import PipelineRunner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["p2", "p3"], required=True)
    parser.add_argument("--config", default="configs/pipelines.yaml")
    parser.add_argument("--dataset", default="configs/dataset.yaml")
    parser.add_argument("--output", default="results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of samples")
    return parser.parse_args()


def main():
    args = parse_args()
    runner = PipelineRunner(
        pipeline_id=args.pipeline,
        pipeline_config_path=Path(args.config),
        dataset_config_path=Path(args.dataset),
        output_dir=Path(args.output),
        device=args.device,
    )
    summary = runner.evaluate(split=args.split, limit=args.limit)
    print(summary)


if __name__ == "__main__":
    main()
