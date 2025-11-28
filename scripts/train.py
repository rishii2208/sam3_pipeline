"""CLI entry point to train a pipeline (P2 or P3)."""
import argparse
from pathlib import Path

from src.pipelines.runner import PipelineRunner


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["p2", "p3"], required=True)
    parser.add_argument("--config", default="configs/pipelines.yaml")
    parser.add_argument("--dataset", default="configs/dataset.yaml")
    parser.add_argument("--output", default="results")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Optional max samples for calibration")
    return parser.parse_args()


def main():
    args = build_args()
    runner = PipelineRunner(
        pipeline_id=args.pipeline,
        pipeline_config_path=Path(args.config),
        dataset_config_path=Path(args.dataset),
        output_dir=Path(args.output),
        device=args.device,
    )
    runner.train(resume_path=args.resume, limit=args.limit)


if __name__ == "__main__":
    main()
