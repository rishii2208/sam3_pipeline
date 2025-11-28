# Fruits Counting Ablation

This workspace tracks the implementation of the P2 and P3 counting pipelines described by the boss. It integrates the Kaggle fruits detection dataset with Meta's SAM3 attention backbone.

## Repo layout

- `docs/plan.md` – task overview and requirements recap.
- `configs/` – YAML configs for datasets, model hyper-parameters, and pipeline toggles.
- `src/` – reusable Python packages (data, models, pipelines, evaluation).
- `scripts/` – entry-point CLI scripts for training, inference, and evaluation.
- `data/` – local dataset downloads (ignored in git eventually).
- `results/` – experiment artifacts, logs, and metrics.
- `reports/` – human-readable experiment summaries.

## Next steps

1. Define environment + dependency files (PyTorch, SAM3, Kaggle API).
2. Implement modular pipeline runner that can switch between S0/S1 segmentation, D2 detection, and C1 counting.
3. Run P2 vs P3 experiments and capture metrics in `results/`.

## SAM3 setup reminder

SAM3 checkpoints are gated on Hugging Face. After installing the package (see `external/sam3`), run `huggingface-cli login` with an approved token so `sam3_detector.py` can download `sam3.pt` automatically. Without this step the evaluation scripts will fail with `Unauthorized` or missing checkpoint errors.
