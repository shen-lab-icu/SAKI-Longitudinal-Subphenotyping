#!/usr/bin/env python3
"""Compute SHAP values for a saved AutoGluon predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sa_aki_pipeline import ExperimentPaths, ShapConfig, TrainingConfig
from sa_aki_pipeline.explainability.shap_runner import compute_shap_values
from sa_aki_pipeline.modeling.autogluon_trainer import AutoGluonTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, help="Directory with the trained model")
    parser.add_argument("--dataset", required=True, help="CSV file to explain (must contain label)")
    parser.add_argument("--output-dir", required=True, help="Directory to write SHAP outputs")
    parser.add_argument("--label", default="groupHPD", help="Target label column")
    parser.add_argument("--baseline-size", type=int, default=1000)
    parser.add_argument("--nsamples", type=int, default=500)
    parser.add_argument("--model-name", help="Optional AutoGluon model to explain")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ExperimentPaths(Path(args.experiment_dir))
    dataset = pd.read_csv(Path(args.dataset).expanduser().resolve())

    trainer = AutoGluonTrainer(paths, TrainingConfig(label=args.label))
    trainer.load()

    shap_cfg = ShapConfig(
        baseline_size=args.baseline_size,
        nsamples=args.nsamples,
        model_name=args.model_name,
    )
    result = compute_shap_values(
        trainer=trainer,
        dataset=dataset,
        config=shap_cfg,
        training_config=TrainingConfig(label=args.label),
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_label, frame in result.values.items():
        frame.to_csv(output_dir / f"shap_{class_label}.csv", index=False)

    metadata = {
        "expected_values": result.expected_values,
        "classes": list(result.values.keys()),
        "model_name": args.model_name or trainer.get_best_model(),
    }
    (output_dir / "shap_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("SHAP values stored under", output_dir)


if __name__ == "__main__":
    main()
