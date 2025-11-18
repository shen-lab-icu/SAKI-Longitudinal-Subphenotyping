#!/usr/bin/env python3
"""Command-line utility for training and evaluating the AutoGluon model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from sa_aki_pipeline import ExperimentPaths, SplitConfig, TrainingConfig
from sa_aki_pipeline.data.split import DatasetSplitter, save_splits
from sa_aki_pipeline.evaluation.reporting import per_class_accuracy
from sa_aki_pipeline.modeling.autogluon_trainer import AutoGluonTrainer


EXTRA_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
    "f1_macro",
    "f1_weighted",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="Path to the merged feature CSV")
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Directory where models/results will be stored",
    )
    parser.add_argument(
        "--strategy",
        default="train2_test1",
        choices=["train1_test2", "train2_test1"],
        help="Dataset split configuration",
    )
    parser.add_argument(
        "--pivot-dataset",
        required=True,
        help="Dataset name referenced by the split strategy",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.7,
        help="Training fraction per dataset when using the train2_test1 strategy",
    )
    parser.add_argument(
        "--label",
        default="groupHPD",
        help="Target label column",
    )
    parser.add_argument(
        "--dataset-column",
        default="dataset",
        help="Column containing dataset identifiers",
    )
    parser.add_argument(
        "--id-column",
        default="stay_id",
        help="Column containing subject identifiers",
    )
    parser.add_argument(
        "--keep-dataset-column",
        action="store_true",
        help="Keep the dataset column during training",
    )
    parser.add_argument(
        "--keep-id-column",
        action="store_true",
        help="Keep the identifier column during training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv).expanduser().resolve()
    df = pd.read_csv(input_path)

    paths = ExperimentPaths(Path(args.experiment_dir))
    split_cfg = SplitConfig(
        strategy=args.strategy,
        pivot_dataset=args.pivot_dataset,
        train_fraction=args.holdout_fraction,
        dataset_column=args.dataset_column,
        id_column=args.id_column,
        label_column=args.label,
        drop_dataset_column=not args.keep_dataset_column,
    )

    splitter = DatasetSplitter(split_cfg)
    product = splitter.split(df)
    save_splits(product, paths.input_dir)
    product.summary.to_csv(paths.result_dir / "partition_summary.csv", index=False)

    drop_cols: List[str] = []
    if not args.keep_id_column:
        drop_cols.append(args.id_column)

    def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.drop(columns=[col for col in drop_cols if col in frame.columns])

    train_df = _prepare(product.train)
    trainer = AutoGluonTrainer(paths, TrainingConfig(label=args.label))
    trainer.fit(train_df)

    datasets = [("train", train_df)]
    if product.validation is not None:
        datasets.append(("validation", _prepare(product.validation)))
    for name, test_df in product.tests.items():
        datasets.append((f"test_{name}", _prepare(test_df)))

    for name, dataset in datasets:
        lb = trainer.leaderboard(dataset, name=name, extra_metrics=EXTRA_METRICS)
        lb.dataframe.to_csv(paths.result_dir / f"leaderboard_{name}.csv", index=False)

        preds = trainer.predict(dataset.drop(columns=[args.label]))
        stats = per_class_accuracy(
            pd.DataFrame({args.label: dataset[args.label], "pred": preds}),
            label_col=args.label,
            pred_col="pred",
        )
        stats.to_csv(paths.result_dir / f"per_class_{name}.csv", index=False)

    print("Training complete. Results stored in", paths.result_dir)


if __name__ == "__main__":
    main()
