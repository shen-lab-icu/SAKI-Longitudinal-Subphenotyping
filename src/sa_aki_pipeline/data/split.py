"""Deterministic dataset splitting utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..config import SplitConfig
from ..utils.io import ensure_dir, save_csv


@dataclass
class SplitProduct:
    """Container describing the result of a dataset split."""

    train: pd.DataFrame
    validation: Optional[pd.DataFrame]
    tests: Dict[str, pd.DataFrame]
    summary: pd.DataFrame

    def drop_columns(self, columns: list[str]) -> "SplitProduct":
        """Return a copy with the provided columns removed when available."""

        def _drop(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df is None:
                return None
            to_drop = [col for col in columns if col in df.columns]
            if not to_drop:
                return df
            return df.drop(columns=to_drop)

        return SplitProduct(
            train=_drop(self.train),
            validation=_drop(self.validation),
            tests={name: _drop(df) for name, df in self.tests.items()},
            summary=self.summary,
        )


class DatasetSplitter:
    """Implements the two dataset strategies used in the original notebooks."""

    def __init__(self, config: SplitConfig) -> None:
        self.config = config
        self.config.validate()

    def split(self, df: pd.DataFrame) -> SplitProduct:
        if self.config.dataset_column not in df.columns:
            raise KeyError(
                f"Input dataframe must contain '{self.config.dataset_column}' column"
            )
        if self.config.id_column not in df.columns:
            raise KeyError(
                f"Input dataframe must contain '{self.config.id_column}' column"
            )

        if self.config.strategy == "train1_test2":
            product = self._train_one_test_two(df)
        else:
            product = self._train_two_test_one(df)

        if self.config.drop_dataset_column:
            product = product.drop_columns([self.config.dataset_column])
        return product

    def _train_one_test_two(self, df: pd.DataFrame) -> SplitProduct:
        pivot = self.config.pivot_dataset
        datasets = df[self.config.dataset_column].unique().tolist()
        if pivot not in datasets:
            raise ValueError(f"pivot_dataset '{pivot}' not found in dataframe")

        train_df = df[df[self.config.dataset_column] == pivot].reset_index(drop=True)
        tests: Dict[str, pd.DataFrame] = {}
        for dataset in datasets:
            if dataset == pivot:
                continue
            tests[dataset] = (
                df[df[self.config.dataset_column] == dataset]
                .reset_index(drop=True)
            )

        summary = self._summarize_partitions(train_df, None, tests)
        return SplitProduct(train=train_df, validation=None, tests=tests, summary=summary)

    def _train_two_test_one(self, df: pd.DataFrame) -> SplitProduct:
        holdout = self.config.pivot_dataset
        datasets = df[self.config.dataset_column].unique().tolist()
        if holdout not in datasets:
            raise ValueError(f"pivot_dataset '{holdout}' not found in dataframe")

        holdout_df = df[df[self.config.dataset_column] == holdout].reset_index(drop=True)
        pool = df[df[self.config.dataset_column] != holdout]

        rng = random.Random(self.config.seed)
        train_parts = []
        val_parts = []
        for dataset, group in pool.groupby(self.config.dataset_column):
            ids = group[self.config.id_column].unique().tolist()
            rng.shuffle(ids)
            cut = max(1, int(len(ids) * self.config.train_fraction))
            train_ids = ids[:cut]
            val_ids = ids[cut:]
            train_parts.append(group[group[self.config.id_column].isin(train_ids)])
            val_parts.append(group[group[self.config.id_column].isin(val_ids)])

        train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pool
        val_df = pd.concat(val_parts, ignore_index=True) if val_parts else None
        tests = {holdout: holdout_df}
        summary = self._summarize_partitions(train_df, val_df, tests)
        return SplitProduct(train=train_df, validation=val_df, tests=tests, summary=summary)

    def _summarize_partitions(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame],
        tests: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        rows = []
        rows.append(self._describe_partition("train", train_df))
        if val_df is not None:
            rows.append(self._describe_partition("validation", val_df))
        for name, df in tests.items():
            rows.append(self._describe_partition(f"test:{name}", df))
        return pd.DataFrame(rows)

    def _describe_partition(self, partition: str, df: pd.DataFrame) -> Dict[str, object]:
        return {
            "partition": partition,
            "rows": len(df),
            "subjects": df[self.config.id_column].nunique(),
        }


def save_splits(product: SplitProduct, output_dir: Path) -> None:
    """Persist the split artefacts to ``output_dir``."""

    ensure_dir(output_dir)
    save_csv(product.train, output_dir / "train.csv")
    if product.validation is not None:
        save_csv(product.validation, output_dir / "validation.csv")
    for name, df in product.tests.items():
        save_csv(df, output_dir / f"test_{name}.csv")
    save_csv(product.summary, output_dir / "summary.csv", index=False)
