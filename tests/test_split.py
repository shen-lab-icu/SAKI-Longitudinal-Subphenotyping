"""Smoke tests for dataset splitting utilities."""

from __future__ import annotations

import pandas as pd

from sa_aki_pipeline.config import SplitConfig
from sa_aki_pipeline.data.split import DatasetSplitter


def _toy_frame() -> pd.DataFrame:
    rows = []
    for dataset in ["mimic", "eicu", "aumcdb"]:
        for stay_id in range(6):
            rows.append(
                {
                    "dataset": dataset,
                    "stay_id": f"{dataset}_{stay_id}",
                    "groupHPD": (stay_id % 3) + 1,
                    "feature": stay_id,
                }
            )
    return pd.DataFrame(rows)


def test_train2_test1_split() -> None:
    df = _toy_frame()
    cfg = SplitConfig(strategy="train2_test1", pivot_dataset="aumcdb", train_fraction=0.5)
    splitter = DatasetSplitter(cfg)
    product = splitter.split(df)

    assert product.validation is not None
    assert set(product.tests.keys()) == {"aumcdb"}
    assert "dataset" not in product.train.columns
    assert product.summary["partition"].isin(["train", "validation", "test:aumcdb"]).all()


def test_train1_test2_split() -> None:
    df = _toy_frame()
    cfg = SplitConfig(strategy="train1_test2", pivot_dataset="mimic")
    splitter = DatasetSplitter(cfg)
    product = splitter.split(df)

    assert product.validation is None
    assert set(product.tests.keys()) == {"eicu", "aumcdb"}
    assert product.train["groupHPD"].notna().all()
