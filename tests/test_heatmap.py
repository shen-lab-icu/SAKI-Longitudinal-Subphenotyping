"""Tests for heatmap preparation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sa_aki_pipeline.visualization.heatmap import normalize_features, prepare_heatmap_data


def _mock_frame() -> pd.DataFrame:
    rows = []
    for dataset in ["mimic", "aumcdb"]:
        for group in [1, 2]:
            for time in [0, 1]:
                rows.append(
                    {
                        "dataset": dataset,
                        "groupHPD": group,
                        "time": time,
                        "Temperature": 36 + group + time,
                        "Creatinine": 1.0 + 0.1 * group + 0.2 * time,
                    }
                )
    return pd.DataFrame(rows)


def test_normalize_features_by_dataset():
    df = _mock_frame()
    normalized = normalize_features(
        df,
        ["Temperature", "Creatinine"],
        dataset_column="dataset",
        method="zscore",
        by_dataset=True,
    )
    # Mean of each dataset block should be close to 0 after z-score
    grouped = normalized.groupby("dataset")["Temperature"].mean().round(6)
    assert np.allclose(grouped.values, 0.0)


def test_prepare_heatmap_data_shapes():
    df = _mock_frame()
    prep = prepare_heatmap_data(
        df,
        features=["Temperature", "Creatinine"],
        dataset_column="dataset",
        group_column="groupHPD",
        time_column="time",
        dataset_order=["mimic", "aumcdb"],
        time_values=[0, 1],
    )
    assert set(prep.matrices.keys()) == {"1", "2"}
    matrix = prep.matrices["1"]
    # 2 datasets * 2 features rows, 2 time points columns
    assert matrix.shape == (4, 2)
    assert list(prep.time_values) == [0, 1]
