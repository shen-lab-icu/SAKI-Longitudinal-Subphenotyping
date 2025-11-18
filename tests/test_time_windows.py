"""Tests for the time-window preprocessing helpers."""

from __future__ import annotations

import pandas as pd

from sa_aki_pipeline.preprocessing.config import AggregationConfig, DatasetConfig
from sa_aki_pipeline.preprocessing.time_windows import generate_time_windows


def test_generate_time_windows(tmp_path):
    label_csv = tmp_path / "labels.csv"
    feature_csv = tmp_path / "features.csv"
    events_csv = tmp_path / "events.csv"

    labels = pd.DataFrame({"stay_id": [1, 2], "groupHPD": [1, 2]})
    labels.to_csv(label_csv, index=False)

    features = pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 2],
            "charttime": [0, 24, 0, 24],
            "creatinine": [1.0, 1.2, 1.5, 1.4],
            "urineoutput": [100, 150, 90, 110],
        }
    )
    features.to_csv(feature_csv, index=False)

    events = pd.DataFrame({"stay_id": [1, 2], "saki_onset": [0, 0]})
    events.to_csv(events_csv, index=False)

    cfg = DatasetConfig(
        name="mimic",
        label_csv=label_csv,
        feature_csv=feature_csv,
        events_csv=events_csv,
        time_unit="hours",
        window_hours=(-1, 48),
        bucket_hours=24,
        shift_non_negative_bucket=False,
        aggregation=AggregationConfig(sum_columns=["urineoutput"]),
    )

    result = generate_time_windows(cfg)
    assert set(result["dataset"].unique()) == {"mimic"}
    assert "groupHPD" in result.columns
    assert result.shape[0] >= 2
    # Ensure aggregation respected the "sum" instruction for urineoutput
    bucket_sum = result[result["time_bucket"] == 0]["urineoutput"].iloc[0]
    assert bucket_sum == 100
