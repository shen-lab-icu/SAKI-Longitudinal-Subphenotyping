"""Tests for the survival time statistics helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sa_aki_pipeline.survival.config import TimeIntervalConfig, TimeStatsDatasetConfig, TimeStatsJobConfig, TimeStatsPlotConfig
from sa_aki_pipeline.survival.time_stats import load_and_plot_time_stats, plot_time_stats, run_time_stats_job


def test_run_time_stats_job(tmp_path):
    events_csv = tmp_path / "events.csv"
    group_csv = tmp_path / "groups.csv"

    events = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "sepsis_onset": ["2020-01-01 00:00:00", "2020-01-02 00:00:00"],
            "saki_onset": ["2020-01-01 12:00:00", "2020-01-02 06:00:00"],
        }
    )
    events.to_csv(events_csv, index=False)

    groups = pd.DataFrame({"stay_id": [1, 2], "groupHPD": [1, 2]})
    groups.to_csv(group_csv, index=False)

    dataset_cfg = TimeStatsDatasetConfig(
        name="mimic",
        events_csv=events_csv,
        group_csv=group_csv,
        datetime_columns=["sepsis_onset", "saki_onset"],
    )
    job_cfg = TimeStatsJobConfig(
        datasets=[dataset_cfg],
        intervals=[
            TimeIntervalConfig(
                name="los_saki-sepsis",
                start_column="sepsis_onset",
                end_column="saki_onset",
                unit="hours",
            )
        ],
        output_csv=tmp_path / "out.csv",
    )

    result = run_time_stats_job(job_cfg)
    assert result.output_csv.exists()
    assert "los_saki-sepsis" in result.frame.columns
    assert result.frame.loc[0, "los_saki-sepsis"] == 12.0


def test_plot_time_stats(tmp_path):
    csv_path = tmp_path / "stats.csv"
    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "groupHPD": [1, 2, 1],
            "dataset": ["mimic", "mimic", "eicu"],
            "los_saki-sepsis": [12, 6, 18],
        }
    )
    df.to_csv(csv_path, index=False)

    plot_cfg = TimeStatsPlotConfig(
        input_csv=csv_path,
        metrics=["los_saki-sepsis"],
        output_dir=tmp_path / "plots",
    )

    outputs = load_and_plot_time_stats(plot_cfg)
    assert "los_saki-sepsis" in outputs
    assert outputs["los_saki-sepsis"].exists()
