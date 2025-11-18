"""Helpers for computing and plotting survival interval statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import TimeIntervalConfig, TimeStatsDatasetConfig, TimeStatsJobConfig, TimeStatsPlotConfig


@dataclass
class TimeStatsResult:
    """Return value for time stats jobs to simplify testing."""

    frame: pd.DataFrame
    output_csv: Path


def _convert_datetimes(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def _to_numeric(series: pd.Series, unit: str) -> pd.Series:
    if np.issubdtype(series.dtype, np.timedelta64):
        if unit == "raw":
            return series
        if unit == "days":
            return series / np.timedelta64(1, "D")
        # default to hours for timedelta inputs
        return series / np.timedelta64(1, "h")
    return series.astype(float)


def _compute_intervals(df: pd.DataFrame, intervals: List[TimeIntervalConfig]) -> pd.DataFrame:
    result = df.copy()
    for interval in intervals:
        delta = result[interval.end_column] - result[interval.start_column]
        values = _to_numeric(delta, interval.unit)
        if interval.unit == "days" and not np.issubdtype(values.dtype, np.timedelta64):
            values = values / 24.0
        if interval.round_decimals is not None:
            values = values.round(interval.round_decimals)
        result[interval.name] = values
    return result


def load_time_stats_dataset(
    cfg: TimeStatsDatasetConfig,
    intervals: List[TimeIntervalConfig],
) -> pd.DataFrame:
    events = pd.read_csv(cfg.events_csv)
    groups = pd.read_csv(cfg.group_csv)[[cfg.id_column, cfg.group_column]].drop_duplicates()
    merged = pd.merge(events, groups, on=cfg.id_column, how="right")
    merged = _convert_datetimes(merged, cfg.datetime_columns)
    enriched = _compute_intervals(merged, intervals)
    enriched["dataset"] = cfg.name
    renamed = enriched.rename(columns={cfg.id_column: "stay_id", cfg.group_column: "groupHPD"})
    ordered_cols = ["stay_id", "groupHPD", "dataset"] + [i.name for i in intervals]
    return renamed[ordered_cols]


def run_time_stats_job(config: TimeStatsJobConfig) -> TimeStatsResult:
    frames = [load_time_stats_dataset(cfg, config.intervals) for cfg in config.datasets]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(config.output_csv, index=False)
    return TimeStatsResult(frame=combined, output_csv=config.output_csv)


def plot_time_stats(
    df: pd.DataFrame,
    plot_cfg: TimeStatsPlotConfig,
) -> Dict[str, Path]:
    sns.set(style="whitegrid")
    output_paths: Dict[str, Path] = {}
    dataset_order = plot_cfg.dataset_order or sorted(df[plot_cfg.dataset_column].dropna().unique())
    group_order = plot_cfg.group_order or sorted(df[plot_cfg.group_column].dropna().unique())
    plot_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for metric in plot_cfg.metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        subset = df[[plot_cfg.dataset_column, plot_cfg.group_column, metric]].dropna()
        sns.boxplot(
            data=subset,
            x=plot_cfg.group_column,
            y=metric,
            hue=plot_cfg.dataset_column,
            order=group_order,
            hue_order=dataset_order,
            palette="Set2",
            ax=ax,
            showfliers=False,
        )
        ax.set_xlabel("Phenotype")
        ylabel = "Hours" if metric.lower().endswith("hours") else metric
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric} by cohort")
        ax.legend(title="Dataset", loc="upper right")
        fig.tight_layout()
        output_path = plot_cfg.output_dir / f"{metric}_boxplot.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        output_paths[metric] = output_path
    return output_paths


def load_and_plot_time_stats(plot_cfg: TimeStatsPlotConfig) -> Dict[str, Path]:
    df = pd.read_csv(plot_cfg.input_csv)
    return plot_time_stats(df, plot_cfg)
