"""Dataset time-window generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

try:  # Optional dependency; informative warning when missing.
    import miceforest as mf
except Exception:  # pragma: no cover
    mf = None  # type: ignore[assignment]

from .config import DatasetConfig
from ..utils.io import ensure_dir, save_csv


def _compute_relative_hours(
    frame: pd.DataFrame,
    chart_col: str,
    onset_col: str,
    time_unit: str,
) -> pd.Series:
    if time_unit == "datetime":
        chart = pd.to_datetime(frame[chart_col])
        onset = pd.to_datetime(frame[onset_col])
        delta = chart - onset
        hours = delta / np.timedelta64(1, "h")
    else:
        chart = frame[chart_col].astype(float)
        onset = frame[onset_col].astype(float)
        hours = chart - onset
        if time_unit == "minutes":
            hours = hours / 60.0
    return hours


def _bucketize(hours: pd.Series, bucket_hours: int, shift_non_negative: bool) -> pd.Series:
    buckets = np.floor_divide(hours, bucket_hours).astype(int)
    if shift_non_negative:
        buckets = buckets.where(buckets < 0, buckets + 1)
    return buckets


def _forward_fill_by_subject(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    filled = (
        df.sort_values([id_col, "time_bucket"])
        .groupby(id_col)
        .ffill()
        .reset_index(drop=True)
    )
    return filled


def _mice_impute(df: pd.DataFrame, config, output_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Apply MICE imputation with explicit parameters for Methods section reproducibility.

    Parameters
    ----------
    df : pd.DataFrame
        Data to impute (forward-filled first).
    config : MICEConfig
        Imputation configuration (n_imputations, iterations, random_state).
    output_dir : Optional[Path]
        If provided, write missingness report to this directory.

    Returns
    -------
    pd.DataFrame
        Completed dataset after MICE.
    """
    if mf is None:  # pragma: no cover
        return df

    # Record missingness before imputation for Methods reporting
    missingness = df.isnull().mean().to_dict()
    if config.save_missingness_report and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        miss_df = pd.DataFrame(list(missingness.items()), columns=["variable", "missingness_fraction"])
        miss_df.to_csv(output_dir / "missingness_report.csv", index=False)

    kernel = mf.ImputationKernel(
        df,
        datasets=config.n_imputations,
        save_all_iterations=False,
        random_state=config.random_state,
    )
    kernel.mice(iterations=config.iterations, n_jobs=-1)
    return kernel.complete_data(dataset=0)


def generate_time_windows(config: DatasetConfig) -> pd.DataFrame:
    """Generate aggregated time windows for a dataset according to *config*."""

    labels = pd.read_csv(config.label_csv)[[config.id_column, config.label_column]].drop_duplicates()
    events = pd.read_csv(config.events_csv)[[config.id_column, config.onset_column]]
    features = pd.read_csv(config.feature_csv)

    merged = (
        features.merge(events, on=config.id_column, how="inner")
        .merge(labels, on=config.id_column, how="inner")
    )
    if config.drop_columns:
        merged = merged.drop(columns=[col for col in config.drop_columns if col in merged.columns])

    merged["relative_hours"] = _compute_relative_hours(
        merged,
        chart_col=config.charttime_column,
        onset_col=config.onset_column,
        time_unit=config.time_unit,
    )

    start, end = config.window_hours
    merged = merged[(merged["relative_hours"] >= start) & (merged["relative_hours"] <= end)]
    merged["time_bucket"] = _bucketize(
        merged["relative_hours"],
        bucket_hours=config.bucket_hours,
        shift_non_negative=config.shift_non_negative_bucket,
    )

    numeric_cols = merged.select_dtypes(include=["number"]).columns.tolist()
    exclude = {
        config.id_column,
        config.label_column,
        "relative_hours",
        "time_bucket",
    }
    agg_candidates = [col for col in numeric_cols if col not in exclude]
    agg_mapping = config.aggregation.build(agg_candidates)

    aggregated = (
        merged.groupby([config.id_column, "time_bucket"]).agg(agg_mapping).reset_index()
    )
    aggregated = aggregated.merge(labels, on=config.id_column, how="left")
    aggregated = aggregated.merge(
        merged[[config.id_column, "time_bucket"]].drop_duplicates(),
        on=[config.id_column, "time_bucket"],
        how="right",
    )

    filled = _forward_fill_by_subject(aggregated, config.id_column)
    output_dir = config.label_csv.parent if hasattr(config.label_csv, "parent") else None
    imputed = _mice_impute(filled, config.mice_config, output_dir=output_dir)
    imputed["dataset"] = config.dataset_column_value or config.name
    return imputed


def merge_datasets(configs: List[DatasetConfig]) -> pd.DataFrame:
    """Concatenate the outputs of multiple dataset configurations."""

    frames = [generate_time_windows(cfg) for cfg in configs]
    return pd.concat(frames, ignore_index=True)


def run_merge_job(configs: List[DatasetConfig], output_csv: str | None = None) -> pd.DataFrame:
    df = merge_datasets(configs)
    if output_csv:
        output_path = Path(output_csv).expanduser().resolve()
        ensure_dir(output_path.parent)
        save_csv(df, output_path)
    return df
