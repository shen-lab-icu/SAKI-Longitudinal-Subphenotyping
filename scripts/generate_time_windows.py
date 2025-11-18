#!/usr/bin/env python3
"""Generate time-windowed feature tables for multiple datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from sa_aki_pipeline.preprocessing.config import AggregationConfig, DatasetConfig, MergeJobConfig
from sa_aki_pipeline.preprocessing.time_windows import merge_datasets
from sa_aki_pipeline.utils.io import ensure_dir, save_csv


def _load_dataset_config(raw: Dict[str, Any], base_dir: Path) -> DatasetConfig:
    agg_cfg = raw.get("aggregation", {})
    aggregation = AggregationConfig(
        mean_columns=agg_cfg.get("mean_columns"),
        sum_columns=agg_cfg.get("sum_columns"),
    )
    dataset = DatasetConfig(
        name=raw["name"],
        label_csv=base_dir / raw["label_csv"],
        feature_csv=base_dir / raw["feature_csv"],
        events_csv=base_dir / raw["events_csv"],
        id_column=raw.get("id_column", "stay_id"),
        label_column=raw.get("label_column", "groupHPD"),
        charttime_column=raw.get("charttime_column", "charttime"),
        onset_column=raw.get("onset_column", "saki_onset"),
        dataset_column_value=raw.get("dataset_column_value"),
        drop_columns=raw.get("drop_columns", []),
        time_unit=raw.get("time_unit", "datetime"),
        window_hours=tuple(raw.get("window_hours", (-48, 168))),
        bucket_hours=raw.get("bucket_hours", 24),
        shift_non_negative_bucket=raw.get("shift_non_negative_bucket", True),
        aggregation=aggregation,
    )
    return dataset


def load_job_config(path: Path) -> MergeJobConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_dir = path.parent
    datasets = [_load_dataset_config(entry, base_dir) for entry in data["datasets"]]
    output_csv = base_dir / data["output_csv"]
    return MergeJobConfig(datasets=datasets, output_csv=output_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML file describing the merge job")
    parser.add_argument(
        "--output-csv",
        help="Override the output CSV path defined in the config",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print partition statistics as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    job = load_job_config(config_path)
    df = merge_datasets(job.datasets)

    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else job.output_csv
    ensure_dir(output_csv.parent)
    save_csv(df, output_csv)

    if args.print_summary:
        summary = {
            "output_csv": str(output_csv),
            "rows": len(df),
            "datasets": df["dataset"].value_counts().to_dict() if "dataset" in df else {},
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
