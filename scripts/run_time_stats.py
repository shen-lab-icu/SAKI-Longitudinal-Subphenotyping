#!/usr/bin/env python3
"""Generate survival interval summaries from YAML instructions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from sa_aki_pipeline.survival.config import (
    TimeIntervalConfig,
    TimeStatsDatasetConfig,
    TimeStatsJobConfig,
)
from sa_aki_pipeline.survival.time_stats import run_time_stats_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML file describing the time stats job")
    return parser.parse_args()


def _load_interval(raw: Dict[str, Any]) -> TimeIntervalConfig:
    return TimeIntervalConfig(
        name=raw["name"],
        start_column=raw["start_column"],
        end_column=raw["end_column"],
        unit=raw.get("unit", "hours"),
        round_decimals=raw.get("round_decimals", 2),
    )


def _load_dataset(raw: Dict[str, Any]) -> TimeStatsDatasetConfig:
    return TimeStatsDatasetConfig(
        name=raw["name"],
        events_csv=Path(raw["events_csv"]),
        group_csv=Path(raw["group_csv"]),
        id_column=raw.get("id_column", "stay_id"),
        group_column=raw.get("group_column", "groupHPD"),
        datetime_columns=raw.get("datetime_columns", []),
    )


def load_config(path: Path) -> TimeStatsJobConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    datasets = [_load_dataset(item) for item in raw["datasets"]]
    intervals = [_load_interval(item) for item in raw["intervals"]]
    cfg = TimeStatsJobConfig(
        datasets=datasets,
        intervals=intervals,
        output_csv=Path(raw["output_csv"]),
    )
    return cfg.resolve(base_dir=path.parent)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    result = run_time_stats_job(cfg)
    print(f"Time stats saved to {result.output_csv}")


if __name__ == "__main__":
    main()
