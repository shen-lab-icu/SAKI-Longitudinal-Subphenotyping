#!/usr/bin/env python3
"""Render boxplots for survival intervals."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from sa_aki_pipeline.survival.config import TimeStatsPlotConfig
from sa_aki_pipeline.survival.time_stats import load_and_plot_time_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML file describing the plot job")
    return parser.parse_args()


def load_config(path: Path) -> TimeStatsPlotConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = TimeStatsPlotConfig(
        input_csv=Path(raw["input_csv"]),
        metrics=raw["metrics"],
        output_dir=Path(raw["output_dir"]),
        dataset_column=raw.get("dataset_column", "dataset"),
        group_column=raw.get("group_column", "groupHPD"),
        dataset_order=raw.get("dataset_order"),
        group_order=raw.get("group_order"),
    )
    return cfg.resolve(base_dir=path.parent)


def main() -> None:
    args = parse_args()
    path = Path(args.config).expanduser().resolve()
    cfg = load_config(path)
    outputs = load_and_plot_time_stats(cfg)
    for metric, image_path in outputs.items():
        print(f"{metric}: {image_path}")


if __name__ == "__main__":
    main()
