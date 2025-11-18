#!/usr/bin/env python3
"""Render longitudinal heatmaps from a YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from sa_aki_pipeline.visualization.config import HeatmapJobConfig
from sa_aki_pipeline.visualization.heatmap import run_heatmap_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML file describing the heatmap job")
    return parser.parse_args()


def load_config(path: Path) -> HeatmapJobConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = HeatmapJobConfig(
        input_csv=Path(raw["input_csv"]),
        output_path=Path(raw["output_path"]),
        features=raw["features"],
        dataset_column=raw.get("dataset_column", "dataset"),
        group_column=raw.get("group_column", "groupHPD"),
        time_column=raw.get("time_column", "time"),
        normalization=raw.get("normalization", "zscore"),
        normalize_by_dataset=raw.get("normalize_by_dataset", True),
        dataset_order=raw.get("dataset_order"),
        time_values=raw.get("time_values"),
        group_labels=raw.get("group_labels"),
        cmap=raw.get("cmap", "coolwarm"),
    )
    return cfg.resolve(base_dir=path.parent)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    run_heatmap_job(config)
    print(f"Heatmap written to {config.output_path}")


if __name__ == "__main__":
    main()
