#!/usr/bin/env python3
"""Generate treatment usage polar plots from YAML instructions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from sa_aki_pipeline.survival.config import TreatmentDatasetConfig, TreatmentUsageJobConfig
from sa_aki_pipeline.survival.treatment import plot_all_treatment_polar, run_treatment_usage_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML file describing the treatment usage job")
    return parser.parse_args()


def _load_dataset(raw: Dict[str, Any]) -> TreatmentDatasetConfig:
    return TreatmentDatasetConfig(
        name=raw["name"],
        treatment_csv=Path(raw["treatment_csv"]),
        group_csv=Path(raw["group_csv"]),
        id_column=raw.get("id_column", "stay_id"),
        group_column=raw.get("group_column", "groupHPD"),
        fill_value=raw.get("fill_value", 0.0),
    )


def load_config(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    datasets = [_load_dataset(item) for item in raw["datasets"]]
    usage_cfg = TreatmentUsageJobConfig(
        datasets=datasets,
        treatment_columns=raw["treatment_columns"],
        output_dir=Path(raw["output_dir"]),
        output_csv=Path(raw["output_csv"]) if raw.get("output_csv") else None,
    ).resolve(base_dir=path.parent)
    plot_meta = {
        "dataset_order": raw.get("dataset_order"),
        "group_order": raw.get("group_order"),
        "treatment_order": raw.get("treatment_columns"),
    }
    return {"usage": usage_cfg, "plot": plot_meta}


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    summary = run_treatment_usage_job(config["usage"])
    dataset_order = config["plot"]["dataset_order"] or sorted(summary["dataset"].unique())
    group_order = config["plot"]["group_order"] or sorted(summary["group"].unique(), key=lambda x: str(x))
    treatment_order = config["plot"]["treatment_order"]
    outputs = plot_all_treatment_polar(summary, config["usage"].output_dir, dataset_order, group_order, treatment_order)
    for dataset, image_path in outputs.items():
        print(f"{dataset}: {image_path}")


if __name__ == "__main__":
    main()
