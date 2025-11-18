#!/usr/bin/env python3
"""Run propensity score matching from YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from sa_aki_pipeline.causal.config import PSMConfig
from sa_aki_pipeline.causal.psm import propensity_score_matching


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML file describing PSM parameters")
    parser.add_argument("--input-csv", required=True, help="Cohort CSV with treatment + covariates")
    return parser.parse_args()


def load_config(path: Path) -> PSMConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = PSMConfig(
        treatment_column=raw["treatment_column"],
        covariates=raw["covariates"],
        caliper=raw.get("caliper", 0.2),
        match_ratio=raw.get("match_ratio", 1),
        random_state=raw.get("random_state", 42),
        strategy=raw.get("strategy", "nearest"),
        replacement=raw.get("replacement", False),
        output_matched_csv=Path(raw["output_matched_csv"]) if raw.get("output_matched_csv") else None,
        output_report_txt=Path(raw["output_report_txt"]) if raw.get("output_report_txt") else None,
    )
    return cfg


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    df = pd.read_csv(args.input_csv)
    result = propensity_score_matching(df, cfg)
    print(f"Matched {result.n_matched_treated} treated + {result.n_matched_control} control")
    if cfg.output_matched_csv:
        print(f"Matched cohort saved to {cfg.output_matched_csv}")
    if cfg.output_report_txt:
        print(f"Diagnostics report written to {cfg.output_report_txt}")


if __name__ == "__main__":
    main()
