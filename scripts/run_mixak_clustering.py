#!/usr/bin/env python3
"""Run mixAK clustering from YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from sa_aki_pipeline.phenotyping.config import MixAKConfig
from sa_aki_pipeline.phenotyping.mixak import run_mixak_clustering


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML file with mixAK parameters")
    return parser.parse_args()


def load_config(path: Path) -> MixAKConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = MixAKConfig(
        input_csv=Path(raw["input_csv"]),
        id_column=raw.get("id_column", "stay_id"),
        time_column=raw.get("time_column", "time"),
        features=raw["features"],
        k_min=raw.get("k_min", 2),
        k_max=raw.get("k_max", 5),
        burn=raw.get("burn", 50),
        keep=raw.get("keep", 2000),
        thin=raw.get("thin", 50),
        info=raw.get("info", 50),
        autocorr_threshold=raw.get("autocorr_threshold", 0.1),
        gelman_rubin_threshold=raw.get("gelman_rubin_threshold", 1.1),
        output_dir=Path(raw.get("output_dir", "outputs/mixak")),
        save_diagnostics=raw.get("save_diagnostics", True),
    )
    return cfg


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    
    print(f"Running mixAK clustering for K={cfg.k_min}..{cfg.k_max}")
    results = run_mixak_clustering(cfg)
    
    print(f"\nSelected K={results.selected_k}")
    print(f"Phenotype assignments saved to: {results.output_csv}")
    if results.diagnostics_txt:
        print(f"Model selection report: {results.diagnostics_txt}")
    
    print("\nModel Selection Summary:")
    print(results.summary_table())


if __name__ == "__main__":
    main()
