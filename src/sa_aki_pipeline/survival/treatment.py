"""Summaries and plots for treatment usage patterns."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import TreatmentDatasetConfig, TreatmentUsageJobConfig


def summarize_treatment_usage(cfg: TreatmentDatasetConfig, treatment_columns: List[str]) -> pd.DataFrame:
    treatments = pd.read_csv(cfg.treatment_csv)
    groups = pd.read_csv(cfg.group_csv)[[cfg.id_column, cfg.group_column]].drop_duplicates()
    merged = pd.merge(groups, treatments, on=cfg.id_column, how="left").fillna(cfg.fill_value)
    records: List[Dict[str, float]] = []
    for treatment in treatment_columns:
        if treatment not in merged.columns:
            raise ValueError(f"Column '{treatment}' not found in {cfg.treatment_csv}")
        group_means = merged.groupby(cfg.group_column)[treatment].mean()
        for group, percent in group_means.items():
            records.append(
                {
                    "dataset": cfg.name,
                    "group": str(group),
                    "treatment": treatment,
                    "percent": float(percent),
                }
            )
    return pd.DataFrame(records)


def run_treatment_usage_job(config: TreatmentUsageJobConfig) -> pd.DataFrame:
    frames = [summarize_treatment_usage(cfg, config.treatment_columns) for cfg in config.datasets]
    combined = pd.concat(frames, ignore_index=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.output_csv:
        combined.to_csv(config.output_csv, index=False)
    return combined


def _build_polar_positions(num_groups: int, num_treatments: int) -> np.ndarray:
    total = num_groups * num_treatments
    return np.linspace(0.0, 2 * np.pi, total, endpoint=False)


def plot_treatment_polar(
    df: pd.DataFrame,
    dataset: str,
    output_dir: Path,
    group_order: List[str],
    treatment_order: List[str],
) -> Path:
    subset = df[df["dataset"] == dataset]
    if subset.empty:
        raise ValueError(f"Dataset '{dataset}' not found in treatment summary")

    num_groups = len(group_order)
    num_treatments = len(treatment_order)
    theta = _build_polar_positions(num_groups, num_treatments)
    radii: List[float] = []
    colors: List[str] = []
    color_map = plt.get_cmap("tab10")

    for treatment in treatment_order:
        for idx, group in enumerate(group_order):
            value = subset[(subset["group"] == str(group)) & (subset["treatment"] == treatment)]["percent"].values
            percent = value[0] if len(value) else 0.0
            radii.append(percent)
            colors.append(color_map(idx))

    width = (2 * np.pi) / (num_groups * num_treatments) * 0.9
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="polar")
    bars = ax.bar(theta, radii, width=width, bottom=0.0, align="edge", linewidth=0.8, edgecolor="black")
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)
    ax.set_title(f"{dataset} treatment usage")
    ax.set_yticklabels([])
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_map(idx)) for idx in range(num_groups)]
    ax.legend(legend_handles, [str(g) for g in group_order], loc="upper right", title="Phenotype")
    output_path = output_dir / f"{dataset}_treatment_polar.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_all_treatment_polar(
    df: pd.DataFrame,
    output_dir: Path,
    dataset_order: List[str],
    group_order: List[str],
    treatment_order: List[str],
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for dataset in dataset_order:
        paths[dataset] = plot_treatment_polar(df, dataset, output_dir, group_order, treatment_order)
    return paths
