"""Utilities to produce multi-panel trajectory heatmaps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import HeatmapJobConfig


@dataclass
class HeatmapData:
    """Prepared matrices ready for plotting."""

    matrices: Dict[str, np.ndarray]
    features: List[str]
    time_values: List[int]
    dataset_order: List[str]


def _normalize_block(df: pd.DataFrame, features: List[str], method: str) -> pd.DataFrame:
    if method == "none":
        return df
    if method == "zscore":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    df[features] = scaler.fit_transform(df[features])
    return df


def normalize_features(
    frame: pd.DataFrame,
    features: List[str],
    dataset_column: str,
    method: str,
    by_dataset: bool,
) -> pd.DataFrame:
    if method == "none":
        return frame
    if by_dataset:
        parts = []
        for dataset, group in frame.groupby(dataset_column):
            parts.append(_normalize_block(group.copy(), features, method))
        return pd.concat(parts, ignore_index=True)
    return _normalize_block(frame.copy(), features, method)


def prepare_heatmap_data(
    df: pd.DataFrame,
    *,
    features: List[str],
    dataset_column: str = "dataset",
    group_column: str = "groupHPD",
    time_column: str = "time",
    dataset_order: Optional[List[str]] = None,
    time_values: Optional[Iterable[int]] = None,
) -> HeatmapData:
    dataset_order = dataset_order or sorted(df[dataset_column].dropna().unique())
    if time_values is None:
        time_values = sorted(df[time_column].dropna().unique())
    time_values = list(time_values)

    matrices: Dict[str, np.ndarray] = {}
    for group, group_df in df.groupby(group_column):
        blocks: List[np.ndarray] = []
        for dataset in dataset_order:
            subset = group_df[group_df[dataset_column] == dataset]
            if subset.empty:
                blocks.append(np.zeros((len(features), len(time_values))))
                continue
            pivot = (
                subset
                .set_index(time_column)[features]
                .reindex(time_values)
                .T
                .fillna(0)
            )
            blocks.append(pivot.values)
        matrices[str(group)] = np.vstack(blocks)
    return HeatmapData(
        matrices=matrices,
        features=features,
        time_values=time_values,
        dataset_order=dataset_order,
    )


def plot_heatmap_panels(
    data: HeatmapData,
    *,
    cmap: str = "coolwarm",
    group_labels: Optional[Dict[str, str]] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (8, 5),
    alpha: float = 0.85,
) -> plt.Figure:
    groups = list(data.matrices.keys())
    fig, axes = plt.subplots(1, len(groups), figsize=figsize, sharey=True)
    if len(groups) == 1:
        axes = [axes]

    y_ticks: List[str] = []
    for dataset in data.dataset_order:
        y_ticks.extend([f"{dataset}::{feat}" for feat in data.features])

    for ax, group in zip(axes, groups):
        matrix = data.matrices[group]
        c = ax.pcolormesh(matrix, cmap=cmap, alpha=alpha)
        ax.set_title(group_labels.get(group, str(group)) if group_labels else str(group))
        ax.set_xticks(
            np.linspace(0.5, matrix.shape[1] - 0.5, len(data.time_values)),
            labels=[str(v) for v in data.time_values],
            rotation=45,
        )
        if ax is axes[0]:
            ax.set_yticks(
                np.linspace(0.5, matrix.shape[0] - 0.5, len(y_ticks)),
                labels=y_ticks,
                fontsize=8,
            )
        else:
            ax.set_yticks([])
    fig.tight_layout()
    fig.colorbar(c, ax=axes, shrink=0.75)
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def run_heatmap_job(config: HeatmapJobConfig) -> plt.Figure:
    df = pd.read_csv(config.input_csv)
    if config.time_values is not None:
        df = df[df[config.time_column].isin(config.time_values)]
    df = normalize_features(
        df,
        config.features,
        dataset_column=config.dataset_column,
        method=config.normalization,
        by_dataset=config.normalize_by_dataset,
    )
    prep = prepare_heatmap_data(
        df,
        features=config.features,
        dataset_column=config.dataset_column,
        group_column=config.group_column,
        time_column=config.time_column,
        dataset_order=config.dataset_order,
        time_values=config.time_values,
    )
    return plot_heatmap_panels(
        prep,
        cmap=config.cmap,
        group_labels=config.group_labels,
        output_path=str(config.output_path),
    )
