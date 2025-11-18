"""Dataclasses describing visualization jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class HeatmapJobConfig:
    """Configuration for the trajectory heatmap plots."""

    input_csv: Path
    output_path: Path
    features: List[str]
    dataset_column: str = "dataset"
    group_column: str = "groupHPD"
    time_column: str = "time"
    normalization: str = "zscore"  # "zscore", "minmax", "none"
    normalize_by_dataset: bool = True
    dataset_order: Optional[List[str]] = None
    time_values: Optional[List[int]] = None
    group_labels: Optional[Dict[str, str]] = None
    cmap: str = "coolwarm"

    def resolve(self, base_dir: Optional[Path] = None) -> "HeatmapJobConfig":
        def _resolve(path: Path) -> Path:
            if base_dir is None:
                return path.expanduser().resolve()
            return (base_dir / path).expanduser().resolve()

        return HeatmapJobConfig(
            input_csv=_resolve(self.input_csv),
            output_path=_resolve(self.output_path),
            features=list(self.features),
            dataset_column=self.dataset_column,
            group_column=self.group_column,
            time_column=self.time_column,
            normalization=self.normalization,
            normalize_by_dataset=self.normalize_by_dataset,
            dataset_order=list(self.dataset_order) if self.dataset_order else None,
            time_values=list(self.time_values) if self.time_values else None,
            group_labels=dict(self.group_labels) if self.group_labels else None,
            cmap=self.cmap,
        )
