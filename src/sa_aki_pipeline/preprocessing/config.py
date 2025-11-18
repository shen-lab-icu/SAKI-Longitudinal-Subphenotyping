"""Configuration models used by the preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

TimeUnit = Literal["minutes", "hours", "datetime"]


@dataclass
class MICEConfig:
    """Configuration for Multiple Imputation by Chained Equations."""

    n_imputations: int = 10
    iterations: int = 20
    random_state: int = 42
    save_missingness_report: bool = True


@dataclass
class AggregationConfig:
    """How to aggregate numeric features within a time bucket."""

    mean_columns: Optional[List[str]] = None
    sum_columns: Optional[List[str]] = None

    def build(self, columns: List[str]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        mean_cols = set(self.mean_columns or [])
        sum_cols = set(self.sum_columns or [])
        for col in columns:
            if col in sum_cols:
                mapping[col] = "sum"
            elif col in mean_cols or not sum_cols:
                mapping[col] = "mean"
            else:
                mapping[col] = "mean"
        return mapping


@dataclass
class DatasetConfig:
    """Configuration describing one dataset's time-window generation."""

    name: str
    label_csv: Path
    feature_csv: Path
    events_csv: Path
    id_column: str = "stay_id"
    label_column: str = "groupHPD"
    charttime_column: str = "charttime"
    onset_column: str = "saki_onset"
    dataset_column_value: Optional[str] = None
    drop_columns: List[str] = field(default_factory=list)
    time_unit: TimeUnit = "datetime"
    window_hours: tuple[int, int] = (-48, 168)
    bucket_hours: int = 24
    shift_non_negative_bucket: bool = True
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    mice_config: MICEConfig = field(default_factory=MICEConfig)

    def resolve(self, base_dir: Optional[Path] = None) -> "DatasetConfig":
        def _resolve(path: Path) -> Path:
            if base_dir is None:
                return path.expanduser().resolve()
            return (base_dir / path).expanduser().resolve()

        return DatasetConfig(
            name=self.name,
            label_csv=_resolve(Path(self.label_csv)),
            feature_csv=_resolve(Path(self.feature_csv)),
            events_csv=_resolve(Path(self.events_csv)),
            id_column=self.id_column,
            label_column=self.label_column,
            charttime_column=self.charttime_column,
            onset_column=self.onset_column,
            dataset_column_value=self.dataset_column_value or self.name,
            drop_columns=list(self.drop_columns),
            time_unit=self.time_unit,
            window_hours=self.window_hours,
            bucket_hours=self.bucket_hours,
            shift_non_negative_bucket=self.shift_non_negative_bucket,
            aggregation=self.aggregation,
            mice_config=self.mice_config,
        )


@dataclass
class MergeJobConfig:
    """Top-level configuration consumed by the CLI."""

    datasets: List[DatasetConfig]
    output_csv: Path

    def resolve(self, base_dir: Optional[Path] = None) -> "MergeJobConfig":
        resolved = [cfg.resolve(base_dir) for cfg in self.datasets]
        out = (base_dir / self.output_csv).resolve() if base_dir else self.output_csv.resolve()
        return MergeJobConfig(datasets=resolved, output_csv=out)
