"""Dataclasses describing survival/treatment jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

IntervalUnit = Literal["raw", "hours", "days"]


def _resolve_path(path: Path, base_dir: Optional[Path]) -> Path:
    if path is None:
        raise ValueError("path cannot be None")
    if base_dir is None:
        return Path(path).expanduser().resolve()
    return (base_dir / path).expanduser().resolve()


@dataclass
class TimeIntervalConfig:
    """How to compute a duration column."""

    name: str
    start_column: str
    end_column: str
    unit: IntervalUnit = "hours"
    round_decimals: Optional[int] = 2


@dataclass
class TimeStatsDatasetConfig:
    """Input sources for one cohort when computing survival intervals."""

    name: str
    events_csv: Path
    group_csv: Path
    id_column: str = "stay_id"
    group_column: str = "groupHPD"
    datetime_columns: List[str] = field(default_factory=list)

    def resolve(self, base_dir: Optional[Path] = None) -> "TimeStatsDatasetConfig":
        return TimeStatsDatasetConfig(
            name=self.name,
            events_csv=_resolve_path(Path(self.events_csv), base_dir),
            group_csv=_resolve_path(Path(self.group_csv), base_dir),
            id_column=self.id_column,
            group_column=self.group_column,
            datetime_columns=list(self.datetime_columns),
        )


@dataclass
class TimeStatsJobConfig:
    """Top-level configuration for the duration aggregation script."""

    datasets: List[TimeStatsDatasetConfig]
    intervals: List[TimeIntervalConfig]
    output_csv: Path

    def resolve(self, base_dir: Optional[Path] = None) -> "TimeStatsJobConfig":
        resolved = [cfg.resolve(base_dir) for cfg in self.datasets]
        output = _resolve_path(Path(self.output_csv), base_dir)
        return TimeStatsJobConfig(datasets=resolved, intervals=list(self.intervals), output_csv=output)


@dataclass
class TimeStatsPlotConfig:
    """Configuration for rendering boxplots of survival intervals."""

    input_csv: Path
    metrics: List[str]
    output_dir: Path
    dataset_column: str = "dataset"
    group_column: str = "groupHPD"
    dataset_order: Optional[List[str]] = None
    group_order: Optional[List[str]] = None

    def resolve(self, base_dir: Optional[Path] = None) -> "TimeStatsPlotConfig":
        return TimeStatsPlotConfig(
            input_csv=_resolve_path(Path(self.input_csv), base_dir),
            metrics=list(self.metrics),
            output_dir=_resolve_path(Path(self.output_dir), base_dir),
            dataset_column=self.dataset_column,
            group_column=self.group_column,
            dataset_order=list(self.dataset_order) if self.dataset_order else None,
            group_order=list(self.group_order) if self.group_order else None,
        )


@dataclass
class TreatmentDatasetConfig:
    """Dataset inputs for treatment usage summaries."""

    name: str
    treatment_csv: Path
    group_csv: Path
    id_column: str = "stay_id"
    group_column: str = "groupHPD"
    fill_value: float = 0.0

    def resolve(self, base_dir: Optional[Path] = None) -> "TreatmentDatasetConfig":
        return TreatmentDatasetConfig(
            name=self.name,
            treatment_csv=_resolve_path(Path(self.treatment_csv), base_dir),
            group_csv=_resolve_path(Path(self.group_csv), base_dir),
            id_column=self.id_column,
            group_column=self.group_column,
            fill_value=self.fill_value,
        )


@dataclass
class TreatmentUsageJobConfig:
    """Configuration for treatment usage polar plots."""

    datasets: List[TreatmentDatasetConfig]
    treatment_columns: List[str]
    output_dir: Path
    output_csv: Optional[Path] = None

    def resolve(self, base_dir: Optional[Path] = None) -> "TreatmentUsageJobConfig":
        resolved = [cfg.resolve(base_dir) for cfg in self.datasets]
        output_dir = _resolve_path(Path(self.output_dir), base_dir)
        output_csv = _resolve_path(Path(self.output_csv), base_dir) if self.output_csv else None
        return TreatmentUsageJobConfig(
            datasets=resolved,
            treatment_columns=list(self.treatment_columns),
            output_dir=output_dir,
            output_csv=output_csv,
        )
