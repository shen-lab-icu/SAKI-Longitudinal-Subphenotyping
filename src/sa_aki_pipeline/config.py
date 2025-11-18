"""Centralized configuration dataclasses used across the SA-AKI package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .utils.io import ensure_dir


@dataclass
class ExperimentPaths:
    """Resolved directories for an experiment run."""

    root: Path

    def __post_init__(self) -> None:
        self.root = self.root.expanduser().resolve()
        self.input_dir = self.root / "input"
        self.result_dir = self.root / "result"
        ensure_dir(self.input_dir)
        ensure_dir(self.result_dir)


@dataclass
class SplitConfig:
    """Dataset split strategy definition.

    Attributes
    ----------
    strategy:
        Either ``"train1_test2"`` (one dataset for training, the others for testing)
        or ``"train2_test1"`` (two datasets for training/validation, one holdout test).
    pivot_dataset:
        Dataset name referenced by ``strategy``. For ``train1_test2`` it is the training
        dataset. For ``train2_test1`` it is the holdout test dataset.
    train_fraction:
        Fraction of subject identifiers per training dataset that go to the training fold
        when ``strategy="train2_test1"``. The rest form the validation fold.
    seed:
        Random seed to keep the identifier split deterministic.
    dataset_column / id_column:
        Column names expected inside the feature table.
    drop_dataset_column:
        Whether to remove the dataset column from the returned splits.
    """

    strategy: str = "train2_test1"
    pivot_dataset: str = "aumcdb"
    train_fraction: float = 0.7
    seed: int = 42
    dataset_column: str = "dataset"
    id_column: str = "stay_id"
    label_column: str = "groupHPD"
    drop_dataset_column: bool = True

    def validate(self) -> None:
        if self.strategy not in {"train1_test2", "train2_test1"}:
            raise ValueError(
                "strategy must be either 'train1_test2' or 'train2_test1', "
                f"got {self.strategy!r}"
            )
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1 (exclusive)")


@dataclass
class TrainingConfig:
    """Configuration block for the AutoGluon trainer."""

    label: str = "groupHPD"
    metric: str = "roc_auc_ovo_macro"
    problem_type: str = "multiclass"
    presets: str = "best_quality"
    auto_stack: bool = True
    use_bag_holdout: bool = True
    fit_weighted_ensemble: bool = False
    excluded_model_types: List[str] = field(default_factory=lambda: ["KNN"])


@dataclass
class ShapConfig:
    """Configuration parameters for SHAP explainability."""

    baseline_size: int = 1000
    nsamples: int = 500
    model_name: Optional[str] = None
    random_state: int = 30
