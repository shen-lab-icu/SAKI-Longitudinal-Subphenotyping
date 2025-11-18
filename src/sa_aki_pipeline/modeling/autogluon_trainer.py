"""Encapsulation of the AutoGluon Tabular workflow used in the project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import confusion_matrix

from ..config import ExperimentPaths, TrainingConfig


@dataclass
class LeaderboardResult:
    """Small helper tying a leaderboard dataframe to its dataset name."""

    name: str
    dataframe: pd.DataFrame


class AutoGluonTrainer:
    """High-level interface used by the CLI scripts."""

    def __init__(self, paths: ExperimentPaths, config: TrainingConfig) -> None:
        self.paths = paths
        self.config = config
        self.predictor: Optional[TabularPredictor] = None

    def fit(self, train_df: pd.DataFrame) -> TabularPredictor:
        self.predictor = TabularPredictor(
            label=self.config.label,
            path=str(self.paths.root),
            problem_type=self.config.problem_type,
            eval_metric=self.config.metric,
        ).fit(
            train_df,
            presets=self.config.presets,
            auto_stack=self.config.auto_stack,
            use_bag_holdout=self.config.use_bag_holdout,
            fit_weighted_ensemble=self.config.fit_weighted_ensemble,
            excluded_model_types=self.config.excluded_model_types,
        )
        return self.predictor

    def load(self) -> TabularPredictor:
        self.predictor = TabularPredictor.load(str(self.paths.root))
        return self.predictor

    # pylint: disable=too-many-arguments
    def leaderboard(
        self,
        dataset: pd.DataFrame,
        name: str,
        extra_metrics: Optional[Iterable[str]] = None,
        silent: bool = True,
    ) -> LeaderboardResult:
        predictor = self._require_predictor()
        df = predictor.leaderboard(
            dataset,
            silent=silent,
            extra_metrics=list(extra_metrics) if extra_metrics else None,
        )
        return LeaderboardResult(name=name, dataframe=df)

    def predict(self, dataset: pd.DataFrame, model: Optional[str] = None) -> pd.Series:
        predictor = self._require_predictor()
        return predictor.predict(dataset, model=model)

    def confusion_matrix(
        self,
        dataset: pd.DataFrame,
        labels: Iterable[int],
        model: Optional[str] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        predictor = self._require_predictor()
        y_true = dataset[self.config.label]
        features = dataset.drop(columns=[self.config.label])
        y_pred = predictor.predict(features, model=model)
        cm = confusion_matrix(y_true, y_pred, labels=list(labels))
        if normalize:
            cm = cm.astype(float)
            cm = cm / cm.sum(axis=1, keepdims=True)
        return cm

    def get_best_model(self) -> str:
        predictor = self._require_predictor()
        return predictor.get_model_best()

    def _require_predictor(self) -> TabularPredictor:
        if self.predictor is None:
            raise RuntimeError("Predictor has not been trained or loaded yet")
        return self.predictor
