"""SHAP helpers compatible with AutoGluon predictors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import shap

from ..config import ShapConfig, TrainingConfig
from ..modeling.autogluon_trainer import AutoGluonTrainer


@dataclass
class ShapResult:
    values: Dict[str, pd.DataFrame]
    expected_values: List[float]


class AutogluonProbabilityWrapper:
    """Callable wrapper that exposes ``predict_proba`` with a stable schema."""

    def __init__(
        self,
        trainer: AutoGluonTrainer,
        feature_names: List[str],
        model_name: Optional[str] = None,
    ) -> None:
        self.trainer = trainer
        self.feature_names = feature_names
        self.model_name = model_name

    def __call__(self, data):
        if isinstance(data, pd.Series):
            data = data.to_frame().T
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=self.feature_names)
        predictor = self.trainer._require_predictor()  # noqa: SLF001
        proba = predictor.predict_proba(data, model=self.model_name)
        return proba.values


def compute_shap_values(
    trainer: AutoGluonTrainer,
    dataset: pd.DataFrame,
    config: ShapConfig,
    training_config: TrainingConfig,
) -> ShapResult:
    """Return SHAP values for the provided dataset."""

    predictor = trainer._require_predictor()  # noqa: SLF001
    features = dataset.drop(columns=[training_config.label])
    baseline = features.sample(
        n=min(config.baseline_size, len(features)),
        random_state=config.random_state,
    )

    wrapper = AutogluonProbabilityWrapper(
        trainer=trainer,
        feature_names=list(features.columns),
        model_name=config.model_name or predictor.get_model_best(),
    )
    explainer = shap.KernelExplainer(wrapper, baseline)
    shap_values = explainer.shap_values(features, nsamples=config.nsamples)

    # Multi-class models return one matrix per class
    class_labels = getattr(predictor, "class_labels", None)
    shap_frames: Dict[str, pd.DataFrame] = {}
    if isinstance(shap_values, list) and class_labels is not None:
        for label, values in zip(class_labels, shap_values):
            shap_frames[str(label)] = pd.DataFrame(values, columns=features.columns)
    else:
        shap_frames[predictor.problem_type] = pd.DataFrame(
            shap_values, columns=features.columns
        )

    expected_values = (
        explainer.expected_value
        if isinstance(explainer.expected_value, list)
        else [explainer.expected_value]
    )
    return ShapResult(values=shap_frames, expected_values=expected_values)
