"""Tests for mixAK model selection utilities."""

from __future__ import annotations

import numpy as np

from sa_aki_pipeline.phenotyping.config import MixAKModelSelection
from sa_aki_pipeline.phenotyping.mixak import evaluate_model_selection


def test_evaluate_model_selection():
    """Test model selection logic matches original notebook criteria."""
    # Simulated data from MIMIC notebook
    models = [
        MixAKModelSelection(k=2, deviance=1464777, autocorr_failed_fraction=0.0, uncertain_obs=112),
        MixAKModelSelection(k=3, deviance=1462740, autocorr_failed_fraction=0.0, uncertain_obs=383),
        MixAKModelSelection(k=4, deviance=1844490, autocorr_failed_fraction=0.75, uncertain_obs=95),
        MixAKModelSelection(k=5, deviance=1885722, autocorr_failed_fraction=0.77, uncertain_obs=360),
    ]
    
    # Mock config with thresholds
    class MockConfig:
        autocorr_threshold = 0.1
        gelman_rubin_threshold = 1.1
    
    selected_k = evaluate_model_selection(models, MockConfig())
    
    # K=3 should be selected (lowest deviance, no autocorrelation issues)
    assert selected_k == 3


def test_mixak_model_selection_quality_score():
    """Ensure quality score computation runs without error."""
    model = MixAKModelSelection(
        k=3,
        deviance=1462740,
        autocorr_failed_fraction=0.0,
        uncertain_obs=383,
    )
    score = model.quality_score()
    assert score > 0
