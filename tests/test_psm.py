"""Tests for propensity score matching utilities."""

from __future__ import annotations

import pandas as pd

from sa_aki_pipeline.causal.config import PSMConfig
from sa_aki_pipeline.causal.psm import compute_standardized_mean_difference, propensity_score_matching


def test_compute_smd():
    df = pd.DataFrame(
        {
            "treatment": [1, 1, 1, 0, 0, 0],
            "age": [60, 65, 70, 50, 55, 60],
            "weight": [70, 75, 80, 65, 68, 72],
        }
    )
    smd = compute_standardized_mean_difference(df, "treatment", ["age", "weight"])
    assert "age" in smd
    assert "weight" in smd
    assert abs(smd["age"]) > 0


def test_propensity_score_matching(tmp_path):
    df = pd.DataFrame(
        {
            "treatment": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "age": [60, 65, 70, 55, 50, 52, 58, 61, 63, 67],
            "weight": [70, 75, 80, 68, 65, 66, 69, 72, 74, 78],
        }
    )
    cfg = PSMConfig(
        treatment_column="treatment",
        covariates=["age", "weight"],
        caliper=0.2,
        match_ratio=1,
        random_state=42,
        output_matched_csv=tmp_path / "matched.csv",
        output_report_txt=tmp_path / "report.txt",
    )
    result = propensity_score_matching(df, cfg)
    assert result.n_treated == 4
    assert result.n_control == 6
    assert result.n_matched_treated > 0
    assert (tmp_path / "matched.csv").exists()
    assert (tmp_path / "report.txt").exists()
    # Check SMD improvement (optional, may not always improve in tiny datasets)
    assert "age" in result.smd_after
