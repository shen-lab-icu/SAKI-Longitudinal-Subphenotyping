"""Tests for treatment usage summaries."""

from __future__ import annotations

import pandas as pd

from sa_aki_pipeline.survival.config import TreatmentDatasetConfig, TreatmentUsageJobConfig
from sa_aki_pipeline.survival.treatment import plot_all_treatment_polar, run_treatment_usage_job


def test_run_treatment_usage_job(tmp_path):
    treatment_csv = tmp_path / "treatments.csv"
    group_csv = tmp_path / "groups.csv"

    treatments = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "is_rrt": [1, 0, 1],
            "is_mv": [0, 1, 1],
            "is_vaso": [1, 1, 0],
        }
    )
    treatments.to_csv(treatment_csv, index=False)

    groups = pd.DataFrame({"stay_id": [1, 2, 3], "groupHPD": [1, 2, 1]})
    groups.to_csv(group_csv, index=False)

    dataset_cfg = TreatmentDatasetConfig(
        name="mimic",
        treatment_csv=treatment_csv,
        group_csv=group_csv,
    )
    job_cfg = TreatmentUsageJobConfig(
        datasets=[dataset_cfg],
        treatment_columns=["is_rrt", "is_mv", "is_vaso"],
        output_dir=tmp_path / "plots",
        output_csv=tmp_path / "summary.csv",
    )
    summary = run_treatment_usage_job(job_cfg)
    assert job_cfg.output_csv and job_cfg.output_csv.exists()
    assert set(summary["treatment"].unique()) == {"is_rrt", "is_mv", "is_vaso"}

    outputs = plot_all_treatment_polar(
        summary,
        output_dir=job_cfg.output_dir,
        dataset_order=["mimic"],
        group_order=["1", "2"],
        treatment_order=["is_rrt", "is_mv", "is_vaso"],
    )
    assert "mimic" in outputs
    assert outputs["mimic"].exists()
