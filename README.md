# SAKI-Phenotyping: Longitudinal Subphenotype Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-12%2F12%20passing-brightgreen.svg)](tests/)

## 📖 Associated Publication

**Title**: Longitudinal subphenotypes in Sepsis-Associated Acute Kidney Injury patients with distinct kidney injury trajectories and diuretic therapy responses

**Authors**: [Li Han, Haibo Zhu, Jun He, Guanghao Liu, Ruoqiong Wu, Hao Fu, Lu Ao, Shixiang Zheng, Zhongheng Zhang, Xiaopei Shen]  
**Journal**: [To be filled]  
**DOI**: [To be filled]

---

## 🎯 Overview

This repository contains a production-ready Python pipeline for identifying and characterizing longitudinal subphenotypes in sepsis-associated acute kidney injury (SAKI) patients. The codebase implements reproducible analyses from raw EHR data to publication-ready figures, with all methodological parameters explicitly documented for transparent reporting.

**Key Features**:
- 🔬 **Longitudinal phenotyping** via mixAK Bayesian clustering with automated model selection
- 🎲 **Robust missing data handling** using MICE (10 imputations, 20 iterations)
- ⚖️ **Causal inference** with propensity score matching (caliper=0.2×SD) and three-way diuretic response analysis
- 📊 **Multi-cohort validation** across MIMIC-IV, AUMCdb, and eICU databases
- 🤖 **Machine learning** with AutoGluon and SHAP-based interpretability
- 📈 **Comprehensive visualization** including trajectory heatmaps, survival curves, and treatment polar charts

This package avoids Jupyter-specific constructs, de-duplicates utilities, and provides clear CLI entry points suitable for HPC environments and automated workflows.

## Repository Layout

```
publication_package/
├── pyproject.toml          # Project metadata & dependencies
├── requirements.txt        # Quick-install dependency list
├── src/sa_aki_pipeline/    # Reusable Python package
│   ├── config.py
│   ├── data/
│   ├── evaluation/
│   ├── explainability/
│   ├── modeling/
│   ├── plots/
│   ├── survival/
│   └── utils/
├── scripts/                # CLI entry points built on the package
└── tests/                  # Lightweight unit tests (pytest)
```

### Key Modules
- `sa_aki_pipeline.config`: Dataclasses describing experiment paths, split strategies, and training/SHAP settings.
- `sa_aki_pipeline.preprocessing`: Dataset-aware time-window generators plus configuration helpers for merging MIMIC-IV/eICU/AUMC tables without copying notebook code. **MICE configuration** with explicit `n_imputations=10`, `iterations=20` for reproducibility; automatic missingness reporting.
- `sa_aki_pipeline.phenotyping`: **mixAK clustering interface** with automated model selection tracking (deviance, autocorrelation, Gelman-Rubin). Generates Methods-ready diagnostics for selecting K=2..8 clusters.
- `sa_aki_pipeline.causal`: Propensity score matching with standardized caliper (0.2×SD of logit propensity score), match ratio control, and SMD diagnostics for Methods section citation.
- `sa_aki_pipeline.survival`: Time interval computation (e.g., sepsis→SAKI delay) + treatment usage visualization (RRT/MV/vasopressor polar charts).
- `sa_aki_pipeline.fluid`: **Three-way diuretic response PSM** using R TriMatch (caliper/OneToN matching) + fluid balance calculation + dose standardization.
- `sa_aki_pipeline.sepsis`: **Sepsis→AKI timing analysis** with SOFA evolution tracking at onset timepoints; supports paired/unpaired comparisons with multiple testing correction.
- `sa_aki_pipeline.visualization`: Configurable plotting helpers (heatmaps, boxplots, polar charts) to avoid ad-hoc matplotlib code in notebooks.
- `sa_aki_pipeline.data.split`: Deterministic dataset splitting utilities that replicate the original notebook logic.
- `sa_aki_pipeline.modeling.autogluon_trainer`: Encapsulation of AutoGluon Tabular training/evaluation.
- `sa_aki_pipeline.evaluation.reporting`: Helpers for per-class accuracy summaries and confusion-matrix-ready tables.
- `sa_aki_pipeline.explainability.shap_runner`: Thin SHAP wrapper tailored to trained AutoGluon predictors.
- `sa_aki_pipeline.plots.confusion_matrix`: Publication-quality confusion matrix plotting helper.

### CLI Scripts
The `scripts/` folder hosts runnable entry points:
1. `generate_time_windows.py` – config-driven time-window merge (24h/6h windows, dataset-specific settings in YAML).
2. `plot_heatmap.py` – reproducible longitudinal feature heatmaps using YAML configs (mirrors the original notebook outputs).
3. `run_mixak_clustering.py` – mixAK longitudinal clustering with automated model selection (K=2..8, deviance + autocorrelation tracking).
4. `run_time_stats.py` – merges cohort-specific event files and emits standardized survival interval tables.
5. `plot_time_stats.py` – boxplots comparing intervals (e.g., sepsis-to-SAKI delay) across datasets/phenotypes.
6. `plot_treatment_usage.py` – computes RRT/MV/vasopressor usage rates by phenotype and renders polar charts.
7. `run_psm.py` – propensity score matching with caliper=0.2×SD(logit), automated SMD reporting for causal analyses.
8. `run_diuretic_psm.py` – **NEW**: three-way diuretic response PSM using R TriMatch or Python approximation.
9. `run_sepsis_aki_timing.py` – **NEW**: sepsis→AKI interval calculation with SOFA evolution analysis.
10. `train_model.py` – end-to-end split + train + evaluate workflow.
11. `compute_shap.py` – batch SHAP extraction for any saved AutoGluon predictor.

Use these scripts as templates; add additional commands (e.g., cohort-specific preprocessing) by importing from `sa_aki_pipeline`.

## Quickstart

### Quick Demo (No Data Required)
```bash
./scripts/quickstart_demo.sh
```
Validates environment and demonstrates all available commands.

### Full Reproduction (Requires Data)
```bash
./scripts/reproduce_paper.sh
```
Executes complete pipeline from preprocessing to final results.

### Setup
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run tests**:
   ```bash
   pytest -v
   ```

### Individual Analysis Steps

3. **Time-windowed features**:
   ```bash
   python scripts/generate_time_windows.py --config configs/time_window_job.yaml
   ```

4. **Phenotype discovery**:
   ```bash
   python scripts/run_mixak_clustering.py --config configs/mixak_job.template.yaml
   ```

5. **Survival & timing analysis**:
   ```bash
   python scripts/run_time_stats.py --config configs/time_stats_job.template.yaml
   python scripts/run_sepsis_aki_timing.py --event-times data/events.csv --phenotypes data/phenotypes.csv --output-csv results/intervals.csv
   ```

6. **Causal analysis (PSM)**:
   ```bash
   python scripts/run_psm.py --config configs/psm_job.template.yaml --input-csv data/cohort.csv
   python scripts/run_diuretic_psm.py --input-csv data/diuretic.csv --output-csv results/matched.csv --use-r
   ```

7. **Visualizations**:
   ```bash
   python scripts/plot_time_stats.py --config configs/time_stats_plot.template.yaml
   python scripts/plot_treatment_usage.py --config configs/treatment_usage_job.template.yaml
   python scripts/plot_heatmap.py --config configs/heatmap_job.template.yaml
   ```

8. **Machine learning**:
   ```bash
   python scripts/train_model.py --input-csv data/features.csv --experiment-dir ./exp --strategy train2_test1
   python scripts/compute_shap.py --experiment-dir ./exp --dataset ./exp/input/test.csv --output ./exp/shap.csv
   ```

## Notes for Publication
- No raw data is included; scripts expect CSV inputs already derived from MIMIC-IV/eICU/AUMC database exports.
- All plotting helpers rely on Matplotlib/Seaborn and avoid notebook magics.
- **MICE parameters** are explicitly configured: 10 imputations, 20 iterations per imputation, with automatic missingness reporting.
- **PSM caliper** defaults to 0.2×SD of propensity score logit (Austin 2011 standard); all SMD diagnostics are logged automatically.
- **Diuretic response PSM** uses three-way TriMatch with phenotype-specific calipers (0.05, OneToN M1=1.5/M2=4, 0.14).
- **Sepsis→AKI timing** analysis includes SOFA evolution with paired comparisons and multiple testing correction.
- The package is designed for future enhancements (e.g., alternative models, additional feature engineering) under `src/sa_aki_pipeline/` with minimal duplication.
- All 11 CLI scripts are executable and fully documented for reproducibility.

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2025saki,
  title={Longitudinal subphenotypes in Sepsis-Associated Acute Kidney Injury patients with distinct kidney injury trajectories and diuretic therapy responses},
  author={[Your Name] and [Co-authors]},
  journal={[Journal Name]},
  year={2025},
  doi={[DOI]},
  url={https://github.com/shen-lab-icu/SAKI-Longitudinal-Subphenotyping}
}
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MIMIC-IV, AUMCdb, and eICU databases for providing the clinical data
- mixAK R package for longitudinal clustering implementation
- AutoGluon team for the machine learning framework
- All contributors to the open-source packages used in this project

## 📧 Contact

For questions or issues:
- Open an issue on GitHub: [Issues](https://github.com/shen-lab-icu/SAKI-Longitudinal-Subphenotyping/issues)
- Contact: bioinfo_zhb@163.com

---

**Data Availability**: Raw patient data cannot be shared due to privacy restrictions. Researchers can obtain access to MIMIC-IV, AUMCdb, and eICU databases through their respective data access processes. All analysis code is provided in this repository.

## Method Parameters for Manuscript Citation

When writing your Methods section, you can cite the following parameters directly from this codebase:

### Missing Data Imputation
> "Variables with >60% missingness after forward-fill were excluded. For remaining variables, we applied Multiple Imputation by Chained Equations (MICE) using predictive mean matching with **10 imputations** and **20 iterations** per imputation (miceforest v5.7+). Missingness rates are documented in `outputs/missingness_report.csv`."

### Propensity Score Matching
> "Propensity scores were estimated via logistic regression. Nearest-neighbor matching was performed with a caliper of **0.2 standard deviations** of the logit of the propensity score (Austin 2011). Standardized mean differences (SMD) before and after matching are reported in `outputs/psm_diagnostics_report.txt`."

### Clustering Model Selection
> "We fitted candidate models with 2–8 clusters using mixAK::GLMM_MCMC (R package) with Bayesian MCMC sampling (**burn-in=50, keep=2000, thin=50**). The optimal number of clusters was selected based on **lowest deviance**, satisfactory **Gelman–Rubin statistics <1.1 for all parameters**, and **<10% of Markov chains showing autocorrelation >0.1**, balancing model fit and convergence stability. Model selection diagnostics are documented in `outputs/mixak_phenotyping/model_selection_summary.txt`."

Feel free to rename or reorganize modules further; the current layout prioritizes clarity for reviewers and journal reproducibility checklists.
