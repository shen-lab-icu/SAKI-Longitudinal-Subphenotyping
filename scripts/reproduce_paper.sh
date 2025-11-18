#!/bin/bash
# Full reproduction script for SAKI-Phenotyping manuscript
# 
# This script demonstrates the complete analysis pipeline from raw data to final results.
# For a quick demo without data dependencies, run: scripts/quickstart_demo.sh
#
# Prerequisites:
# - Virtual environment activated (.venv)
# - Data files in specified locations
# - R installed with mixAK, TriMatch, survival packages

set -e  # Exit on error
set -u  # Exit on undefined variable

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/publication_package"

echo "=========================================="
echo "SAKI-Phenotyping Full Reproduction Script"
echo "=========================================="
echo

# Configuration
PYTHON="../.venv/bin/python"
PYTEST="../.venv/bin/pytest"
DATA_DIR="../data"  # Adjust to your data location
RESULTS_DIR="./results"
FIGURES_DIR="./figures"

# Create output directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$FIGURES_DIR"

# ==========================================
# Phase 0: Environment Validation
# ==========================================
echo "Phase 0: Validating environment..."
echo "-----------------------------------"

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python virtual environment not found at ../.venv"
    echo "Please run:"
    echo "  cd $REPO_ROOT"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r publication_package/requirements.txt"
    exit 1
fi

echo "✓ Python environment: $($PYTHON --version)"

# Check R availability
if ! command -v Rscript &> /dev/null; then
    echo "⚠️  WARNING: R not found. mixAK clustering and diuretic PSM will be skipped."
    SKIP_R=1
else
    echo "✓ R: $(Rscript --version 2>&1 | head -n1)"
    SKIP_R=0
fi

# Run tests
echo
echo "Running unit tests..."
PYTHONPATH=src $PYTEST tests -v --tb=short
echo "✓ All tests passed"
echo

# ==========================================
# Phase 1: Data Preprocessing
# ==========================================
echo "Phase 1: Data Preprocessing"
echo "-----------------------------------"

# Step 1.1: Generate time-windowed features
echo "Step 1.1: Generating time-windowed features..."
if [ -f "$DATA_DIR/mimic_raw.csv" ]; then
    $PYTHON scripts/generate_time_windows.py \
        --config configs/time_window_job.template.yaml \
        --input-csv "$DATA_DIR/mimic_raw.csv" \
        --output-dir "$RESULTS_DIR/time_windows" \
        --dataset mimic
    echo "✓ Time windows generated"
else
    echo "⚠️  Skipped (data not found): $DATA_DIR/mimic_raw.csv"
fi
echo

# ==========================================
# Phase 2: Phenotype Discovery
# ==========================================
echo "Phase 2: Phenotype Discovery via mixAK"
echo "-----------------------------------"

# Step 2.1: Run mixAK clustering
echo "Step 2.1: Running mixAK longitudinal clustering..."
if [ $SKIP_R -eq 0 ] && [ -f "$RESULTS_DIR/time_windows/mimic_6h.csv" ]; then
    $PYTHON scripts/run_mixak_clustering.py \
        --config configs/mixak_job.template.yaml \
        --input-csv "$RESULTS_DIR/time_windows/mimic_6h.csv" \
        --output-dir "$RESULTS_DIR/mixak" \
        --dataset mimic
    echo "✓ Phenotypes identified"
else
    echo "⚠️  Skipped (R not available or data missing)"
fi
echo

# ==========================================
# Phase 3: Survival & Treatment Analysis
# ==========================================
echo "Phase 3: Survival & Treatment Analysis"
echo "-----------------------------------"

# Step 3.1: Compute survival intervals
echo "Step 3.1: Computing survival intervals (sepsis→SAKI delay)..."
if [ -f "$DATA_DIR/event_times.csv" ]; then
    $PYTHON scripts/run_time_stats.py \
        --config configs/time_stats_job.template.yaml \
        --input-csv "$DATA_DIR/event_times.csv" \
        --output-csv "$RESULTS_DIR/time_stats.csv"
    echo "✓ Survival intervals computed"
else
    echo "⚠️  Skipped (data not found): $DATA_DIR/event_times.csv"
fi

# Step 3.2: Plot interval distributions
echo "Step 3.2: Plotting interval distributions..."
if [ -f "$RESULTS_DIR/time_stats.csv" ]; then
    $PYTHON scripts/plot_time_stats.py \
        --config configs/time_stats_plot.template.yaml \
        --input-csv "$RESULTS_DIR/time_stats.csv" \
        --output-dir "$FIGURES_DIR"
    echo "✓ Boxplots generated"
fi

# Step 3.3: Treatment usage analysis
echo "Step 3.3: Analyzing treatment usage patterns..."
if [ -f "$DATA_DIR/treatment_data.csv" ]; then
    $PYTHON scripts/plot_treatment_usage.py \
        --config configs/treatment_usage_job.template.yaml \
        --input-csv "$DATA_DIR/treatment_data.csv" \
        --output-dir "$FIGURES_DIR"
    echo "✓ Treatment polar charts created"
else
    echo "⚠️  Skipped (data not found)"
fi
echo

# ==========================================
# Phase 4: Causal Analysis (PSM)
# ==========================================
echo "Phase 4: Causal Analysis via PSM"
echo "-----------------------------------"

# Step 4.1: Standard binary PSM
echo "Step 4.1: Running propensity score matching..."
if [ -f "$DATA_DIR/cohort_with_phenotypes.csv" ]; then
    $PYTHON scripts/run_psm.py \
        --config configs/psm_job.template.yaml \
        --input-csv "$DATA_DIR/cohort_with_phenotypes.csv" \
        --treatment-col groupHPD \
        --treatment-value 2 \
        --control-value 1 \
        --output-csv "$RESULTS_DIR/psm_matched.csv"
    echo "✓ PSM completed"
else
    echo "⚠️  Skipped (data not found)"
fi

# Step 4.2: Three-way diuretic response PSM (requires R)
echo "Step 4.2: Running three-way diuretic response PSM..."
if [ $SKIP_R -eq 0 ] && [ -f "$DATA_DIR/diuretic_response.csv" ]; then
    $PYTHON scripts/run_diuretic_psm.py \
        --input-csv "$DATA_DIR/diuretic_response.csv" \
        --output-csv "$RESULTS_DIR/diuretic_matched.csv"
    echo "✓ Diuretic PSM completed"
else
    echo "⚠️  Skipped (R not available or data missing)"
fi
echo

# ==========================================
# Phase 5: Sepsis-AKI Timing Analysis
# ==========================================
echo "Phase 5: Sepsis-AKI Timing Analysis"
echo "-----------------------------------"

echo "Step 5.1: Calculating sepsis→AKI intervals..."
if [ -f "$DATA_DIR/event_times.csv" ]; then
    $PYTHON scripts/run_sepsis_aki_timing.py \
        --event-times "$DATA_DIR/event_times.csv" \
        --phenotypes "$RESULTS_DIR/mixak/phenotype_assignments.csv" \
        --output-csv "$RESULTS_DIR/sepsis_aki_intervals.csv" \
        --output-plot "$FIGURES_DIR/sepsis_aki_intervals.pdf"
    echo "✓ Timing analysis completed"
else
    echo "⚠️  Skipped (data not found)"
fi
echo

# ==========================================
# Phase 6: Visualization
# ==========================================
echo "Phase 6: Generating Visualizations"
echo "-----------------------------------"

# Step 6.1: Longitudinal trajectory heatmaps
echo "Step 6.1: Creating longitudinal heatmaps..."
if [ -f "$RESULTS_DIR/time_windows/mimic_6h.csv" ]; then
    $PYTHON scripts/plot_heatmap.py \
        --config configs/heatmap_job.template.yaml \
        --input-csv "$RESULTS_DIR/time_windows/mimic_6h.csv" \
        --output-dir "$FIGURES_DIR/heatmaps"
    echo "✓ Heatmaps generated"
else
    echo "⚠️  Skipped (data not found)"
fi
echo

# ==========================================
# Phase 7: Machine Learning (Optional)
# ==========================================
echo "Phase 7: Machine Learning Classifier (Optional)"
echo "-----------------------------------"

echo "Step 7.1: Training AutoGluon classifier..."
if [ -f "$RESULTS_DIR/mixak/features_with_labels.csv" ]; then
    $PYTHON scripts/train_model.py \
        --input-csv "$RESULTS_DIR/mixak/features_with_labels.csv" \
        --experiment-dir "$RESULTS_DIR/classifier" \
        --strategy train2_test1 \
        --time-limit 600
    echo "✓ Model trained"
    
    # Step 7.2: SHAP explanations
    echo "Step 7.2: Computing SHAP feature importance..."
    $PYTHON scripts/compute_shap.py \
        --experiment-dir "$RESULTS_DIR/classifier" \
        --dataset "$RESULTS_DIR/classifier/input/test.csv" \
        --output "$RESULTS_DIR/classifier/shap_values.csv"
    echo "✓ SHAP values computed"
else
    echo "⚠️  Skipped (data not found)"
fi
echo

# ==========================================
# Phase 8: Summary Report
# ==========================================
echo "=========================================="
echo "Reproduction Complete!"
echo "=========================================="
echo
echo "Results saved to: $RESULTS_DIR"
echo "Figures saved to: $FIGURES_DIR"
echo
echo "Key outputs:"
echo "  - Phenotype assignments: $RESULTS_DIR/mixak/phenotype_assignments.csv"
echo "  - Survival intervals: $RESULTS_DIR/time_stats.csv"
echo "  - PSM matched cohort: $RESULTS_DIR/psm_matched.csv"
echo "  - Classifier models: $RESULTS_DIR/classifier/"
echo
echo "For parameter details, see:"
echo "  - MICE: src/sa_aki_pipeline/preprocessing/config.py"
echo "  - PSM: src/sa_aki_pipeline/causal/config.py"
echo "  - mixAK: src/sa_aki_pipeline/phenotyping/config.py"
echo
echo "To cite in manuscript Methods:"
echo "  'Time-windowed features were imputed using MICE with 10 imputations"
echo "   and 20 iterations. Propensity score matching used caliper=0.2×SD(logit)."
echo "   Longitudinal phenotypes identified via mixAK with burn-in=50, keep=2000.'"
echo
