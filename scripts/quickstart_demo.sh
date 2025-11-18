#!/bin/bash
# Quick start guide for the SA-AKI phenotyping pipeline
# This script demonstrates the end-to-end workflow

set -e  # Exit on error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/publication_package"

echo "=== SA-AKI Phenotyping Pipeline Demo ==="
echo

# Ensure virtual environment
if [ ! -d "../.venv" ]; then
    echo "Error: Virtual environment not found. Please run:"
    echo "  python3 -m venv ../.venv"
    echo "  source ../.venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

PYTHON="../.venv/bin/python"
PYTEST="../.venv/bin/pytest"

# Step 0: Run tests
echo "Step 0: Running unit tests..."
PYTHONPATH=src $PYTEST tests -v
echo "✅ All tests passed"
echo

# Step 1: Generate time windows (example - requires data)
echo "Step 1: Generate time-windowed features"
echo "  Command: $PYTHON scripts/generate_time_windows.py --config configs/time_window_job.template.yaml"
echo "  (Skipped in demo - requires real data files)"
echo

# Step 2: Identify phenotypes via mixAK clustering
echo "Step 2: Identify longitudinal phenotypes via mixAK"
echo "  Command: $PYTHON scripts/run_mixak_clustering.py --config configs/mixak_job.template.yaml"
echo "  (Skipped in demo - requires R environment + mixAK package)"
echo

# Step 3: Compute survival intervals
echo "Step 3: Compute survival intervals (sepsis→SAKI delay)"
echo "  Command: $PYTHON scripts/run_time_stats.py --config configs/time_stats_job.template.yaml"
echo "  (Skipped in demo - requires event time files)"
echo

# Step 4: Plot interval distributions
echo "Step 4: Plot boxplots for interval comparisons"
echo "  Command: $PYTHON scripts/plot_time_stats.py --config configs/time_stats_plot.template.yaml"
echo

# Step 5: Visualize treatment usage
echo "Step 5: Generate treatment usage polar charts"
echo "  Command: $PYTHON scripts/plot_treatment_usage.py --config configs/treatment_usage_job.template.yaml"
echo

# Step 6: Propensity score matching
echo "Step 6: Run propensity score matching for causal analysis"
echo "  Command: $PYTHON scripts/run_psm.py --config configs/psm_job.template.yaml --input-csv cohort.csv"
echo

# Step 7: Plot longitudinal heatmaps
echo "Step 7: Render longitudinal trajectory heatmaps"
echo "  Command: $PYTHON scripts/plot_heatmap.py --config configs/heatmap_job.template.yaml"
echo

# Step 8: Train classifier
echo "Step 8: Train AutoGluon phenotype classifier"
echo "  Command: $PYTHON scripts/train_model.py --input-csv data.csv --experiment-dir ./exp --strategy train2_test1"
echo

# Step 9: Compute SHAP explanations
echo "Step 9: Extract SHAP feature importance"
echo "  Command: $PYTHON scripts/compute_shap.py --experiment-dir ./exp --dataset ./exp/input/test.csv --output ./exp/shap.csv"
echo

echo "=== Pipeline Overview Complete ==="
echo
echo "📖 For detailed instructions, see:"
echo "   - README.md"
echo "   - REFACTOR_PROGRESS.md"
echo
echo "📊 Methods parameters documented in:"
echo "   - src/sa_aki_pipeline/preprocessing/config.py (MICE)"
echo "   - src/sa_aki_pipeline/causal/config.py (PSM)"
echo "   - src/sa_aki_pipeline/phenotyping/config.py (mixAK)"
echo
echo "🧪 Run tests anytime with:"
echo "   PYTHONPATH=src $PYTEST tests -v"
