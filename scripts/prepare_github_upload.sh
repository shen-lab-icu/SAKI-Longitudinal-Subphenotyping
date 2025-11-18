#!/bin/bash
# Prepare repository for GitHub upload
# This script helps ensure your repository is clean and ready for public release

set -e

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=========================================="
echo "GitHub Repository Preparation Checklist"
echo "=========================================="
echo

# Check 1: Verify no sensitive data
echo "✓ Check 1: Scanning for sensitive data..."
cd "$PACKAGE_DIR"

if find . -name "*.csv" -o -name "*.parquet" -o -name "*.pkl" | grep -v ".venv" | grep -q .; then
    echo "⚠️  WARNING: Found data files that should not be committed:"
    find . -name "*.csv" -o -name "*.parquet" -o -name "*.pkl" | grep -v ".venv" | head -10
    echo
    echo "Please review .gitignore and remove sensitive files."
else
    echo "✓ No obvious data files found"
fi
echo

# Check 2: Verify .gitignore exists
echo "✓ Check 2: Verifying .gitignore..."
if [ -f "$PACKAGE_DIR/.gitignore" ]; then
    echo "✓ .gitignore exists ($(wc -l < $PACKAGE_DIR/.gitignore) lines)"
else
    echo "❌ ERROR: .gitignore not found!"
    exit 1
fi
echo

# Check 3: Verify LICENSE
echo "✓ Check 3: Verifying LICENSE..."
if [ -f "$PACKAGE_DIR/LICENSE" ]; then
    echo "✓ LICENSE exists"
else
    echo "⚠️  WARNING: No LICENSE file found. Consider adding one."
fi
echo

# Check 4: Check README
echo "✓ Check 4: Verifying README..."
if [ -f "$PACKAGE_DIR/README.md" ]; then
    if grep -q "Longitudinal subphenotypes in Sepsis-Associated Acute Kidney Injury" "$PACKAGE_DIR/README.md"; then
        echo "✓ README.md contains paper title"
    else
        echo "⚠️  WARNING: README may need paper title update"
    fi
else
    echo "❌ ERROR: README.md not found!"
    exit 1
fi
echo

# Check 5: Run tests
echo "✓ Check 5: Running test suite..."
cd "$PACKAGE_DIR"
if [ -d "../.venv" ]; then
    PYTHON="../.venv/bin/python"
    PYTEST="../.venv/bin/pytest"
    
    if PYTHONPATH=src $PYTEST tests -v --tb=short -q; then
        echo "✓ All tests passed"
    else
        echo "❌ ERROR: Some tests failed. Fix before uploading."
        exit 1
    fi
else
    echo "⚠️  WARNING: Virtual environment not found, skipping tests"
fi
echo

# Check 6: Verify required files
echo "✓ Check 6: Verifying required files..."
REQUIRED_FILES=(
    "README.md"
    "requirements.txt"
    "LICENSE"
    ".gitignore"
    "CONTRIBUTING.md"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$PACKAGE_DIR/$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ❌ Missing: $file"
    fi
done
echo

# Check 7: Check for TODOs in code
echo "✓ Check 7: Scanning for TODO/FIXME markers..."
TODO_COUNT=$(grep -r "TODO\|FIXME\|XXX" --include="*.py" "$PACKAGE_DIR/src" 2>/dev/null | wc -l || echo 0)
if [ "$TODO_COUNT" -gt 0 ]; then
    echo "⚠️  Found $TODO_COUNT TODO/FIXME markers in code:"
    grep -r "TODO\|FIXME\|XXX" --include="*.py" "$PACKAGE_DIR/src" 2>/dev/null | head -5
    echo "  (Review these before release)"
else
    echo "✓ No TODO markers found"
fi
echo

# Summary
echo "=========================================="
echo "Pre-upload Summary"
echo "=========================================="
echo
echo "Repository structure:"
echo "  - Python package: src/sa_aki_pipeline/"
echo "  - CLI scripts: scripts/ (11 scripts)"
echo "  - Tests: tests/ (12 tests)"
echo "  - Configs: configs/ (7 YAML templates)"
echo "  - Documentation: README.md, CONTRIBUTING.md, etc."
echo
echo "Next steps:"
echo "1. Review all WARNING messages above"
echo "2. Update LICENSE with your name/institution"
echo "3. Update README.md with your author info and contact"
echo "4. Initialize git repository:"
echo "   cd $PACKAGE_DIR"
echo "   git init"
echo "   git add ."
echo "   git commit -m 'Initial commit: SAKI-Phenotyping v1.0'"
echo "5. Create GitHub repository and push:"
echo "   git remote add origin https://github.com/shen-lab-icu/SAKI-Longitudinal-Subphenotyping.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo
echo "✓ Repository preparation complete!"
