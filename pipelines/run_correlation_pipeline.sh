#!/bin/bash
# Master pipeline for FLEURS-NLLB correlation analysis
# Usage: ./run_correlation_pipeline.sh [test|full]

set -e  # Exit on error

MODE=${1:-test}  # Default to test mode (sw, rw, fr)

echo "=========================================="
echo "FLEURS-NLLB Correlation Analysis Pipeline"
echo "Mode: $MODE"
echo "=========================================="

# Configuration
BATCH_SIZE=8
NLLB_MODEL="facebook/nllb-200-3.3B"
COMET_MODEL="Unbabel/wmt22-comet-da"

# Determine which languages to process
if [ "$MODE" == "test" ]; then
    LANG_FLAG="--test-only"
    echo "Running in TEST mode (Swahili, Kinyarwanda, French only)"
elif [ "$MODE" == "full" ]; then
    LANG_FLAG=""
    echo "Running in FULL mode (all 25 languages)"
else
    echo "Invalid mode: $MODE"
    echo "Usage: ./run_correlation_pipeline.sh [test|full]"
    exit 1
fi

# Create main output directory
mkdir -p outputs

echo ""
echo "=========================================="
echo "STEP 1: Extract FLEURS Data"
echo "=========================================="
python scripts/extract_fleurs_data.py \
    --output-dir outputs/fleurs_data \
    --splits dev test \
    $LANG_FLAG

echo ""
echo "=========================================="
echo "STEP 2: Run NLLB Translation"
echo "=========================================="
echo "Model: $NLLB_MODEL"
echo "Batch size: $BATCH_SIZE"
echo "This may take 30min-3hours depending on GPU and number of languages"
echo ""

python scripts/run_nllb_translation.py \
    --input-dir outputs/fleurs_data \
    --output-dir outputs/nllb_translations \
    --model "$NLLB_MODEL" \
    --batch-size $BATCH_SIZE \
    --splits dev test \
    $LANG_FLAG

echo ""
echo "=========================================="
echo "STEP 3: Compute Metrics (chrF++, BLEU, QE)"
echo "=========================================="
echo "COMET Model: $COMET_MODEL"
echo ""

python scripts/compute_metrics.py \
    --input-dir outputs/nllb_translations \
    --output-dir outputs/metrics_results \
    --comet-model "$COMET_MODEL" \
    --batch-size $BATCH_SIZE \
    --splits dev test \
    $LANG_FLAG

echo ""
echo "=========================================="
echo "STEP 4: Analyze Correlations"
echo "=========================================="
python scripts/analyze_correlations.py \
    --input-dir outputs/metrics_results \
    --output-dir outputs/correlation_analysis \
    --splits dev test \
    $LANG_FLAG

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results directory structure:"
echo "  outputs/"
echo "    ├── fleurs_data/              # Extracted FLEURS data"
echo "    ├── nllb_translations/        # NLLB predictions"
echo "    ├── metrics_results/          # All metrics (chrF++, BLEU, QE)"
echo "    └── correlation_analysis/     # Correlation results & plots"
echo ""
echo "Key files to review:"
echo "  - outputs/correlation_analysis/correlations_dev.csv"
echo "  - outputs/correlation_analysis/correlations_test.csv"
echo "  - outputs/correlation_analysis/correlation_heatmap_*.png"
echo "  - outputs/correlation_analysis/scatter_plots/*.png"
echo ""
echo "Next steps:"
if [ "$MODE" == "test" ]; then
    echo "  1. Review results for test languages (Swahili, Kinyarwanda, French)"
    echo "  2. If results look good, run on all languages:"
    echo "     ./run_correlation_pipeline.sh full"
fi
echo "  - Examine scatter plots to understand correlations"
echo "  - Check correlation_*.csv for numeric results"
echo "  - Use findings to inform QE score thresholds"