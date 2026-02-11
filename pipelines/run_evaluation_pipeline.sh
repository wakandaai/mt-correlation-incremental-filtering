#!/bin/bash
# Pipeline script to evaluate both Swahili and Kinyarwanda translations
# Usage: ./run_evaluation_pipeline.sh [dev|train]

set -e  # Exit on error

SPLIT=${1:-dev}  # Default to dev split
BATCH_SIZE=16

echo "=========================================="
echo "SSA-COMET-QE Evaluation Pipeline"
echo "Split: $SPLIT"
echo "=========================================="

# Create output directories
mkdir -p results
mkdir -p plots

# Function to run evaluation for a language
evaluate_language() {
    local lang=$1
    echo ""
    echo "=========================================="
    echo "Processing ${lang^^} translations..."
    echo "=========================================="
    
    # Paths - ADJUST THESE TO YOUR ACTUAL FILE LOCATIONS
    SOURCE_TSV="../../../shared/datasets/CommonVoice/cv-corpus-22.0-2025-06-20/${lang}/${SPLIT}.tsv"
    TARGET_TSV="../../../shared/datasets/CommonVoice/cv-corpus-22.0-2025-06-20/${lang}/eng_Latn/${SPLIT}.tsv"
    OUTPUT_CSV="results/${lang}_${SPLIT}_scores.csv"
    
    # Check if files exist
    if [ ! -f "$SOURCE_TSV" ]; then
        echo "ERROR: Source file not found: $SOURCE_TSV"
        echo "Please update the file paths in this script"
        return 1
    fi
    
    if [ ! -f "$TARGET_TSV" ]; then
        echo "ERROR: Target file not found: $TARGET_TSV"
        echo "Please update the file paths in this script"
        return 1
    fi
    
    # Run evaluation
    # echo "Running SSA-COMET-QE evaluation..."
    # python evaluate_translations.py \
    #     --source-tsv "$SOURCE_TSV" \
    #     --target-tsv "$TARGET_TSV" \
    #     --output-csv "$OUTPUT_CSV" \
    #     --language "$lang" \
    #     --batch-size "$BATCH_SIZE"
    
    # Generate visualizations
    echo ""
    echo "Generating visualizations..."
    python scripts/visualize_scores.py \
        --input-csv "$OUTPUT_CSV" \
        --language "$lang" \
        --output-dir "plots"
    
    echo ""
    echo "${lang^^} evaluation complete!"
    echo "Results saved to: $OUTPUT_CSV"
    echo "Plots saved to: plots/${lang}_${SPLIT}*.png"
}

# Evaluate both languages
evaluate_language "sw"
# evaluate_language "rw"

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Swahili scores: results/sw_${SPLIT}_scores.csv"
echo "  - Kinyarwanda scores: results/rw_${SPLIT}_scores.csv"
echo "  - Visualizations: plots/*.png"
echo ""
echo "Next steps:"
echo "  1. Review the plots in the plots/ directory"
echo "  2. Examine the CSV files to identify quality thresholds"
echo "  3. Based on score distributions, decide on filtering criteria"
echo "  4. Run on train.tsv if dev.tsv results look good:"
echo "     ./run_evaluation_pipeline.sh train"