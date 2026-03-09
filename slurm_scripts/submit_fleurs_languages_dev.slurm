#!/bin/bash
#SBATCH --job-name=all-lang-dev
#SBATCH --output=logs/all_languages_dev_%j.out
#SBATCH --error=logs/all_languages_dev_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --partition=GPU-shared
#SBATCH --time=06:00:00
#SBATCH --account=cis250145p  

echo "=========================================="
echo "SSA Comet Languages Correlation Analysis (DEV)"
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

# Load modules
module purge
module load anaconda3 cuda/11.7.1

# Activate environment
source activate $PROJECT/conda_envs/mt-correlation

# Navigate
cd $SLURM_SUBMIT_DIR

# Set paths
export HF_HOME=$PROJECT/.cache/huggingface
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

# Create directories
mkdir -p logs outputs/fleurs_data outputs/nllb_translations outputs/metrics_results outputs/correlation_analysis

echo ""
echo "=========================================="
echo "STEP 1: Extract FLEURS Data (DEV only)"
echo "=========================================="
python scripts/extract_fleurs_data.py \
    --output-dir outputs/fleurs_data \
    --splits dev \
    --test-only

echo ""
echo "Extraction completed. Files created:"
ls -lh outputs/fleurs_data/ | grep dev

echo ""
echo "=========================================="
echo "STEP 2: NLLB Translation (DEV only)"
echo "=========================================="
python scripts/run_nllb_translation.py \
    --input-dir outputs/fleurs_data \
    --output-dir outputs/nllb_translations \
    --model facebook/nllb-200-3.3B \
    --batch-size 8 \
    --splits dev \
    --test-only

echo ""
echo "Translation completed. Files created:"
ls -lh outputs/nllb_translations/ | grep dev

echo ""
echo "=========================================="
echo "STEP 3: Compute Metrics (DEV only)"
echo "=========================================="
python scripts/compute_metrics.py \
    --input-dir outputs/nllb_translations \
    --output-dir outputs/metrics_results \
    --comet-model Unbabel/wmt22-comet-da \
    --batch-size 8 \
    --splits dev \
    --test-only

echo ""
echo "Metrics completed. Files created:"
ls -lh outputs/metrics_results/ | grep dev

echo ""
echo "=========================================="
echo "STEP 4: Analyze Correlations (DEV only)"
echo "=========================================="
python scripts/analyze_correlations.py \
    --input-dir outputs/metrics_results \
    --output-dir outputs/correlation_analysis \
    --splits dev \
    --test-only

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "Completed: $(date)"
echo "Total runtime: $SECONDS seconds ($((SECONDS/60)) minutes)"
echo "=========================================="

# Show final results
if [ -f outputs/correlation_analysis/correlations_dev.csv ]; then
    echo ""
    echo "✓ SUCCESS! Correlation Results:"
    echo ""
    cat outputs/correlation_analysis/correlations_dev.csv
else
    echo ""
    echo "✗ ERROR: Correlation results not found"
    echo "Check error log: logs/all_languages_dev_${SLURM_JOB_ID}.err"
fi