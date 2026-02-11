#!/bin/bash
#SBATCH --job-name=corr-test
#SBATCH --output=logs/correlation_test_%j.out
#SBATCH --error=logs/correlation_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100-32:1
#SBATCH --partition=GPU-shared
#SBATCH --time=8:00:00
#SBATCH --account=cis250145p

echo "=========================================="
echo "Job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "=========================================="

module purge
module load anaconda3
module load cuda/11.7.1

source activate mt-correlation

echo ""
echo "Python version:"
python --version

echo ""
echo "GPU Info:"
nvidia-smi

echo ""
echo "=========================================="
echo "Starting correlation analysis (TEST MODE)"
echo "=========================================="

cd $SLURM_SUBMIT_DIR
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

mkdir -p logs

bash pipelines/run_correlation_pipeline.sh test

echo ""
echo "=========================================="
echo "Job completed on $(date)"
echo "Total runtime: $SECONDS seconds ($((SECONDS/60)) minutes)"
echo "=========================================="

if [ -f outputs/correlation_analysis/correlations_dev.csv ]; then
    echo ""
    echo "SUCCESS! Results available in:"
    echo "  - outputs/correlation_analysis/correlations_dev.csv"
    echo "  - outputs/correlation_analysis/correlations_test.csv"
    echo "  - outputs/correlation_analysis/scatter_plots/"
else
    echo ""
    echo "WARNING: Expected output files not found. Check error log."
fi