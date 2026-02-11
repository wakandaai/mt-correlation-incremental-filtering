#!/bin/bash
# Setup script for MT Correlation Analysis on PSC Bridges-2
# Run this once to set up your environment

echo "=========================================="
echo "Setting up MT Correlation Analysis Environment"
echo "=========================================="

# Check if we're on PSC Bridges-2
if [[ $(hostname) == *"bridges2"* ]]; then
    echo "✓ Running on PSC Bridges-2"
else
    echo "⚠ Warning: This script is designed for PSC Bridges-2"
    echo "Current host: $(hostname)"
fi

# Load required modules
echo ""
echo "Loading modules..."
module purge
module load anaconda3
module load cuda/11.7.1  # Adjust if needed

# Create conda environment
echo ""
echo "Creating conda environment: mt-correlation"
read -p "Environment name (default: mt-correlation): " ENV_NAME
ENV_NAME=${ENV_NAME:-mt-correlation}

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment $ENV_NAME already exists."
    read -p "Remove and recreate? (y/n): " RECREATE
    if [[ $RECREATE == "y" ]]; then
        conda env remove -n $ENV_NAME
        conda create -n $ENV_NAME python=3.9 -y
    fi
else
    conda create -n $ENV_NAME python=3.9 -y
fi

# Activate environment
echo ""
echo "Activating environment..."
source activate $ENV_NAME

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA 11.7..."
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# Install other dependencies
echo ""
echo "Installing project dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 2>/dev/null || echo "CUDA will be available on compute nodes"

python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import datasets; print('✓ Datasets installed')"
python -c "import comet; print('✓ COMET installed')"
python -c "import sacrebleu; print('✓ sacrebleu installed')"

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p logs
mkdir -p outputs
mkdir -p outputs/fleurs_data
mkdir -p outputs/nllb_translations
mkdir -p outputs/metrics_results
mkdir -p outputs/correlation_analysis

# Make scripts executable
echo ""
echo "Setting executable permissions..."
chmod +x pipelines/*.sh
chmod +x utils/preflight_check.py

# Test import
echo ""
echo "Testing language_config import..."
python -c "from config.language_config import get_test_languages; print(f'✓ Found {len(get_test_languages())} test languages')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Environment: $ENV_NAME"
echo ""
echo "To use this environment in future sessions:"
echo "  module load anaconda3"
echo "  source activate $ENV_NAME"
echo ""
echo "To submit jobs:"
echo "  1. Update --account in slurm_scripts/*.slurm files"
echo "  2. Run: sbatch slurm_scripts/submit_correlation_test.slurm"
echo ""
echo "Check your PSC account with: projects"