#!/usr/bin/env python3
"""
Pre-flight check script
Verifies all dependencies and configuration before running pipeline
"""

import sys
import importlib
from pathlib import Path

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError:
        print(f"✗ {package_name or module_name} - NOT INSTALLED")
        return False

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print(f"✓ GPU available: {gpu_count}x {gpu_name}")
            return True
        else:
            print(f"⚠ GPU not available - will use CPU (slower)")
            return False
    except:
        return False

def main():
    print("="*70)
    print("Pre-Flight Check for FLEURS-NLLB Correlation Pipeline")
    print("="*70)
    
    print("\n1. Checking core dependencies...")
    all_good = True
    
    deps = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("datasets", "datasets (HuggingFace)"),
        ("transformers", "transformers"),
        ("sentencepiece", "sentencepiece"),
        ("torch", "PyTorch"),
        ("comet", "unbabel-comet"),
        ("sacrebleu", "sacrebleu"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
    ]
    
    for module, name in deps:
        if not check_import(module, name):
            all_good = False
    
    print("\n2. Checking GPU availability...")
    has_gpu = check_gpu()
    
    print("\n3. Checking configuration files...")
    config_files = [
        "config/language_config.py",
        "scripts/extract_fleurs_data.py",
        "scripts/run_nllb_translation.py",
        "scripts/compute_metrics.py",
        "scripts/analyze_correlations.py",
        "pipelines/run_correlation_pipeline.sh"
    ]
    
    for f in config_files:
        if Path(f).exists():
            print(f"✓ {f}")
        else:
            print(f"✗ {f} - NOT FOUND")
            all_good = False
    
    print("\n" + "="*70)
    if all_good:
        print("✓ All checks passed! Ready to run pipeline.")
        print("\nNext step:")
        print("  chmod +x run_correlation_pipeline.sh")
        print("  ./run_correlation_pipeline.sh test")
    else:
        print("✗ Some checks failed. Please install missing dependencies:")
        print("  pip install -r requirements_correlation.txt")
        print("\nFor GPU support, ensure you have CUDA-enabled PyTorch:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    print("="*70)
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())