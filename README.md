# MT Correlation Analysis - Test Version (Swahili & Kinyarwanda)

Automated pipeline for analyzing correlation between SSA-COMET-QE quality estimation scores and traditional MT metrics (chrF++, BLEU).

**Test Version:** This implementation focuses on **Swahili (sw)** and **Kinyarwanda (rw)** for initial validation.

## 🎯 Project Overview

This project investigates whether SSA-COMET-QE (a reference-free quality estimation metric) correlates with traditional reference-based metrics:
- **chrF++**: Character-level F-score  
- **BLEU**: Word-level n-gram similarity

Using:
- **Dataset**: Google FLEURS (dev + test splits)
- **Translation**: Meta NLLB-200-3.3B
- **Languages**: Swahili and Kinyarwanda
- **Analysis**: Pearson & Spearman correlations

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python utils/preflight_check.py
```

### Run Analysis

**Test Mode** (Swahili and Kinyarwanda)
```bash
./pipelines/run_correlation_pipeline.sh test
```

## 📁 Project Structure

```
├── language_config.py        # Language configurations (sw, rw)
├── requirements.txt           # All dependencies
├── README.md                  # This file
├── setup_psc.sh              # PSC environment setup
│
├── scripts/                  # Core analysis scripts
│   ├── extract_fleurs_data.py
│   ├── run_nllb_translation.py
│   ├── compute_metrics.py
│   ├── analyze_correlations.py
│   ├── evaluate_translations.py
│   └── visualize_scores.py
│
├── pipelines/                # Automation scripts
│   ├── run_correlation_pipeline.sh
│   └── run_evaluation_pipeline.sh
│
├── slurm_scripts/           # PSC batch job scripts
│   └── submit_correlation_test.slurm
│
├── utils/                    # Helper utilities
│   └── preflight_check.py
│
├── data/                     # Input data
└── outputs/                  # Generated results (gitignored)
```

## 📊 Languages Supported

- **Swahili** (sw_ke → swh_Latn)
- **Kinyarwanda** (rw_rw → kin_Latn)

## ⏱️ Runtime Estimates

- **Test mode (sw + rw)**: 45-90 minutes (GPU) / 2-4 hours (CPU)

## 🔬 Methodology

1. Extract FLEURS data (source + reference translations)
2. Translate using NLLB-200-3.3B (Swahili/Kinyarwanda → English)
3. Compute metrics: chrF++, BLEU, SSA-COMET-QE
4. Analyze correlations (Pearson & Spearman)
5. Generate visualizations (scatter plots, heatmaps)

## 📈 Output Files

Results are saved in `outputs/`:
- `correlation_analysis/correlations_dev.csv` - Correlation coefficients for dev split
- `correlation_analysis/correlations_test.csv` - Correlation coefficients for test split
- `correlation_analysis/scatter_plots/` - QE vs metric visualizations
- `correlation_analysis/correlation_heatmap_*.png` - Cross-language comparison

### CSV Structure

```csv
language,split,num_sentences,pearson_qe_chrf,spearman_qe_chrf,pearson_qe_bleu,spearman_qe_bleu
Swahili,dev,647,0.756,0.742,0.683,0.671
Kinyarwanda,dev,647,0.801,0.789,0.724,0.715
```

## 🖥️ Running on PSC Bridges-2

### Initial Setup
```bash
# 1. SSH to PSC
ssh USERNAME@bridges2.psc.edu
cd ~/mt-correlation-analysis

# 2. Find your account
projects

# 3. Update slurm script with your account
nano slurm_scripts/submit_correlation_test.slurm
# Change: #SBATCH --account=<your-account>
# To: #SBATCH --account=YOUR_ACTUAL_ACCOUNT

# 4. Run setup
bash setup_psc.sh
```

### Submit Job
```bash
# Submit test job (Swahili + Kinyarwanda)
sbatch slurm_scripts/submit_correlation_test.slurm

# Monitor
squeue -u $USER
tail -f logs/correlation_test_*.out
```

## 📊 Expected Results

For Swahili and Kinyarwanda, you should see:
- **Pearson correlations** between QE and chrF++/BLEU
- **Spearman correlations** (rank-based)
- **Scatter plots** showing relationship between metrics
- **Statistical summary** in console output

### Interpreting Correlations
- **0.9 - 1.0**: Very strong positive correlation
- **0.7 - 0.9**: Strong positive correlation (typical for QE vs chrF++)
- **0.5 - 0.7**: Moderate positive correlation
- **< 0.5**: Weak correlation (investigate further)

## 🔧 Troubleshooting

### Import Errors
```bash
# Test imports
python -c "from config.language_config import get_test_languages; print(get_test_languages())"
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

## 📝 Dependencies

- Python 3.9+
- PyTorch 2.0+ with CUDA support
- Transformers, Datasets (HuggingFace)
- COMET, sacrebleu
- Pandas, NumPy, Matplotlib, Seaborn

See `requirements.txt` for complete list.

## 🎓 Next Steps

After successful test run:
1. Review correlation results for Swahili and Kinyarwanda
2. Examine scatter plots to understand metric relationships
3. Use findings to inform quality thresholds for filtering
4. Expand to additional languages if needed

## 🤝 Contributing

Wakanda AI Research Team


## 📞 Support

For questions about the correlation analysis methodology, refer to the SSA-COMET paper: https://arxiv.org/pdf/2506.04557