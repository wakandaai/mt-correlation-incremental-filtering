# MT Correlation Analysis for African Languages

Automated pipeline for analyzing correlation between SSA-COMET-QE quality estimation scores and traditional MT metrics (chrF++, BLEU) across 25 African languages.

## 🎯 Project Overview

This project investigates whether SSA-COMET-QE (a reference-free quality estimation metric) correlates with traditional reference-based metrics:
- **chrF++**: Character-level F-score  
- **BLEU**: Word-level n-gram similarity

Using:
- **Dataset**: Google FLEURS (dev + test splits)
- **Translation**: Meta NLLB-200-3.3B
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

**Test Mode** (3 languages: Swahili, Kinyarwanda, French)
```bash
./pipelines/run_correlation_pipeline.sh test
```

**Full Mode** (25 languages)
```bash
./pipelines/run_correlation_pipeline.sh full
```

## 📁 Project Structure
```
├── config/               # Language configurations
├── scripts/              # Core analysis scripts
├── pipelines/            # Automation scripts
├── utils/                # Helper utilities
├── data/                 # Input data
└── outputs/              # Generated results (gitignored)
```

## 📊 Languages Supported

French, Portuguese, Arabic, Afrikaans, Swahili, Somali, Hausa, Amharic, Malagasy, Kinyarwanda, Xhosa, Zulu, Chichewa, Sesotho, Shona, Igbo, Yoruba, Tigrinya, Luganda, Lingala, Setswana, Wolof, Bemba, Fongbe

## 📖 Documentation

- [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) - Quick reference guide
- [data/README.md](data/README.md) - Data acquisition instructions

## ⏱️ Runtime Estimates

- **Test mode**: 1-2 hours (GPU) / 4-7 hours (CPU)
- **Full mode**: 8-16 hours (GPU) / 2-4 days (CPU)

## 🔬 Methodology

1. Extract FLEURS data (source + reference translations)
2. Translate using NLLB-200-3.3B
3. Compute metrics: chrF++, BLEU, SSA-COMET-QE
4. Analyze correlations (Pearson & Spearman)
5. Generate visualizations (scatter plots, heatmaps)

## 📈 Output Files

Results are saved in `outputs/`:
- `correlation_analysis/correlations_{split}.csv` - Correlation coefficients
- `correlation_analysis/scatter_plots/` - QE vs metric visualizations
- `correlation_analysis/correlation_heatmap_{split}.png` - Cross-language comparison

## 🤝 Contributing

Wakanda AI Research Team