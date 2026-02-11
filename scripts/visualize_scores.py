#!/usr/bin/env python3
"""
Visualization Script for SSA-COMET-QE Scores
Generates histograms and distribution plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_score_distribution(df, language, output_dir, split):
    """
    Generate histogram and distribution plots for QE scores
    
    Args:
        df: DataFrame with 'ssa_comet_qe_score' column
        language: Language code (sw or rw)
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scores = df['ssa_comet_qe_score']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(scores, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.4f}')
    axes[0].axvline(scores.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {scores.median():.4f}')
    axes[0].set_xlabel('SSA-COMET-QE Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Distribution of SSA-COMET-QE Scores ({language.upper()})', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    box_data = axes[1].boxplot([scores], vert=True, patch_artist=True, 
                                labels=[f'{language.upper()}'])
    box_data['boxes'][0].set_facecolor('lightblue')
    axes[1].set_ylabel('SSA-COMET-QE Score', fontsize=12)
    axes[1].set_title(f'Box Plot of Scores ({language.upper()})', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'{language}_{split}_score_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {output_file}")
    plt.close()
    
    # Additional: KDE plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(scores, fill=True, alpha=0.5, ax=ax)
    ax.axvline(scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.4f}')
    ax.set_xlabel('SSA-COMET-QE Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Kernel Density Estimate of Scores ({language.upper()})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    kde_file = output_dir / f'{language}_{split}_score_kde.png'
    plt.savefig(kde_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved KDE plot to {kde_file}")
    plt.close()


def print_quantile_analysis(df, language, split):
    """Print quantile analysis of scores"""
    scores = df['ssa_comet_qe_score']
    
    logger.info("\n" + "="*60)
    logger.info(f"QUANTILE ANALYSIS - {language.upper()}")
    logger.info("="*60)
    
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    for q in quantiles:
        value = scores.quantile(q)
        logger.info(f"{int(q*100)}th percentile: {value:.4f}")
    
    logger.info("="*60)
    
    # Print samples from different score ranges
    logger.info("\n" + "="*60)
    logger.info("SAMPLE TRANSLATIONS BY SCORE RANGE")
    logger.info("="*60)
    
    # Low scores (bottom 10%)
    low_threshold = scores.quantile(0.1)
    low_samples = df[df['ssa_comet_qe_score'] <= low_threshold].head(3)
    
    logger.info(f"\nLOW SCORES (≤ {low_threshold:.4f}):")
    for idx, row in low_samples.iterrows():
        logger.info(f"\nScore: {row['ssa_comet_qe_score']:.4f}")
        logger.info(f"Source: {row['source'][:100]}...")
        logger.info(f"Translation: {row['translation'][:100]}...")
    
    # High scores (top 10%)
    high_threshold = scores.quantile(0.9)
    high_samples = df[df['ssa_comet_qe_score'] >= high_threshold].head(3)
    
    logger.info(f"\nHIGH SCORES (≥ {high_threshold:.4f}):")
    for idx, row in high_samples.iterrows():
        logger.info(f"\nScore: {row['ssa_comet_qe_score']:.4f}")
        logger.info(f"Source: {row['source'][:100]}...")
        logger.info(f"Translation: {row['translation'][:100]}...")
    
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='Visualize SSA-COMET-QE scores')
    parser.add_argument('--input-csv', required=True, help='Path to CSV file with scores')
    parser.add_argument('--language', required=True, choices=['sw', 'rw'], help='Language code')
    parser.add_argument('--output-dir', default='./plots', help='Directory to save plots')
    parser.add_argument('--split', required=True, help='Split which has the plots generated')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading scores from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    
    # Validate columns
    required_cols = ['sentence_id', 'source', 'translation', 'ssa_comet_qe_score']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    logger.info(f"Loaded {len(df)} scored translations")
    
    # Generate plots
    plot_score_distribution(df, args.language, args.output_dir, args.split)
    
    # Print quantile analysis
    print_quantile_analysis(df, args.language, args.split)


if __name__ == "__main__":
    main()