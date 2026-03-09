#!/usr/bin/env python3
"""
Analyze correlations between SSA-COMET-QE, chrF++, and BLEU
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import argparse
import logging
from config.language_config import LANGUAGE_CONFIG, get_test_languages, get_all_languages

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def compute_correlations(df, language, split):
    """
    Compute Pearson and Spearman correlations
    
    Args:
        df: DataFrame with metric scores
        language: Language name
        split: Split name (dev/test)
    
    Returns:
        Dictionary with correlation results
    """
    results = {
        'language': language,
        'split': split,
        'num_sentences': len(df)
    }
    
    # QE vs chrF++
    pearson_qe_chrf, p_qe_chrf = stats.pearsonr(
        df['comet_score'], 
        df['chrf_score']
    )
    spearman_qe_chrf, sp_qe_chrf = stats.spearmanr(
        df['comet_score'], 
        df['chrf_score']
    )
    
    # QE vs BLEU
    pearson_qe_bleu, p_qe_bleu = stats.pearsonr(
        df['comet_score'], 
        df['bleu_score']
    )
    spearman_qe_bleu, sp_qe_bleu = stats.spearmanr(
        df['comet_score'], 
        df['bleu_score']
    )
    
    # chrF++ vs BLEU (for reference)
    pearson_chrf_bleu, p_chrf_bleu = stats.pearsonr(
        df['chrf_score'], 
        df['bleu_score']
    )
    spearman_chrf_bleu, sp_chrf_bleu = stats.spearmanr(
        df['chrf_score'], 
        df['bleu_score']
    )
    
    results.update({
        'pearson_qe_chrf': pearson_qe_chrf,
        'pearson_qe_chrf_pval': p_qe_chrf,
        'spearman_qe_chrf': spearman_qe_chrf,
        'spearman_qe_chrf_pval': sp_qe_chrf,
        
        'pearson_qe_bleu': pearson_qe_bleu,
        'pearson_qe_bleu_pval': p_qe_bleu,
        'spearman_qe_bleu': spearman_qe_bleu,
        'spearman_qe_bleu_pval': sp_qe_bleu,
        
        'pearson_chrf_bleu': pearson_chrf_bleu,
        'spearman_chrf_bleu': spearman_chrf_bleu
    })
    
    return results


def plot_scatter_correlations(df, language, split, output_dir):
    """
    Create scatter plots showing correlations
    
    Args:
        df: DataFrame with metric scores
        language: Language name
        split: Split name
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # QE vs chrF++
    axes[0].scatter(df['comet_score'], df['chrf_score'], alpha=0.5, s=20)
    r_pearson = stats.pearsonr(df['comet_score'], df['chrf_score'])[0]
    r_spearman = stats.spearmanr(df['comet_score'], df['chrf_score'])[0]
    
    # Add regression line
    z = np.polyfit(df['comet_score'], df['chrf_score'], 1)
    p = np.poly1d(z)
    axes[0].plot(df['comet_score'], p(df['comet_score']), 
                "r--", alpha=0.8, linewidth=2)
    
    axes[0].set_xlabel('SSA-COMET-QE Score', fontsize=12)
    axes[0].set_ylabel('chrF++ Score', fontsize=12)
    axes[0].set_title(f'QE vs chrF++\nPearson: {r_pearson:.3f}, Spearman: {r_spearman:.3f}', 
                     fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # QE vs BLEU
    axes[1].scatter(df['comet_score'], df['bleu_score'], alpha=0.5, s=20)
    r_pearson = stats.pearsonr(df['comet_score'], df['bleu_score'])[0]
    r_spearman = stats.spearmanr(df['comet_score'], df['bleu_score'])[0]
    
    # Add regression line
    z = np.polyfit(df['comet_score'], df['bleu_score'], 1)
    p = np.poly1d(z)
    axes[1].plot(df['comet_score'], p(df['comet_score']), 
                "r--", alpha=0.8, linewidth=2)
    
    axes[1].set_xlabel('SSA-COMET-QE Score', fontsize=12)
    axes[1].set_ylabel('BLEU Score', fontsize=12)
    axes[1].set_title(f'QE vs BLEU\nPearson: {r_pearson:.3f}, Spearman: {r_spearman:.3f}', 
                     fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # chrF++ vs BLEU (reference)
    axes[2].scatter(df['chrf_score'], df['bleu_score'], alpha=0.5, s=20)
    r_pearson = stats.pearsonr(df['chrf_score'], df['bleu_score'])[0]
    r_spearman = stats.spearmanr(df['chrf_score'], df['bleu_score'])[0]
    
    # Add regression line
    z = np.polyfit(df['chrf_score'], df['bleu_score'], 1)
    p = np.poly1d(z)
    axes[2].plot(df['chrf_score'], p(df['chrf_score']), 
                "r--", alpha=0.8, linewidth=2)
    
    axes[2].set_xlabel('chrF++ Score', fontsize=12)
    axes[2].set_ylabel('BLEU Score', fontsize=12)
    axes[2].set_title(f'chrF++ vs BLEU\nPearson: {r_pearson:.3f}, Spearman: {r_spearman:.3f}', 
                     fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'{language} - {split.upper()} Split', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / f'{language.lower().replace(" ", "_")}_{split}_scatter.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved scatter plot to {output_file}")
    plt.close()


def plot_correlation_heatmap(correlation_df, output_dir, split='combined'):
    """
    Create heatmap of correlations across languages
    
    Args:
        correlation_df: DataFrame with correlation results
        output_dir: Directory to save plot
        split: Split name for title
    """
    output_dir = Path(output_dir)
    
    # Prepare data for heatmap
    languages = correlation_df['language'].tolist()
    
    # Create matrix for each correlation type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Pearson QE-chrF++
    data = correlation_df.set_index('language')['pearson_qe_chrf'].values.reshape(-1, 1)
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                yticklabels=languages, xticklabels=['Pearson'], 
                ax=axes[0, 0], vmin=0, vmax=1, cbar_kws={'label': 'Correlation'})
    axes[0, 0].set_title('QE vs chrF++ (Pearson)', fontweight='bold')
    
    # Spearman QE-chrF++
    data = correlation_df.set_index('language')['spearman_qe_chrf'].values.reshape(-1, 1)
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                yticklabels=languages, xticklabels=['Spearman'], 
                ax=axes[0, 1], vmin=0, vmax=1, cbar_kws={'label': 'Correlation'})
    axes[0, 1].set_title('QE vs chrF++ (Spearman)', fontweight='bold')
    
    # Pearson QE-BLEU
    data = correlation_df.set_index('language')['pearson_qe_bleu'].values.reshape(-1, 1)
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                yticklabels=languages, xticklabels=['Pearson'], 
                ax=axes[1, 0], vmin=0, vmax=1, cbar_kws={'label': 'Correlation'})
    axes[1, 0].set_title('QE vs BLEU (Pearson)', fontweight='bold')
    
    # Spearman QE-BLEU
    data = correlation_df.set_index('language')['spearman_qe_bleu'].values.reshape(-1, 1)
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                yticklabels=languages, xticklabels=['Spearman'], 
                ax=axes[1, 1], vmin=0, vmax=1, cbar_kws={'label': 'Correlation'})
    axes[1, 1].set_title('QE vs BLEU (Spearman)', fontweight='bold')
    
    plt.suptitle(f'Metric Correlations Across Languages ({split.upper()})', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_dir / f'correlation_heatmap_{split}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved heatmap to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze metric correlations')
    parser.add_argument('--input-dir', default='./metrics_results',
                       help='Directory with computed metrics')
    parser.add_argument('--output-dir', default='./correlation_analysis',
                       help='Directory to save analysis results')
    parser.add_argument('--test-only', action='store_true',
                       help='Analyze only test languages')
    parser.add_argument('--languages', nargs='+',
                       help='Specific languages to analyze')
    parser.add_argument('--splits', nargs='+', default=['dev', 'test'],
                       help='Splits to analyze')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine languages
    if args.languages:
        languages = {k: v for k, v in LANGUAGE_CONFIG.items() if k in args.languages}
    elif args.test_only:
        languages = get_test_languages()
    else:
        languages = get_all_languages()
    
    if not languages:
        logger.error("No languages selected")
        return
    
    logger.info(f"Analyzing correlations for {len(languages)} languages")
    
    # Process each split
    for split in args.splits:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing {split.upper()} split")
        logger.info(f"{'='*70}")
        
        all_correlations = []
        
        for lang_name in languages.keys():
            try:
                # Load metrics
                input_file = Path(args.input_dir) / f"{lang_name.lower().replace(' ', '_')}_{split}_metrics.csv"
                
                if not input_file.exists():
                    logger.warning(f"File not found: {input_file}")
                    continue
                
                logger.info(f"Analyzing {lang_name}")
                df = pd.read_csv(input_file)
                
                # Compute correlations
                corr_results = compute_correlations(df, lang_name, split)
                all_correlations.append(corr_results)
                
                # Log results
                logger.info(f"  Pearson QE-chrF++: {corr_results['pearson_qe_chrf']:.3f}")
                logger.info(f"  Pearson QE-BLEU:   {corr_results['pearson_qe_bleu']:.3f}")
                logger.info(f"  Spearman QE-chrF++: {corr_results['spearman_qe_chrf']:.3f}")
                logger.info(f"  Spearman QE-BLEU:   {corr_results['spearman_qe_bleu']:.3f}")
                
                # Create scatter plots
                plot_scatter_correlations(df, lang_name, split, output_dir / 'scatter_plots')
                
            except Exception as e:
                logger.error(f"Error analyzing {lang_name} {split}: {e}")
        
        # Save correlation results
        if all_correlations:
            corr_df = pd.DataFrame(all_correlations)
            corr_file = output_dir / f'correlations_{split}.csv'
            corr_df.to_csv(corr_file, index=False)
            logger.info(f"\nCorrelation results saved to {corr_file}")
            
            # Create heatmap
            plot_correlation_heatmap(corr_df, output_dir, split)
            
            # Print summary statistics
            logger.info("\n" + "="*70)
            logger.info(f"SUMMARY STATISTICS - {split.upper()}")
            logger.info("="*70)
            logger.info(f"\nMean Pearson QE-chrF++: {corr_df['pearson_qe_chrf'].mean():.3f} (±{corr_df['pearson_qe_chrf'].std():.3f})")
            logger.info(f"Mean Pearson QE-BLEU:   {corr_df['pearson_qe_bleu'].mean():.3f} (±{corr_df['pearson_qe_bleu'].std():.3f})")
            logger.info(f"\nMean Spearman QE-chrF++: {corr_df['spearman_qe_chrf'].mean():.3f} (±{corr_df['spearman_qe_chrf'].std():.3f})")
            logger.info(f"Mean Spearman QE-BLEU:   {corr_df['spearman_qe_bleu'].mean():.3f} (±{corr_df['spearman_qe_bleu'].std():.3f})")
            
            logger.info("\n" + "="*70)


if __name__ == "__main__":
    main()