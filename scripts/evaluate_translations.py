#!/usr/bin/env python3
"""
SSA-COMET-QE Evaluation Script
Evaluates synthetic MT translations using quality estimation (no reference needed)
"""

import pandas as pd
import torch
from comet import download_model, load_from_checkpoint
from pathlib import Path
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_tsv(filepath):
    """Load TSV file and return dataframe"""
    logger.info(f"Loading {filepath}")
    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        engine="python",      # more tolerant than the C engine
        on_bad_lines="skip"   # skip malformed lines instead of crashing
    )
    return df


def align_source_target(source_df, target_df):
    """
    Align source and target dataframes by sentence_id
    
    Args:
        source_df: DataFrame with columns [sentence_id, sentence]
        target_df: DataFrame with columns [sentence_id, sentence]
    
    Returns:
        DataFrame with columns [sentence_id, source, translation]
    """
    logger.info("Aligning source and target by sentence_id")
    
    # Rename columns for clarity
    source_df = source_df.rename(columns={'sentence': 'source'})
    target_df = target_df.rename(columns={'sentence': 'translation'})
    
    # Merge on sentence_id
    aligned = pd.merge(
        source_df[['sentence_id', 'source']], 
        target_df[['sentence_id', 'translation']], 
        on='sentence_id',
        how='inner'
    )
    
    logger.info(f"Aligned {len(aligned)} sentence pairs")
    return aligned


def score_translations_qe(aligned_df, model, batch_size=16):
    """
    Score translations using SSA-COMET-QE model
    
    Args:
        aligned_df: DataFrame with [sentence_id, source, translation]
        model: Loaded COMET model
        batch_size: Batch size for inference
    
    Returns:
        DataFrame with added 'ssa_comet_qe_score' column
    """
    logger.info(f"Scoring {len(aligned_df)} translations with batch_size={batch_size}")
    
    # Prepare data in COMET format (QE only needs source and translation)
    data = [
        {"src": row['source'], "mt": row['translation']}
        for _, row in aligned_df.iterrows()
    ]
    
    # Run model prediction with batching
    model_output = model.predict(data, batch_size=batch_size, gpus=1 if torch.cuda.is_available() else 0)
    
    # Add scores to dataframe
    aligned_df['ssa_comet_qe_score'] = model_output.scores
    
    logger.info(f"Scoring complete. Mean score: {aligned_df['ssa_comet_qe_score'].mean():.4f}")
    
    return aligned_df


def main():
    parser = argparse.ArgumentParser(description='Evaluate MT translations using SSA-COMET-QE')
    parser.add_argument('--source-tsv', required=True, help='Path to source TSV file (CV data)')
    parser.add_argument('--target-tsv', required=True, help='Path to target TSV file (MT output)')
    parser.add_argument('--output-csv', required=True, help='Path to output CSV file')
    parser.add_argument('--language', required=True, choices=['sw', 'rw'], help='Language code (sw or rw)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--model-name', default='McGill-NLP/ssa-comet-qe', 
                       help='COMET model to use (default: XCOMET-XL which supports QE)')
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model_path = download_model(args.model_name)
    model = load_from_checkpoint(model_path)
    
    # Load data
    source_df = load_tsv(args.source_tsv)
    target_df = load_tsv(args.target_tsv)
    
    # Align source and target
    aligned_df = align_source_target(source_df, target_df)
    
    # Score translations
    results_df = score_translations_qe(aligned_df, model, batch_size=args.batch_size)
    
    # Save results
    logger.info(f"Saving results to {args.output_csv}")
    results_df.to_csv(args.output_csv, index=False)
    
    # Print summary statistics
    logger.info("\n" + "="*50)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*50)
    logger.info(f"Language: {args.language}")
    logger.info(f"Total sentences: {len(results_df)}")
    logger.info(f"Mean score: {results_df['ssa_comet_qe_score'].mean():.4f}")
    logger.info(f"Median score: {results_df['ssa_comet_qe_score'].median():.4f}")
    logger.info(f"Std dev: {results_df['ssa_comet_qe_score'].std():.4f}")
    logger.info(f"Min score: {results_df['ssa_comet_qe_score'].min():.4f}")
    logger.info(f"Max score: {results_df['ssa_comet_qe_score'].max():.4f}")
    logger.info("="*50)


if __name__ == "__main__":
    main()