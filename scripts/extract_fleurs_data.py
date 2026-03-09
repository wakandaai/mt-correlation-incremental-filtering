#!/usr/bin/env python3
"""
Extract FLEURS data from shared PSC directory
Data already processed by RA - just need to copy and format
"""

import argparse
import logging
from pathlib import Path
import sys
import pandas as pd
import shutil

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from config.language_config import get_test_languages, get_all_languages

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# RA's processed data location
FLEURS_BASE = Path("/ocean/projects/cis250145p/shared/datasets/FLEURS/splits")


def extract_fleurs_from_shared(language_name, fleurs_code, output_dir, splits=['dev', 'test']):
    """Extract FLEURS data from RA's processed files"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {'language': language_name, 'fleurs_code': fleurs_code}
    
    try:
        logger.info(f"Processing {language_name} ({fleurs_code})")
        
        for split in splits:
            # Path to RA's processed file
            # Format: {lang}-en_us.csv (parallel file with English)
            source_file = FLEURS_BASE / split / f"{fleurs_code}-en_us.csv"
            
            if not source_file.exists():
                raise FileNotFoundError(f"File not found: {source_file}")
            
            logger.info(f"  Reading {split} split from: {source_file}")
            
            # Read the parallel file
            df = pd.read_csv(source_file)
            
            # Extract relevant columns
            # Source language columns: {lang}-id, {lang}-transcript
            # English columns: en_us-id, en_us-transcript
            source_id_col = f"{fleurs_code}-id"
            source_transcript_col = f"{fleurs_code}-transcript"
            reference_id_col = "en_us-id"
            reference_transcript_col = "en_us-transcript"
            
            # Create clean dataframe
            clean_df = pd.DataFrame({
                'sentence_id': df[source_id_col],
                'source': df[source_transcript_col],
                'reference': df[reference_transcript_col]
            })
            
            # Save to output directory
            output_file = output_dir / f"{language_name.lower().replace(' ', '_')}_{split}.csv"
            clean_df.to_csv(output_file, index=False)
            
            logger.info(f"  ✓ {split}: {len(clean_df)} sentences saved to {output_file}")
            stats[f'{split}_count'] = len(clean_df)
        
        stats['status'] = 'success'
        
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        stats['status'] = 'failed'
        stats['error'] = str(e)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Extract FLEURS from shared PSC directory')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--splits', nargs='+', default=['dev', 'test'], help='Splits to extract')
    parser.add_argument('--test-only', action='store_true', help='Only process test languages')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    languages = get_test_languages() if args.test_only else get_all_languages()
    
    logger.info(f"Extracting FLEURS from: {FLEURS_BASE}")
    logger.info(f"Languages: {list(languages.keys())}")
    logger.info(f"Splits: {args.splits}\n")
    
    results = []
    for language, config in languages.items():
        result = extract_fleurs_from_shared(language, config['fleurs_code'], output_dir, args.splits)
        results.append(result)
    
    logger.info(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    summary_df = pd.DataFrame(results)
    logger.info(f"\n{summary_df}\n")
    
    summary_df.to_csv(output_dir / "extraction_summary.csv", index=False)
    
    successful = len(summary_df[summary_df['status'] == 'success'])
    logger.info(f"Successful: {successful}/{len(results)}")


if __name__ == "__main__":
    main()