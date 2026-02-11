#!/usr/bin/env python3
"""
Extract FLEURS data for correlation analysis
Loads devtest and test splits, extracts source and reference translations
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path
import argparse
import logging
from config.language_config import LANGUAGE_CONFIG, get_test_languages, get_all_languages

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_fleurs_data(language_name, fleurs_code, output_dir, splits=['dev', 'test']):
    """
    Extract FLEURS data for a specific language
    
    Args:
        language_name: Human-readable language name
        fleurs_code: FLEURS dataset language code (e.g., 'sw_ke')
        output_dir: Directory to save extracted data
        splits: List of splits to extract (default: ['dev', 'test'])
    
    Returns:
        Dictionary with statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {'language': language_name, 'fleurs_code': fleurs_code}
    
    try:
        logger.info(f"Loading FLEURS data for {language_name} ({fleurs_code})")
        
        # Load FLEURS dataset
        # FLEURS structure: each language has source text + English translation
        dataset = load_dataset("google/fleurs", fleurs_code, trust_remote_code=True)
        
        for split in splits:
            if split not in dataset:
                logger.warning(f"Split '{split}' not found for {language_name}. Available: {list(dataset.keys())}")
                continue
            
            data = dataset[split]
            
            # Extract relevant fields
            # FLEURS has: id, transcription (source language), raw_transcription, english_translation
            records = []
            for item in data:
                records.append({
                    'sentence_id': item['id'],
                    'source': item['transcription'],  # Source language text
                    'reference': item['english_translation']  # English reference
                })
            
            df = pd.DataFrame(records)
            
            # Save to CSV
            output_file = output_dir / f"{language_name.lower().replace(' ', '_')}_{split}.csv"
            df.to_csv(output_file, index=False)
            
            logger.info(f"  {split}: {len(df)} sentences saved to {output_file}")
            stats[f'{split}_count'] = len(df)
        
        stats['status'] = 'success'
        
    except Exception as e:
        logger.error(f"Error extracting {language_name}: {e}")
        stats['status'] = 'failed'
        stats['error'] = str(e)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Extract FLEURS data for correlation analysis')
    parser.add_argument('--output-dir', default='./fleurs_data', 
                       help='Directory to save extracted data')
    parser.add_argument('--splits', nargs='+', default=['dev', 'test'],
                       help='Splits to extract (default: dev test)')
    parser.add_argument('--test-only', action='store_true',
                       help='Extract only test languages (sw, rw, fr)')
    parser.add_argument('--languages', nargs='+', 
                       help='Specific languages to extract (e.g., Swahili Kinyarwanda)')
    
    args = parser.parse_args()
    
    # Determine which languages to extract
    if args.languages:
        # Extract specific languages
        languages = {k: v for k, v in LANGUAGE_CONFIG.items() 
                    if k in args.languages}
    elif args.test_only:
        # Extract test languages only
        languages = get_test_languages()
    else:
        # Extract all languages
        languages = get_all_languages()
    
    if not languages:
        logger.error("No languages selected. Check --languages argument.")
        return
    
    logger.info(f"Extracting data for {len(languages)} languages")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Extract data for each language
    all_stats = []
    for lang_name, lang_config in languages.items():
        stats = extract_fleurs_data(
            lang_name,
            lang_config['fleurs_code'],
            args.output_dir,
            args.splits
        )
        all_stats.append(stats)
    
    # Save extraction summary
    summary_df = pd.DataFrame(all_stats)
    summary_file = Path(args.output_dir) / 'extraction_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    
    logger.info("\n" + "="*70)
    logger.info("EXTRACTION SUMMARY")
    logger.info("="*70)
    logger.info(f"\n{summary_df.to_string()}")
    logger.info(f"\nSummary saved to: {summary_file}")
    
    # Report successes and failures
    successes = summary_df[summary_df['status'] == 'success']
    failures = summary_df[summary_df['status'] == 'failed']
    
    logger.info(f"\nSuccessful: {len(successes)}/{len(all_stats)}")
    if len(failures) > 0:
        logger.warning(f"Failed: {len(failures)}")
        logger.warning(f"Failed languages: {list(failures['language'])}")


if __name__ == "__main__":
    main()