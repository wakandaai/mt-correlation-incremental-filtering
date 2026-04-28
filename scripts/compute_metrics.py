#!/usr/bin/env python3
"""
Compute MT metrics: chrF++, BLEU, and COMET
"""

import pandas as pd
import torch
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import sacrebleu
from comet import download_model, load_from_checkpoint
from config.language_config import LANGUAGE_CONFIG, get_test_languages, get_all_languages

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_chrf_bleu(predictions, references):
    """Compute chrF++ and BLEU scores"""
    chrf_scores = []
    bleu_scores = []
    
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions), 
                          desc="Computing chrF++/BLEU"):
        chrf_score = sacrebleu.sentence_chrf(pred, [ref], word_order=2).score
        chrf_scores.append(chrf_score)
        
        bleu_score = sacrebleu.sentence_bleu(pred, [ref]).score
        bleu_scores.append(bleu_score)
    
    return chrf_scores, bleu_scores


def compute_ssa_qe(sources, predictions, model, batch_size=16):
    """Reference-free QE scoring — SSA-COMET-QE takes only src + mt"""
    data = [
        {"src": src, "mt": pred}
        for src, pred in zip(sources, predictions)
    ]
    model_output = model.predict(
        data,
        batch_size=batch_size,
        gpus=1 if torch.cuda.is_available() else 0
    )
    return model_output.scores


def compute_metrics_for_file(input_file, output_file, comet_model, batch_size=16):
    """Compute all metrics for a single file"""
    logger.info(f"Processing {input_file}")
    
    # Load data
    df = pd.read_csv(input_file)
    
    # Validate columns
    required_cols = ['source', 'reference', 'prediction']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {df.columns.tolist()}")
    
    # Compute chrF++ and BLEU
    logger.info("Computing chrF++ and BLEU scores")
    chrf_scores, bleu_scores = compute_chrf_bleu(
        df['prediction'].tolist(),
        df['reference'].tolist()
    )
    
    df['chrf_score'] = chrf_scores
    df['bleu_score'] = bleu_scores
    
    # Compute COMET
    ssa_qe_scores = compute_ssa_qe(
    df['source'].tolist(),
    df['prediction'].tolist(),
    comet_model,
    batch_size
    )
    df['ssa_qe_score'] = ssa_qe_scores
    
    # Save
    df.to_csv(output_file, index=False)
    logger.info(f"Saved to {output_file}")
    
    # Compute statistics
    stats = {
        'input_file': str(input_file),
        'output_file': str(output_file),
        'num_sentences': len(df),
        'mean_chrf': df['chrf_score'].mean(),
        'mean_bleu': df['bleu_score'].mean(),
        'mean_ssa_qe': df['ssa_qe_score'].mean(),
        'median_chrf': df['chrf_score'].median(),
        'median_bleu': df['bleu_score'].median(),
        'median_ssa_qe': df['ssa_qe_score'].median(),
        'status': 'success'
    }
    
    logger.info(f"  Mean chrF++: {stats['mean_chrf']:.2f}")
    logger.info(f"  Mean BLEU: {stats['mean_bleu']:.2f}")
    logger.info(f"  Mean COMET: {stats['mean_comet']:.4f}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Compute MT metrics')
    parser.add_argument('--input-dir', default='./nllb_translations')
    parser.add_argument('--output-dir', default='./metrics_results')
    parser.add_argument('--qe-model', default='McGill-NLP/ssa-comet-qe')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--languages', nargs='+')
    parser.add_argument('--splits', nargs='+', default=['dev', 'test'])
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.languages:
        languages = {k: v for k, v in LANGUAGE_CONFIG.items() if k in args.languages}
    elif args.test_only:
        languages = get_test_languages()
    else:
        languages = get_all_languages()
    
    if not languages:
        logger.error("No languages selected")
        return
    
    logger.info(f"Computing metrics for {len(languages)} languages")
    
    # Load COMET model
    logger.info(f"Loading QE model: {args.qe_model}")
    model_path = download_model(args.qe_model)
    qe_model = load_from_checkpoint(model_path)
    logger.info("QE model loaded")
    
    # Process each language and split
    all_stats = []
    for lang_name in languages.keys():
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing {lang_name}")
        logger.info(f"{'='*70}")
        
        for split in args.splits:
            try:
                input_file = Path(args.input_dir) / f"{lang_name.lower().replace(' ', '_')}_{split}_translated.csv"
                output_file = output_dir / f"{lang_name.lower().replace(' ', '_')}_{split}_metrics.csv"
                
                if not input_file.exists():
                    logger.warning(f"Input file not found: {input_file}")
                    continue
                
                stats = compute_metrics_for_file(
                    input_file,
                    output_file,
                    comet_model,
                    args.batch_size
                )
                
                stats['language'] = lang_name
                stats['split'] = split
                all_stats.append(stats)
                
            except Exception as e:
                logger.error(f"Error processing {lang_name} {split}: {e}")
                all_stats.append({
                    'language': lang_name,
                    'split': split,
                    'status': 'failed',
                    'error': str(e)
                })
    
    # Save summary
    summary_df = pd.DataFrame(all_stats)
    summary_file = output_dir / 'metrics_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    
    logger.info("\n" + "="*70)
    logger.info("METRICS COMPUTATION SUMMARY")
    logger.info("="*70)
    
    if len(summary_df[summary_df['status'] == 'success']) > 0:
        success_df = summary_df[summary_df['status'] == 'success']
        display_cols = ['language', 'split', 'num_sentences', 'mean_chrf', 'mean_bleu', 'mean_comet']
        logger.info(f"\n{success_df[display_cols].to_string()}")
    
    logger.info(f"\nSummary saved to: {summary_file}")
    
    successes = summary_df[summary_df['status'] == 'success']
    failures = summary_df[summary_df['status'] == 'failed']
    
    logger.info(f"\nSuccessful: {len(successes)}/{len(all_stats)}")
    if len(failures) > 0:
        logger.warning(f"Failed: {len(failures)}")


if __name__ == "__main__":
    main()