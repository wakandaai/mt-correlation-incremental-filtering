#!/usr/bin/env python3
"""
Run NLLB translation on FLEURS data
Translates African languages to English using NLLB-200 model
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
from config.language_config import LANGUAGE_CONFIG, get_test_languages, get_all_languages

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NLLBTranslator:
    """NLLB translation wrapper with batching support"""
    
    def __init__(self, model_name="facebook/nllb-200-3.3B", device=None):
        """
        Initialize NLLB translator
        
        Args:
            model_name: HuggingFace model name
            device: cuda or cpu (auto-detected if None)
        """
        self.model_name = model_name
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading NLLB model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def translate_batch(self, texts, src_lang, tgt_lang="eng_Latn", max_length=512, batch_size=8):
        """
        Translate a batch of texts
        
        Args:
            texts: List of source texts
            src_lang: Source language code (NLLB format, e.g., 'swh_Latn')
            tgt_lang: Target language code (default: 'eng_Latn')
            max_length: Maximum generation length
            batch_size: Batch size for processing
        
        Returns:
            List of translations
        """
        translations = []
        
        # Set source language
        self.tokenizer.src_lang = src_lang
        
        # Get target language token ID (compatible with newer transformers)
        if hasattr(self.tokenizer, 'lang_code_to_id'):
            # Old method (transformers < 4.30)
            tgt_lang_id = self.tokenizer.lang_code_to_id[tgt_lang]
        else:
            # New method (transformers >= 4.30)
            tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
            batch = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Generate translations
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,  # Use the compatible variable
                    max_length=max_length,
                    num_beams=5,  # Beam search for better quality
                    early_stopping=True
                )
            
            # Decode
            batch_translations = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )
            
            translations.extend(batch_translations)
        
        return translations


def translate_language_data(input_file, output_file, nllb_code, translator, batch_size=8):
    """
    Translate data from a single file
    
    Args:
        input_file: Path to input CSV (with 'source' column)
        output_file: Path to output CSV
        nllb_code: NLLB language code for source language
        translator: NLLBTranslator instance
        batch_size: Batch size for translation
    
    Returns:
        Statistics dictionary
    """
    logger.info(f"Processing {input_file}")
    
    # Load data
    df = pd.read_csv(input_file)
    
    if 'source' not in df.columns:
        raise ValueError(f"Input file must have 'source' column. Found: {df.columns}")
    
    # Translate
    logger.info(f"Translating {len(df)} sentences from {nllb_code} to eng_Latn")
    translations = translator.translate_batch(
        df['source'].tolist(),
        src_lang=nllb_code,
        batch_size=batch_size
    )
    
    # Add translations to dataframe
    df['prediction'] = translations
    
    # Save
    df.to_csv(output_file, index=False)
    logger.info(f"Saved to {output_file}")
    
    return {
        'input_file': str(input_file),
        'output_file': str(output_file),
        'num_sentences': len(df),
        'status': 'success'
    }


def main():
    parser = argparse.ArgumentParser(description='Run NLLB translation on FLEURS data')
    parser.add_argument('--input-dir', default='./fleurs_data',
                       help='Directory with extracted FLEURS data')
    parser.add_argument('--output-dir', default='./nllb_translations',
                       help='Directory to save translations')
    parser.add_argument('--model', default='facebook/nllb-200-3.3B',
                       help='NLLB model name (default: 3.3B)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for translation')
    parser.add_argument('--test-only', action='store_true',
                       help='Translate only test languages')
    parser.add_argument('--languages', nargs='+',
                       help='Specific languages to translate')
    parser.add_argument('--splits', nargs='+', default=['dev', 'test'],
                       help='Splits to process')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which languages to process
    if args.languages:
        languages = {k: v for k, v in LANGUAGE_CONFIG.items() if k in args.languages}
    elif args.test_only:
        languages = get_test_languages()
    else:
        languages = get_all_languages()
    
    if not languages:
        logger.error("No languages selected")
        return
    
    logger.info(f"Translating {len(languages)} languages")
    logger.info(f"Splits: {args.splits}")
    
    # Initialize translator
    translator = NLLBTranslator(model_name=args.model)
    
    # Process each language and split
    all_stats = []
    for lang_name, lang_config in languages.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing {lang_name}")
        logger.info(f"{'='*70}")
        
        for split in args.splits:
            try:
                # Construct file paths
                input_file = Path(args.input_dir) / f"{lang_name.lower().replace(' ', '_')}_{split}.csv"
                output_file = output_dir / f"{lang_name.lower().replace(' ', '_')}_{split}_translated.csv"
                
                if not input_file.exists():
                    logger.warning(f"Input file not found: {input_file}")
                    continue
                
                # Translate
                stats = translate_language_data(
                    input_file,
                    output_file,
                    lang_config['nllb_code'],
                    translator,
                    args.batch_size
                )
                
                stats['language'] = lang_name
                stats['split'] = split
                stats['nllb_code'] = lang_config['nllb_code']
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
    summary_file = output_dir / 'translation_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    
    logger.info("\n" + "="*70)
    logger.info("TRANSLATION SUMMARY")
    logger.info("="*70)
    logger.info(f"\n{summary_df.to_string()}")
    logger.info(f"\nSummary saved to: {summary_file}")
    
    # Report results
    successes = summary_df[summary_df['status'] == 'success']
    failures = summary_df[summary_df['status'] == 'failed']
    
    logger.info(f"\nSuccessful: {len(successes)}/{len(all_stats)}")
    if len(failures) > 0:
        logger.warning(f"Failed: {len(failures)}")


if __name__ == "__main__":
    main()