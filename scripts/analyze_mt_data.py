import json
import os
import pandas as pd
from pathlib import Path

DATA_DIR = "/ocean/projects/cis250145p/shared/translated"  # Directory with synthetic data 
OUTPUT_DIR = "../outputs/mt_analysis"
LANGUAGES = {
    'bem': 'Bemba',
    'hau': 'Hausa',
    'ibo': 'Igbo',
    'yor': 'Yoruba'
}

def load_jsonl(filepath, max_records=None):
    """Load JSONL file into list of dicts"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_records and i >= max_records:
                break
            data.append(json.loads(line))
    return data

def count_lines_fast(filepath):
    """Fast line count without loading full file"""
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)

def get_nested_value(record, path):
    """Get value from nested dict using dot notation (e.g., 'meta.ssa_score')"""
    keys = path.split('.')
    value = record
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value

def find_qe_column(sample_record):
    """Identify the QE score column from a sample record (handles nested fields)"""
    
    # Priority order for QE columns (including nested paths)
    qe_candidates = [
        'meta.ssa_score',           # ← Your nested field
        'meta.ssa_comet_qe_score',
        'meta.comet_score',
        'meta.qe_score',
        'ssa_comet_qe_score',
        'comet_qe_score',
        'qe_score',
        'comet_score',
        'quality_score',
        'score'
    ]
    
    # Check candidates (including nested paths)
    for candidate in qe_candidates:
        value = get_nested_value(sample_record, candidate)
        if value is not None:
            return candidate
    
    # Fallback: search for nested 'score' fields
    if 'meta' in sample_record and isinstance(sample_record['meta'], dict):
        for key in sample_record['meta'].keys():
            if 'score' in key.lower() and 'chrf' not in key.lower() and 'bleu' not in key.lower():
                return f'meta.{key}'
    
    # Fallback: top-level score fields
    for key in sample_record.keys():
        key_lower = key.lower()
        if 'score' in key_lower and 'chrf' not in key_lower and 'bleu' not in key_lower:
            return key
    
    return None

def extract_qe_scores(data, qe_column):
    """Extract QE scores from data (handles nested fields)"""
    scores = []
    for record in data:
        score = get_nested_value(record, qe_column)
        if score is not None:
            scores.append(score)
        else:
            scores.append(None)
    return scores

def analyze_filtered_language(lang_code, lang_name):
    """Analyze one filtered language dataset"""
    
    filepath = f"{DATA_DIR}/filtered_{lang_code}_Latn.jsonl"
    
    print(f"{'='*70}")
    print(f"{lang_name.upper()} ANALYSIS")
    print(f"{'='*70}")
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        return None
    
    # Fast count
    print("Counting records...")
    total_count = count_lines_fast(filepath)
    print(f"Total records: {total_count:,}")
    
    # Load sample to identify structure
    print("Loading sample...")
    sample = load_jsonl(filepath, max_records=1000)
    
    if not sample:
        print("✗ No data loaded!")
        return None
    
    # Show first record structure
    print(f"\nFirst record structure:")
    print(f"  Top-level keys: {list(sample[0].keys())}")
    if 'meta' in sample[0] and isinstance(sample[0]['meta'], dict):
        print(f"  Meta keys: {list(sample[0]['meta'].keys())}")
    
    # Identify QE column
    qe_col = find_qe_column(sample[0])
    
    if not qe_col:
        print(f"\n✗ No QE score column found!")
        print(f"  Available columns: {list(sample[0].keys())}")
        if 'meta' in sample[0]:
            print(f"  Meta columns: {list(sample[0].get('meta', {}).keys())}")
        return None
    
    print(f"\n✓ QE column identified: {qe_col}")
    
    # Extract QE scores (handles nested fields)
    print("Extracting QE scores...")
    qe_scores = extract_qe_scores(sample, qe_col)
    
    # Create DataFrame with extracted scores
    sample_data = []
    for i, record in enumerate(sample):
        sample_data.append({
            'qe_score': qe_scores[i],
            'source': record.get('source', record.get('src', 'N/A')),
            'translation': record.get('translation', record.get('tgt', record.get('target', 'N/A')))
        })
    
    sample_df = pd.DataFrame(sample_data)
    
    # Check if QE column has valid numeric data
    try:
        sample_df['qe_score'] = pd.to_numeric(sample_df['qe_score'], errors='coerce')
        
        # Remove NaN values
        valid_sample = sample_df[sample_df['qe_score'].notna()]
        
        if len(valid_sample) == 0:
            print(f"\n✗ No valid QE scores found in sample!")
            return None
        
        # Sample statistics
        sample_mean = valid_sample['qe_score'].mean()
        sample_median = valid_sample['qe_score'].median()
        sample_min = valid_sample['qe_score'].min()
        sample_max = valid_sample['qe_score'].max()
        sample_std = valid_sample['qe_score'].std()
        
        print(f"\nQE STATISTICS (from sample of {len(valid_sample):,}):")
        print(f"  Mean:   {sample_mean:.3f}")
        print(f"  Median: {sample_median:.3f}")
        print(f"  Std:    {sample_std:.3f}")
        print(f"  Min:    {sample_min:.3f}")
        print(f"  Max:    {sample_max:.3f}")
        
        # Percentiles
        p25 = valid_sample['qe_score'].quantile(0.25)
        p75 = valid_sample['qe_score'].quantile(0.75)
        print(f"  25th percentile: {p25:.3f}")
        print(f"  75th percentile: {p75:.3f}")
        
        # Show sample translations
        print(f"\nSAMPLE TRANSLATIONS (5 random):")
        random_sample = valid_sample.sample(min(5, len(valid_sample)))
        for idx, row in random_sample.iterrows():
            qe = row['qe_score']
            src = str(row['source'])[:60]
            tgt = str(row['translation'])[:60]
            print(f"  QE={qe:.3f}: {src}... → {tgt}...")
        
        print()
        
        return {
            'language': lang_name,
            'lang_code': lang_code,
            'total_count': total_count,
            'qe_column': qe_col,
            'mean_qe': sample_mean,
            'median_qe': sample_median,
            'std_qe': sample_std,
            'min_qe': sample_min,
            'max_qe': sample_max,
            'p25_qe': p25,
            'p75_qe': p75,
            'sample_size': len(valid_sample),
            'filepath': filepath
        }
        
    except Exception as e:
        print(f"\n✗ Error analyzing QE scores: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*70)
    print("FILTERED DATA ANALYSIS")
    print("="*70)
    print()
    
    results = []
    
    # Analyze each language
    for lang_code, lang_name in LANGUAGES.items():
        result = analyze_filtered_language(lang_code, lang_name)
        if result:
            results.append(result)
    
    # Overall summary
    if results:
        print("="*70)
        print("OVERALL SUMMARY")
        print("="*70)
        
        summary_df = pd.DataFrame(results)
        
        # Display summary table
        display_cols = ['language', 'total_count', 'mean_qe', 'median_qe', 'min_qe', 'max_qe']
        print(f"\n{summary_df[display_cols].to_string(index=False)}")
        
        print(f"\n")
        print(f"Total datasets: {len(results)}")
        print(f"Total translations: {summary_df['total_count'].sum():,}")
        print(f"Average mean QE: {summary_df['mean_qe'].mean():.3f}")
        print(f"QE range across languages: {summary_df['min_qe'].min():.3f} - {summary_df['max_qe'].max():.3f}")
        
        # Save summary
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = f"{OUTPUT_DIR}/filtered_data_summary.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"\nSummary saved to: {output_file}")
        
        # Check if QE columns are consistent
        qe_columns = summary_df['qe_column'].unique()
        if len(qe_columns) == 1:
            print(f"\n✓ All languages use same QE column: {qe_columns[0]}")
        else:
            print(f"\n⚠️  Different QE columns found: {list(qe_columns)}")
    else:
        print("\n✗ No data successfully analyzed!")
        print("Check file paths and data structure.")

if __name__ == '__main__':
    main()