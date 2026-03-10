import json
import pandas as pd
import os
from pathlib import Path

# CONFIGURATION
INPUT_DIR = "/ocean/projects/cis250145p/shared/translated"  # Directory with filtered_*.jsonl files
OUTPUT_DIR = "../outputs/incremental_filtered"
QE_SCORE_COLUMN = "meta.ssa_score"  # UPDATE THIS based on Step 1!

LANGUAGES = {
    'bem': 'Bemba',
    'hau': 'Hausa',
    'ibo': 'Igbo',
    'yor': 'Yoruba'
}

# Filtering levels (remove bottom X%)
FILTER_LEVELS = {
    'top90': 0.10,  # Remove bottom 10%, keep top 90%
    'top75': 0.25,  # Remove bottom 25%, keep top 75%
    'top50': 0.50,  # Remove bottom 50%, keep top 50%
}

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

def load_jsonl(filepath):
    """Load JSONL file into list of dicts"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, filepath):
    """Save list of dicts to JSONL file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def extract_qe_scores(data, qe_column):
    """Extract QE scores from data (handles nested fields)"""
    scores = []
    for record in data:
        score = get_nested_value(record, qe_column)
        if score is not None:
            try:
                scores.append(float(score))
            except (ValueError, TypeError):
                scores.append(None)
        else:
            scores.append(None)
    return scores

def apply_incremental_filtering(lang_code, lang_name):
    """Apply 10%, 25%, 50% filtering to one language"""
    
    input_file = f"{INPUT_DIR}/filtered_{lang_code}_Latn.jsonl"
    
    print(f"{'='*70}")
    print(f"Processing {lang_name.upper()}")
    print(f"{'='*70}")
    
    # Load data
    print(f"Loading {input_file}...")
    try:
        data = load_jsonl(input_file)
    except FileNotFoundError:
        print(f"✗ File not found: {input_file}")
        return []
    
    print(f"Loaded {len(data):,} records")
    
    # Extract QE scores (handles nested fields)
    print(f"Extracting QE scores from '{QE_SCORE_COLUMN}'...")
    qe_scores = extract_qe_scores(data, QE_SCORE_COLUMN)
    
    # Create DataFrame with records and their QE scores
    records_with_scores = []
    for i, record in enumerate(data):
        if qe_scores[i] is not None:
            records_with_scores.append({
                'record': record,
                'qe_score': qe_scores[i]
            })
    
    if not records_with_scores:
        print(f"✗ No valid QE scores found!")
        return []
    
    df = pd.DataFrame(records_with_scores)
    
    # Original stats
    orig_count = len(df)
    orig_mean = df['qe_score'].mean()
    orig_median = df['qe_score'].median()
    orig_min = df['qe_score'].min()
    orig_max = df['qe_score'].max()
    
    print(f"\nOriginal dataset:")
    print(f"  Count: {orig_count:,}")
    print(f"  Mean QE: {orig_mean:.3f}")
    print(f"  Median: {orig_median:.3f}")
    print(f"  Range: {orig_min:.3f} - {orig_max:.3f}")
    print()
    
    results = []
    
    # Apply each filtering level
    for level_name, remove_pct in FILTER_LEVELS.items():
        # Calculate threshold (bottom X percentile)
        threshold = df['qe_score'].quantile(remove_pct)
        
        # Filter: keep only scores >= threshold
        filtered_df = df[df['qe_score'] >= threshold]
        
        # Stats
        filt_count = len(filtered_df)
        filt_mean = filtered_df['qe_score'].mean()
        filt_median = filtered_df['qe_score'].median()
        filt_min = filtered_df['qe_score'].min()
        retention = (filt_count / orig_count) * 100
        removed = orig_count - filt_count
        qe_improvement = filt_mean - orig_mean
        
        # Extract original records (not the wrapped ones)
        filtered_records = filtered_df['record'].tolist()
        
        # Save filtered data
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = f"{OUTPUT_DIR}/{lang_name.lower()}_{level_name}.jsonl"
        
        save_jsonl(filtered_records, output_file)
        
        # Report
        print(f"✓ {level_name.upper()} (remove bottom {remove_pct*100:.0f}%)")
        print(f"  Threshold: QE >= {threshold:.3f}")
        print(f"  Retained: {filt_count:,} ({retention:.1f}%)")
        print(f"  Removed: {removed:,} ({100-retention:.1f}%)")
        print(f"  Mean QE: {orig_mean:.3f} → {filt_mean:.3f} (+{qe_improvement:.3f})")
        print(f"  Median: {orig_median:.3f} → {filt_median:.3f}")
        print(f"  New range: {filt_min:.3f} - {orig_max:.3f}")
        print(f"  Saved: {output_file}")
        print()
        
        results.append({
            'language': lang_name,
            'filter_level': level_name,
            'remove_pct': remove_pct * 100,
            'threshold': threshold,
            'original_count': orig_count,
            'filtered_count': filt_count,
            'removed_count': removed,
            'retention_pct': retention,
            'original_mean_qe': orig_mean,
            'filtered_mean_qe': filt_mean,
            'qe_improvement': qe_improvement,
            'output_file': output_file
        })
    
    return results

def main():
    print("="*70)
    print("INCREMENTAL FILTERING PIPELINE")
    print("="*70)
    print(f"QE Score Column: {QE_SCORE_COLUMN}")
    print(f"Filtering Levels: {list(FILTER_LEVELS.keys())}")
    print("="*70)
    print()
    
    all_results = []
    
    # Process each language
    for lang_code, lang_name in LANGUAGES.items():
        results = apply_incremental_filtering(lang_code, lang_name)
        all_results.extend(results)
    
    # Save summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_file = f"{OUTPUT_DIR}/filtering_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print("="*70)
        print("OVERALL SUMMARY")
        print("="*70)
        print(f"\nProcessed: {len(LANGUAGES)} languages × {len(FILTER_LEVELS)} levels = {len(all_results)} datasets")
        print(f"\nSummary saved to: {summary_file}")
        print(f"\nOutput files in: {OUTPUT_DIR}/")
        
        # Quick stats
        print("\nQuick Stats by Filter Level:")
        for level in FILTER_LEVELS.keys():
            level_data = summary_df[summary_df['filter_level'] == level]
            avg_retention = level_data['retention_pct'].mean()
            avg_improvement = level_data['qe_improvement'].mean()
            total_kept = level_data['filtered_count'].sum()
            print(f"  {level}: {avg_retention:.1f}% avg retention, "
                  f"+{avg_improvement:.3f} avg QE improvement, "
                  f"{total_kept:,} total translations")
    else:
        print("\n✗ No data processed successfully!")

if __name__ == '__main__':
    main()