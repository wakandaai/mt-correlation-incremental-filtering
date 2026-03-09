# Incrementally Filtered Synthetic Translation Data

## Overview
High-quality synthetic English→African language translation data filtered at three retention levels (90%, 75%, 50%) using SSA-COMET quality estimation scores.

## Languages
- **Bemba** (bem_Latn) - 795K original → 715K/596K/397K filtered
- **Hausa** (hau_Latn) - 27K original → 24K/20K/13K filtered  
- **Yoruba** (yor_Latn) - 473K original → 426K/355K/236K filtered

**Total:** 1.29M original translations → 1.16M/971K/647K filtered

## Filtering Methodology

### QE Model
- **Model:** SSA-COMET (Sub-Saharan African COMET)
- **Metric:** `meta.ssa_score` (0-1 scale, higher = better quality)
- **Source:** Pre-filtered synthetic translations

### Filtering Levels

#### Top 90% (Minimal Filtering)
- **Retention:** 90% of original data
- **Quality Improvement:** +0.009 avg QE score
- **Use Case:** Maximum data retention, slight quality boost
- **Files:** `*_top90.jsonl`

#### Top 75% (Balanced)
- **Retention:** 75% of original data  
- **Quality Improvement:** +0.020 avg QE score
- **Use Case:** Recommended baseline for training
- **Files:** `*_top75.jsonl`

#### Top 50% (High Quality)
- **Retention:** 50% of original data
- **Quality Improvement:** +0.038 avg QE score
- **Use Case:** High-quality subset for fine-tuning or small models
- **Files:** `*_top50.jsonl`

## Quality Statistics

### Original vs Filtered (Mean QE Scores)

| Language | Original | Top90 | Top75 | Top50 |
|----------|----------|-------|-------|-------|
| Bemba    | 0.520    | 0.532 | 0.544 | 0.563 |
| Hausa    | 0.545    | 0.549 | 0.557 | 0.572 |
| Yoruba   | 0.550    | 0.562 | 0.574 | 0.592 |

### Key Insights
- **Hausa** was already well-filtered (min QE: 0.500)
- **Bemba** shows most improvement potential (widest QE range)
- All languages cap at ~0.70-0.75 max QE (SSA-COMET ceiling)

## File Format

Each JSONL file contains records with:
```json
{
  "id": "unique_identifier",
  "source": "English source text",
  "translation": "Target language translation",
  "meta": {
    "ssa_score": 0.XXX,
    ...
  },
  ...
}
```

## Usage Recommendations

### For Model Training
1. **Start with top75** - Best quality/quantity balance
2. **Evaluate on top90** - Test generalization to slightly lower quality
3. **Fine-tune on top50** - Optional refinement step

### For Comparative Studies
Train identical models on each filtering level to quantify quality/quantity trade-off for your specific task (ASR, MT, etc.).

### Expected Impact
Based on correlation validation (15 languages):
- Bemba: Limited QE validation available
- Hausa: Strong QE reliability (r=0.725 with chrF++)
- Yoruba: Good QE reliability (r=0.688 with chrF++)

Higher QE reliability → More effective filtering → Greater downstream improvement

## Statistics

See `stats/filtering_summary.csv` for:
- Exact thresholds per language/level
- Retention rates
- QE score distributions
- Filtered dataset sizes

## Citation

Sylvia Kipkemoi (2026). "Incrementally Filtered Synthetic Translation Data for African Languages"  
Based on QE correlation validation across 15 African languages.  
Filtering pipeline: https://github.com/mt-correlation-analysis

## Contact

For questions or issues: skipkemo@andrew.cmu.edu