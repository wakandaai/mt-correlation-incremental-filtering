# Incrementally Filtered Synthetic Translation Data

## Overview

High-quality synthetic English→African language translation data filtered at three retention levels (90%, 75%, 50%) using SSA-COMET quality estimation scores.

## Languages

- **Bemba** (bem_Latn) - 795K original → 715K/596K/397K filtered
- **Hausa** (hau_Latn) - 27K original → 24K/20K/13K filtered  
- **Igbo** (ibo_Latn) - 116K original → 105K/87K/58K filtered
- **Yoruba** (yor_Latn) - 473K original → 426K/355K/236K filtered

**Total:** 1.41M original translations → 1.27M/1.06M/705K filtered

---

## Filtering Methodology

### QE Model

- **Model:** SSA-COMET (Sub-Saharan African COMET)
- **Metric:** `meta.ssa_score` (0-1 scale, higher = better quality)
- **Source:** Pre-filtered synthetic translations

### Filtering Levels

#### Top 90% (Minimal Filtering)

- **Retention:** 90% of original data
- **Quality Improvement:** +0.008 avg QE score
- **Use Case:** Maximum data retention, slight quality boost
- **Files:** `*_top90.jsonl`

#### Top 75% (Balanced)

- **Retention:** 75% of original data  
- **Quality Improvement:** +0.018 avg QE score
- **Use Case:** Recommended baseline for training
- **Files:** `*_top75.jsonl`

#### Top 50% (High Quality)

- **Retention:** 50% of original data
- **Quality Improvement:** +0.035 avg QE score
- **Use Case:** High-quality subset for fine-tuning or small models
- **Files:** `*_top50.jsonl`

---

## Quality Statistics

### Original vs Filtered (Mean QE Scores)

| Language | Original | Top90 | Top75 | Top50 |
|----------|----------|-------|-------|-------|
| Bemba    | 0.520    | 0.532 | 0.544 | 0.563 |
| Hausa    | 0.545    | 0.549 | 0.557 | 0.572 |
| Igbo     | 0.545    | 0.549 | 0.557 | 0.573 |
| Yoruba   | 0.550    | 0.562 | 0.574 | 0.592 |

### QE Score Ranges

| Language | Min QE | Max QE | Range | Quality Profile |
|----------|--------|--------|-------|-----------------|
| Hausa    | 0.500  | 0.702  | 0.202 | Most consistent, highest baseline |
| Igbo     | 0.456  | 0.737  | 0.281 | Consistent, high baseline |
| Yoruba   | 0.407  | 0.752  | 0.345 | Moderate variation |
| Bemba    | 0.237  | 0.752  | 0.515 | Widest variation, most filtering potential |

### Key Insights

- **Hausa and Igbo** were already well-filtered (min QE: 0.500 and 0.456)
- **Bemba** shows most improvement potential (widest QE range: 0.515)
- **Yoruba** demonstrates balanced filtering opportunities
- All languages cap at ~0.70-0.75 max QE (SSA-COMET ceiling)

---

## File Format

Each JSONL file contains records with:
```json
{
  "doc_id": "unique_identifier",
  "source_text": "English source text",
  "translated_text": "Target language translation",
  "source_tokens": 123,
  "target_tokens": 145,
  "num_sentences": 5,
  "chrf_score": 0.XXX,
  "bleu_score": 0.XXX,
  "min_chrf": 0.XXX,
  "max_chrf": 0.XXX,
  "processing_time": 0.XX,
  "meta": {
    "ssa_score": 0.XXX
  }
}
```

---

## Usage Recommendations

### For Model Training

1. **Start with top75** - Best quality/quantity balance (~1.06M translations)
2. **Evaluate on top90** - Test generalization to slightly lower quality (~1.27M translations)
3. **Fine-tune on top50** - Optional refinement step (~705K translations)

### For Comparative Studies

Train identical models on each filtering level to quantify quality/quantity trade-off for your specific task (ASR, MT, etc.).

### Language-Specific Recommendations

#### Bemba (Large, Variable Quality)

- Use all three levels to demonstrate full filtering impact
- top50 still provides 397K translations (sufficient for training)
- Largest quality gains (+0.043 at top50)

#### Yoruba (Medium, Moderate Quality)

- All levels suitable for balanced experiments
- Good demonstration of filtering effectiveness
- Consistent improvements across levels

#### Igbo (Medium, High Baseline)

- top90 and top75 recommended (preserve data quantity)
- Already high quality baseline (min QE: 0.456)
- More conservative filtering appropriate

#### Hausa (Small, Highest Baseline)

- top90 likely sufficient (already min QE: 0.500)
- Limited data (27K) constrains aggressive filtering
- Use for high-quality validation/testing

### Expected Impact

Based on correlation validation (15 African languages):

- **Hausa:** Strong QE reliability (r=0.725 with chrF++)
- **Yoruba:** Good QE reliability (r=0.688 with chrF++)
- **Igbo:** Good QE reliability (r=0.687 with chrF++)
- **Bemba:** Limited validation data available

**Higher QE reliability → More effective filtering → Greater downstream improvement**

---

## Dataset Composition

### By Size

- Bemba: 56.4% of total (largest)
- Yoruba: 33.5% of total
- Igbo: 8.2% of total
- Hausa: 1.9% of total (smallest)

### By Quality Tier

**High-Baseline Tier (min QE > 0.45):**

- Hausa (min: 0.500)
- Igbo (min: 0.456)

**Mixed-Quality Tier (min QE < 0.41):**

- Yoruba (min: 0.407)
- Bemba (min: 0.237)

---

## Statistics

See `stats/filtering_summary.csv` for:

- Exact thresholds per language/level
- Retention rates
- QE score distributions
- Filtered dataset sizes
- Per-language quality improvements

---
