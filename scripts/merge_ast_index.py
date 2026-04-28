#!/usr/bin/env python3
"""
Merge Real + Synthetic AST Data → AST_SYNTH_INDEX.csv

Runs SSA-COMET-QE on the real AST_INDEX translations, then merges
with synthetic outputs from generate_ast_synthetic.py.

Usage:
    python scripts/merge_ast_index.py
    python scripts/merge_ast_index.py --skip-qe-real   # if already scored
    python scripts/merge_ast_index.py --dry-run         # print stats only
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from comet import download_model, load_from_checkpoint
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

AST_INDEX_PATH   = Path("/ocean/projects/cis250145p/shared/datasets/AST_INDEX.csv")
INTERMEDIATE_DIR = Path(
    "/ocean/projects/cis250145p/gichamba/stt/mt-correlation-incremental-filtering/outputs/ast_synth"
)
OUTPUT_PATH = Path("/ocean/projects/cis250145p/shared/datasets/AST_SYNTH_INDEX.csv")

QE_MODEL   = "McGill-NLP/ssa-comet-qe"
LANGUAGES  = ["igbo", "bemba", "hausa", "yoruba"]

AST_SCHEMA = [
    "audio_id", "path", "transcript", "translation",
    "src_language", "tgt_language", "split",
    "source", "speaker_id", "sample_rate", "duration",
    "translation_source", "qe_score",
]

# ─── QE scoring ───────────────────────────────────────────────────────────────

def load_qe_model(model_name: str = QE_MODEL):
    logger.info(f"Loading QE model: {model_name}")
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    logger.info("QE model loaded.")
    return model


def score_dataframe_qe(df: pd.DataFrame, qe_model, batch_size: int = 32) -> list[float]:
    """
    Score all rows in a dataframe with SSA-COMET-QE.
    Uses transcript as src, translation as mt.
    """
    transcripts  = df["transcript"].fillna("").tolist()
    translations = df["translation"].fillna("").tolist()

    all_scores = []

    for i in tqdm(range(0, len(transcripts), batch_size), desc="QE scoring"):
        batch_src = transcripts[i:i + batch_size]
        batch_mt  = translations[i:i + batch_size]

        data = [{"src": s, "mt": m} for s, m in zip(batch_src, batch_mt)]

        output = qe_model.predict(
            data,
            batch_size=len(data),
            gpus=1 if torch.cuda.is_available() else 0,
        )
        all_scores.extend(output.scores)

    return all_scores


# ─── Load helpers ─────────────────────────────────────────────────────────────

def load_real_ast(skip_qe: bool, qe_model) -> pd.DataFrame:
    """Load AST_INDEX, add translation_source and optionally qe_score."""
    logger.info(f"Loading real AST data from {AST_INDEX_PATH}")
    df = pd.read_csv(AST_INDEX_PATH, dtype=str)
    logger.info(f"Real AST rows: {len(df):,}")

    df["translation_source"] = "real"

    if skip_qe:
        logger.info("Skipping QE scoring for real data (--skip-qe-real flag set).")
        df["qe_score"] = None
    else:
        logger.info("Running SSA-COMET-QE on real translations...")
        df["qe_score"] = score_dataframe_qe(df, qe_model)
        logger.info(
            f"Real data QE — mean: {df['qe_score'].mean():.4f}, "
            f"min: {df['qe_score'].min():.4f}, "
            f"max: {df['qe_score'].max():.4f}"
        )

    return df


def load_synthetic_ast() -> pd.DataFrame:
    """Load all per-language synthetic CSVs from intermediate dir."""
    frames = []
    missing = []

    for lang in LANGUAGES:
        synth_file = INTERMEDIATE_DIR / f"{lang}_synth.csv"
        checkpoint_file = INTERMEDIATE_DIR / f"{lang}_synth_checkpoint.csv"

        if synth_file.exists():
            logger.info(f"[{lang}] Loading finalized synthetic file: {synth_file}")
            frames.append(pd.read_csv(synth_file, dtype=str))
        elif checkpoint_file.exists():
            logger.warning(
                f"[{lang}] Only checkpoint found (job may not have finished). "
                f"Loading checkpoint: {checkpoint_file}"
            )
            frames.append(pd.read_csv(checkpoint_file, dtype=str))
        else:
            logger.warning(f"[{lang}] No synthetic file found. Skipping.")
            missing.append(lang)

    if missing:
        logger.warning(f"Missing synthetic data for: {missing}")

    if not frames:
        logger.error("No synthetic data found. Run generate_ast_synthetic.py first.")
        sys.exit(1)

    synth_df = pd.concat(frames, ignore_index=True)
    logger.info(f"Total synthetic rows loaded: {len(synth_df):,}")
    return synth_df


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Merge real + synthetic AST data")
    parser.add_argument(
        "--skip-qe-real",
        action="store_true",
        help="Skip QE scoring for real AST data (set qe_score=null for real rows)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing output file",
    )
    parser.add_argument(
        "--qe-model",
        default=QE_MODEL,
        help=f"QE model name (default: {QE_MODEL})",
    )
    parser.add_argument(
        "--qe-batch-size",
        type=int,
        default=32,
        help="Batch size for QE scoring of real data (default: 32)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output path (default: {OUTPUT_PATH})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load QE model (needed unless --skip-qe-real and no scoring needed)
    qe_model = None
    if not args.skip_qe_real:
        qe_model = load_qe_model(args.qe_model)

    # Load real data
    real_df = load_real_ast(skip_qe=args.skip_qe_real, qe_model=qe_model)

    # Load synthetic data
    synth_df = load_synthetic_ast()

    # Align schemas — add any missing columns
    for col in AST_SCHEMA:
        if col not in real_df.columns:
            real_df[col] = None
        if col not in synth_df.columns:
            synth_df[col] = None

    real_df  = real_df[AST_SCHEMA]
    synth_df = synth_df[AST_SCHEMA]

    # Merge
    merged_df = pd.concat([real_df, synth_df], ignore_index=True)
    merged_df = merged_df.sort_values("audio_id").reset_index(drop=True)

    # ── Statistics ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("MERGED DATASET STATISTICS")
    logger.info("=" * 70)
    logger.info(f"Total rows:      {len(merged_df):,}")
    logger.info(f"  Real:          {len(real_df):,}")
    logger.info(f"  Synthetic:     {len(synth_df):,}")

    logger.info("\nBreakdown by language:")
    lang_counts = merged_df.groupby(["src_language", "translation_source"]).size().unstack(fill_value=0)
    logger.info(f"\n{lang_counts.to_string()}")

    logger.info("\nBreakdown by split:")
    split_counts = merged_df.groupby(["split", "translation_source"]).size().unstack(fill_value=0)
    logger.info(f"\n{split_counts.to_string()}")

    numeric_qe = pd.to_numeric(merged_df["qe_score"], errors="coerce")
    if numeric_qe.notna().any():
        logger.info(f"\nQE score stats (scored rows only):")
        logger.info(f"  Mean:   {numeric_qe.mean():.4f}")
        logger.info(f"  Median: {numeric_qe.median():.4f}")
        logger.info(f"  Min:    {numeric_qe.min():.4f}")
        logger.info(f"  Max:    {numeric_qe.max():.4f}")

        for lang in LANGUAGES:
            lang_mask = (merged_df["src_language"] == lang)
            lang_qe   = numeric_qe[lang_mask]
            if lang_qe.notna().any():
                logger.info(
                    f"  {lang:<8}: mean={lang_qe.mean():.4f} "
                    f"min={lang_qe.min():.4f} max={lang_qe.max():.4f} "
                    f"n={lang_qe.notna().sum():,}"
                )

    if args.dry_run:
        logger.info("\n--dry-run set. No output written.")
        return

    # ── Write output ──────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(args.output, index=False)
    logger.info(f"\nOutput written to: {args.output}")
    logger.info(f"Rows: {len(merged_df):,}")


if __name__ == "__main__":
    main()
