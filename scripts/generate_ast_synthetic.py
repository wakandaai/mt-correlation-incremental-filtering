#!/usr/bin/env python3
"""
Synthetic AST Generation Script
Translates African language transcripts to English using NLLB-200,
scores with SSA-COMET-QE, and checkpoints every N batches.

Usage:
    python scripts/generate_ast_synthetic.py --languages igbo
    python scripts/generate_ast_synthetic.py --languages all
    python scripts/generate_ast_synthetic.py --languages igbo bemba
"""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from comet import download_model, load_from_checkpoint
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

ASR_INDEX_PATH = Path("/ocean/projects/cis250145p/shared/datasets/ASR_INDEX.csv")
INTERMEDIATE_DIR = Path(
    "/ocean/projects/cis250145p/gichamba/stt/mt-correlation-incremental-filtering/outputs/ast_synth"
)

# ─── Language config ──────────────────────────────────────────────────────────

LANGUAGE_CONFIG = {
    "igbo":   {"nllb_code": "ibo_Latn", "tgt_code": "eng_Latn"},
    "bemba":  {"nllb_code": "bem_Latn", "tgt_code": "eng_Latn"},
    "hausa":  {"nllb_code": "hau_Latn", "tgt_code": "eng_Latn"},
    "yoruba": {"nllb_code": "yor_Latn", "tgt_code": "eng_Latn"},
}

ALL_LANGUAGES = list(LANGUAGE_CONFIG.keys())

# ─── Constants ────────────────────────────────────────────────────────────────

CHECKPOINT_EVERY_N_BATCHES = 50   # checkpoint frequency
NLLB_MODEL = "facebook/nllb-200-3.3B"
QE_MODEL   = "McGill-NLP/ssa-comet-qe"

AST_SCHEMA = [
    "audio_id", "path", "transcript", "translation",
    "src_language", "tgt_language", "split",
    "source", "speaker_id", "sample_rate", "duration",
    "translation_source", "qe_score",
]


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def progress_path(language: str) -> Path:
    return INTERMEDIATE_DIR / f"{language}_synth_progress.json"

def checkpoint_path(language: str) -> Path:
    return INTERMEDIATE_DIR / f"{language}_synth_checkpoint.csv"

def output_path(language: str) -> Path:
    return INTERMEDIATE_DIR / f"{language}_synth.csv"


def load_progress(language: str) -> dict | None:
    p = progress_path(language)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def save_progress(language: str, progress: dict):
    progress["updated_at"] = datetime.utcnow().isoformat()
    with open(progress_path(language), "w") as f:
        json.dump(progress, f, indent=2)


def append_checkpoint(language: str, rows: list[dict]):
    """Append a list of row dicts to the checkpoint CSV."""
    cp = checkpoint_path(language)
    df = pd.DataFrame(rows, columns=AST_SCHEMA)
    write_header = not cp.exists()
    df.to_csv(cp, mode="a", header=write_header, index=False)


def finalize(language: str):
    """Rename checkpoint → final output, remove progress file."""
    shutil.move(str(checkpoint_path(language)), str(output_path(language)))
    p = progress_path(language)
    if p.exists():
        p.unlink()
    logger.info(f"[{language}] Finalized → {output_path(language)}")


# ─── Model loading ────────────────────────────────────────────────────────────

class NLLBTranslator:
    def __init__(self, model_name: str = NLLB_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading NLLB model on {self.device}: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("NLLB model loaded.")

    def translate(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str = "eng_Latn",
        max_length: int = 512,
    ) -> list[str]:
        self.tokenizer.src_lang = src_lang
        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
            )

        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


def load_qe_model(model_name: str = QE_MODEL):
    logger.info(f"Loading QE model: {model_name}")
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    logger.info("QE model loaded.")
    return model


def score_qe(
    transcripts: list[str],
    translations: list[str],
    qe_model,
) -> list[float]:
    """SSA-COMET-QE: reference-free, takes (src, mt) only."""
    data = [
        {"src": src, "mt": mt}
        for src, mt in zip(transcripts, translations)
    ]
    output = qe_model.predict(
        data,
        batch_size=len(data),
        gpus=1 if torch.cuda.is_available() else 0,
    )
    return output.scores


# ─── Core processing ──────────────────────────────────────────────────────────

def process_language(
    language: str,
    nllb: NLLBTranslator,
    qe_model,
    batch_size: int,
):
    lang_cfg = LANGUAGE_CONFIG[language]
    src_lang  = lang_cfg["nllb_code"]

    # ── Load ASR index, filter to this language train split ──────────────────
    logger.info(f"[{language}] Loading ASR_INDEX...")
    asr_df = pd.read_csv(ASR_INDEX_PATH, dtype=str)
    asr_df = asr_df[
        (asr_df["language"] == language) &
        (asr_df["split"] == "train")
    ].reset_index(drop=True)

    total_rows = len(asr_df)
    logger.info(f"[{language}] {total_rows:,} train rows to process.")

    if total_rows == 0:
        logger.warning(f"[{language}] No rows found. Skipping.")
        return

    # ── Resume logic ─────────────────────────────────────────────────────────
    progress = load_progress(language)
    if progress:
        start_batch = progress["last_completed_batch"] + 1
        rows_done   = progress["last_completed_row"] + 1
        logger.info(
            f"[{language}] Resuming from batch {start_batch} "
            f"({rows_done:,} rows already done)."
        )
    else:
        start_batch = 0
        rows_done   = 0
        progress = {
            "language":             language,
            "total_rows":           total_rows,
            "last_completed_batch": -1,
            "last_completed_row":   -1,
            "batches_total":        (total_rows + batch_size - 1) // batch_size,
            "started_at":           datetime.utcnow().isoformat(),
            "updated_at":           datetime.utcnow().isoformat(),
        }
        save_progress(language, progress)

    # ── Slice to unprocessed rows ─────────────────────────────────────────────
    remaining_df = asr_df.iloc[rows_done:].reset_index(drop=True)
    total_batches = (len(remaining_df) + batch_size - 1) // batch_size

    logger.info(f"[{language}] {len(remaining_df):,} rows remaining across {total_batches} batches.")

    # ── Interleaved translate + QE + checkpoint ───────────────────────────────
    buffer: list[dict] = []

    with tqdm(total=len(remaining_df), desc=f"{language}", unit="row") as pbar:
        for batch_idx in range(total_batches):
            global_batch_idx = start_batch + batch_idx

            batch_start = batch_idx * batch_size
            batch_end   = min(batch_start + batch_size, len(remaining_df))
            batch_df    = remaining_df.iloc[batch_start:batch_end]

            transcripts = batch_df["transcript"].fillna("").tolist()

            # Translate
            try:
                translations = nllb.translate(transcripts, src_lang=src_lang)
            except Exception as e:
                logger.error(f"[{language}] NLLB failed on batch {global_batch_idx}: {e}")
                raise

            # QE score
            try:
                qe_scores = score_qe(transcripts, translations, qe_model)
            except Exception as e:
                logger.error(f"[{language}] QE failed on batch {global_batch_idx}: {e}")
                raise

            # Build rows
            for i, (_, row) in enumerate(batch_df.iterrows()):
                buffer.append({
                    "audio_id":           row.get("audio_id", ""),
                    "path":               row.get("path", ""),
                    "transcript":         transcripts[i],
                    "translation":        translations[i],
                    "src_language":       language,
                    "tgt_language":       "english",
                    "split":              row.get("split", "train"),
                    "source":             row.get("source", ""),
                    "speaker_id":         row.get("speaker_id", ""),
                    "sample_rate":        row.get("sample_rate", ""),
                    "duration":           row.get("duration", ""),
                    "translation_source": "synthetic",
                    "qe_score":           qe_scores[i],
                })

            pbar.update(len(batch_df))

            # Checkpoint every N batches
            if (batch_idx + 1) % CHECKPOINT_EVERY_N_BATCHES == 0 or batch_idx == total_batches - 1:
                append_checkpoint(language, buffer)
                buffer = []

                last_row = rows_done + batch_end - 1
                progress["last_completed_batch"] = global_batch_idx
                progress["last_completed_row"]   = last_row
                save_progress(language, progress)

                logger.info(
                    f"[{language}] Checkpointed at batch {global_batch_idx} "
                    f"| rows done: {last_row + 1:,}/{total_rows:,}"
                )

    # Flush any remaining buffer (shouldn't happen but safety net)
    if buffer:
        append_checkpoint(language, buffer)

    finalize(language)
    logger.info(f"[{language}] Complete. Output: {output_path(language)}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic AST translations")
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help="Languages to process: igbo bemba hausa yoruba | all",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for NLLB translation and QE scoring (default: 8)",
    )
    parser.add_argument(
        "--nllb-model",
        default=NLLB_MODEL,
        help=f"NLLB model name (default: {NLLB_MODEL})",
    )
    parser.add_argument(
        "--qe-model",
        default=QE_MODEL,
        help=f"QE model name (default: {QE_MODEL})",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY_N_BATCHES,
        help="Checkpoint every N batches (default: 50)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve language list
    if args.languages == ["all"]:
        languages = ALL_LANGUAGES
    else:
        languages = []
        for lang in args.languages:
            if lang not in LANGUAGE_CONFIG:
                logger.error(f"Unknown language: {lang}. Choose from {ALL_LANGUAGES} or 'all'.")
                sys.exit(1)
            languages.append(lang)

    logger.info(f"Languages to process: {languages}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Checkpoint every: {args.checkpoint_every} batches")

    # Create output dir
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    # Load models once, reuse across languages
    nllb     = NLLBTranslator(model_name=args.nllb_model)
    qe_model = load_qe_model(model_name=args.qe_model)

    # Process each language
    for language in languages:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {language.upper()}")
        logger.info(f"{'='*70}")
        try:
            process_language(
                language=language,
                nllb=nllb,
                qe_model=qe_model,
                batch_size=args.batch_size,
            )
        except Exception as e:
            logger.error(f"[{language}] Failed: {e}")
            logger.error("Progress saved — re-run to resume from last checkpoint.")
            sys.exit(1)

    logger.info("\nAll languages complete.")


if __name__ == "__main__":
    main()
