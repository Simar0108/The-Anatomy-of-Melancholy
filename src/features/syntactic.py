"""
Phase 2.3: Syntactic features per chunk — sentence length (mean, std), punctuation density.
Uses spaCy when available, with a regex fallback so the pipeline runs without spaCy model.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "corpus.parquet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "features"


def _sentence_lengths_regex(text: str) -> list[float]:
    """Split on sentence boundaries (., !, ?) and return list of word counts per sentence."""
    if not text or not text.strip():
        return [0.0]
    # Simple split; keep periods etc. attached then count words per fragment
    fragments = re.split(r"[.!?]+\s+", text)
    lengths = [len(f.split()) for f in fragments if f.strip()]
    return [float(x) for x in lengths] if lengths else [0.0]


def _sentence_lengths_spacy(text: str, nlp) -> list[float]:
    """Use spaCy to get sentence boundaries and word counts per sentence."""
    if not text or not text.strip():
        return [0.0]
    doc = nlp(text[:1_000_000])  # cap size for long chunks
    lengths = [len(sent) for sent in doc.sents]
    return [float(x) for x in lengths] if lengths else [0.0]


def _punctuation_density(text: str) -> float:
    """Ratio of punctuation characters to total non-space characters."""
    if not text or not text.strip():
        return 0.0
    punc = sum(1 for c in text if c in ".,;:!?\"'—–-()[]")
    total = sum(1 for c in text if not c.isspace())
    return punc / total if total else 0.0


def extract_syntactic_one(text: str, nlp=None) -> dict:
    """Extract mean_sent_len, std_sent_len, punc_density for one chunk."""
    if nlp is not None:
        lengths = _sentence_lengths_spacy(text, nlp)
    else:
        lengths = _sentence_lengths_regex(text)
    mean_sent_len = float(np.mean(lengths)) if lengths else 0.0
    std_sent_len = float(np.std(lengths)) if len(lengths) > 1 else 0.0
    punc_density = _punctuation_density(text)
    return {
        "mean_sent_len": mean_sent_len,
        "std_sent_len": std_sent_len,
        "punc_density": punc_density,
    }


def run(
    corpus_path: Path | None = None,
    out_dir: Path | None = None,
    use_spacy: bool = True,
) -> pd.DataFrame:
    """
    Load corpus, compute syntactic features per chunk, merge into dataframe and save.
    Returns DataFrame with corpus index plus mean_sent_len, std_sent_len, punc_density.
    """
    corpus_path = corpus_path or DEFAULT_CORPUS_PATH
    out_dir = out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(corpus_path)
    texts = df["text"].astype(str).tolist()

    nlp = None
    if use_spacy:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except OSError:
            print("spaCy model 'en_core_web_sm' not found; using regex sentence split.")
            use_spacy = False

    rows = []
    for i, text in enumerate(texts):
        row = extract_syntactic_one(text, nlp=nlp)
        rows.append(row)

    syn_df = pd.DataFrame(rows)
    out_path = out_dir / "syntactic.parquet"
    syn_df.to_parquet(out_path, index=False)
    print(f"Syntactic: {len(syn_df)} rows -> {out_path}")
    return syn_df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Extract syntactic features from corpus")
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-spacy", action="store_true", help="Use regex only")
    args = p.parse_args()
    run(corpus_path=args.corpus, out_dir=args.out_dir, use_spacy=not args.no_spacy)
