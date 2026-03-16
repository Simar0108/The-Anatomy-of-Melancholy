#!/usr/bin/env python3
"""
Phase 2 pipeline: extract all features (TF-IDF, embeddings, syntactic, sentiment)
and produce data/features/ outputs + corpus_features.parquet (metadata + tabular features).

Run from project root with venv activated:
  source .venv/bin/activate
  python run_phase2.py

Requires: data/processed/corpus.parquet (from run_phase1.py).
Outputs:
  data/features/tfidf_matrix.npy, tfidf_vocab.json
  data/features/embeddings.npy, embeddings_meta.txt
  data/features/syntactic.parquet
  data/features/sentiment.parquet
  data/features/corpus_features.parquet  (corpus rows + syntactic + sentiment; matrices by index)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Run from project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "corpus.parquet"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"


def main() -> None:
    if not CORPUS_PATH.exists():
        print(f"Missing {CORPUS_PATH}. Run: python run_phase1.py")
        sys.exit(1)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # 2.1 TF-IDF
    from src.features.tfidf import run as run_tfidf
    run_tfidf(corpus_path=CORPUS_PATH, out_dir=FEATURES_DIR)

    # 2.2 Embeddings (can be slow on first run due to model download)
    from src.features.embeddings import run as run_embeddings
    run_embeddings(corpus_path=CORPUS_PATH, out_dir=FEATURES_DIR)

    # 2.3 Syntactic
    from src.features.syntactic import run as run_syntactic
    run_syntactic(corpus_path=CORPUS_PATH, out_dir=FEATURES_DIR)

    # 2.4 Sentiment
    from src.features.sentiment import run as run_sentiment
    run_sentiment(corpus_path=CORPUS_PATH, out_dir=FEATURES_DIR)

    # 2.5 Single feature table: corpus columns + syntactic + sentiment (matrices stay as .npy)
    corpus = pd.read_parquet(CORPUS_PATH)
    syn = pd.read_parquet(FEATURES_DIR / "syntactic.parquet")
    sent = pd.read_parquet(FEATURES_DIR / "sentiment.parquet")
    features_df = pd.concat([corpus, syn, sent], axis=1)
    out_path = FEATURES_DIR / "corpus_features.parquet"
    features_df.to_parquet(out_path, index=False)
    print(f"Corpus features table: {len(features_df)} rows -> {out_path}")

    print("Phase 2 done. Use data/features/ for Phase 3–5 (clustering, UMAP, trajectory).")


if __name__ == "__main__":
    main()
