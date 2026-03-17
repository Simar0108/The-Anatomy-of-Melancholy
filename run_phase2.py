#!/usr/bin/env python3
"""
Phase 2 pipeline: extract all features (TF-IDF, embeddings, syntactic, sentiment)
and produce data/features/ outputs + corpus_features.parquet (metadata + tabular features).

Run from project root with venv activated:
  source .venv/bin/activate
  python run_phase2.py
  python run_phase2.py --embedding-model all-mpnet-base-v2 --sentiment transformer

Requires: data/processed/corpus.parquet (from run_phase1.py).
Outputs:
  data/features/tfidf_matrix.npy, tfidf_vocab.json
  data/features/embeddings.npy, embeddings_meta.txt
  data/features/syntactic.parquet
  data/features/sentiment.parquet, sentiment_meta.txt (if transformer)
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
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def _load_config() -> dict:
    out = {"embedding_model": "all-MiniLM-L6-v2", "sentiment_backend": "vader"}
    if CONFIG_PATH.exists():
        try:
            import yaml
            with open(CONFIG_PATH, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            out.update({k: v for k, v in data.items() if v is not None})
        except ImportError:
            pass  # PyYAML optional; pip install pyyaml to use config/default.yaml
        except Exception:
            pass
    return out


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Phase 2: extract all features")
    p.add_argument("--embedding-model", type=str, default=None,
                    help="Sentence-transformer model (e.g. all-MiniLM-L6-v2, all-mpnet-base-v2)")
    p.add_argument("--sentiment", type=str, choices=["vader", "transformer"], default=None,
                    help="Sentiment backend: vader or transformer (HF)")
    args = p.parse_args()

    cfg = _load_config()
    embedding_model = args.embedding_model or cfg["embedding_model"]
    sentiment_backend = args.sentiment or cfg["sentiment_backend"]

    if not CORPUS_PATH.exists():
        print(f"Missing {CORPUS_PATH}. Run: python run_phase1.py")
        sys.exit(1)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # 2.1 TF-IDF
    from src.features.tfidf import run as run_tfidf
    run_tfidf(corpus_path=CORPUS_PATH, out_dir=FEATURES_DIR)

    # 2.2 Embeddings (can be slow on first run due to model download)
    from src.features.embeddings import run as run_embeddings
    run_embeddings(corpus_path=CORPUS_PATH, out_dir=FEATURES_DIR, model_name=embedding_model)

    # 2.3 Syntactic
    from src.features.syntactic import run as run_syntactic
    run_syntactic(corpus_path=CORPUS_PATH, out_dir=FEATURES_DIR)

    # 2.4 Sentiment
    if sentiment_backend == "transformer":
        from src.features.sentiment_transformer import run as run_sentiment
        run_sentiment(corpus_path=CORPUS_PATH, out_dir=FEATURES_DIR)
    else:
        from src.features.sentiment import run as run_sentiment
        run_sentiment(corpus_path=CORPUS_PATH, out_dir=FEATURES_DIR)
    with open(FEATURES_DIR / "sentiment_meta.txt", "w", encoding="utf-8") as f:
        f.write(f"backend={sentiment_backend}\n")

    # 2.5 Single feature table: corpus columns + syntactic + sentiment (matrices stay as .npy)
    corpus = pd.read_parquet(CORPUS_PATH)
    syn = pd.read_parquet(FEATURES_DIR / "syntactic.parquet")
    sent = pd.read_parquet(FEATURES_DIR / "sentiment.parquet")
    features_df = pd.concat([corpus, syn, sent], axis=1)
    out_path = FEATURES_DIR / "corpus_features.parquet"
    features_df.to_parquet(out_path, index=False)
    print(f"Corpus features table: {len(features_df)} rows -> {out_path}")

    print(f"Phase 2 done. Embedding model: {embedding_model}; Sentiment: {sentiment_backend}.")


if __name__ == "__main__":
    main()
