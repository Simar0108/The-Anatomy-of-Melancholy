"""
Phase 2.1: TF-IDF over corpus chunks. Fit on corpus vocabulary, transform chunks;
save matrix and vocabulary for clustering and anchor-word analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "corpus.parquet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "features"

# Reasonable defaults for literary text
MAX_FEATURES = 20_000
MIN_DF = 2
MAX_DF = 0.95
NGRAM_RANGE = (1, 2)


def fit_transform_tfidf(
    texts: list[str],
    *,
    max_features: int = MAX_FEATURES,
    min_df: int = MIN_DF,
    max_df: float = MAX_DF,
    ngram_range: tuple[int, int] = NGRAM_RANGE,
) -> tuple[np.ndarray, TfidfVectorizer]:
    """Fit TfidfVectorizer on texts and return (matrix, fitted vectorizer)."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        strip_accents="unicode",
        stop_words="english",
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    return X.astype(np.float32).toarray(), vectorizer


def run(
    corpus_path: Path | None = None,
    out_dir: Path | None = None,
    max_features: int = MAX_FEATURES,
    min_df: int = MIN_DF,
    max_df: float = MAX_DF,
) -> tuple[np.ndarray, TfidfVectorizer, list[str]]:
    """
    Load corpus, fit TF-IDF, save matrix and vocabulary. Returns (matrix, vectorizer, vocab).
    """
    corpus_path = corpus_path or DEFAULT_CORPUS_PATH
    out_dir = out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(corpus_path)
    texts = df["text"].astype(str).tolist()

    X, vectorizer = fit_transform_tfidf(
        texts, max_features=max_features, min_df=min_df, max_df=max_df
    )
    vocab = vectorizer.get_feature_names_out().tolist()

    np.save(out_dir / "tfidf_matrix.npy", X)
    with open(out_dir / "tfidf_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    print(f"TF-IDF: matrix shape {X.shape}, vocab size {len(vocab)} -> {out_dir}")
    return X, vectorizer, vocab


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Extract TF-IDF features from corpus")
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    run(corpus_path=args.corpus, out_dir=args.out_dir)
