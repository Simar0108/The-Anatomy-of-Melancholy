"""
Phase 7.1: Load corpus chunk embeddings and metadata for similarity search.
Uses same embeddings as Phase 2 (sentence-transformers); row order matches corpus.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "data" / "features"


def load_index(features_dir: Path | None = None) -> tuple[np.ndarray, pd.DataFrame, str]:
    """
    Load embedding matrix and corpus metadata from Phase 2 outputs.
    Returns (embeddings, corpus_df, model_name).
    - embeddings: (n_chunks, dim) float32, L2-normalized
    - corpus_df: has columns book_id, chunk_id, text (and others)
    - model_name: from embeddings_meta.txt for consistent query encoding
    """
    features_dir = features_dir or DEFAULT_FEATURES_DIR
    emb_path = features_dir / "embeddings.npy"
    meta_path = features_dir / "embeddings_meta.txt"
    corpus_path = features_dir / "corpus_features.parquet"

    if not emb_path.exists():
        raise FileNotFoundError(
            f"Missing {emb_path}. Run: python run_phase2.py"
        )
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Missing {corpus_path}. Run: python run_phase2.py"
        )

    X = np.load(emb_path).astype(np.float32)
    df = pd.read_parquet(corpus_path)
    if len(df) != len(X):
        raise ValueError(
            f"Row count mismatch: corpus {len(df)} vs embeddings {len(X)}"
        )

    model_name = "all-MiniLM-L6-v2"
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").strip().splitlines():
            if line.startswith("model="):
                model_name = line.split("=", 1)[1].strip()
                break

    return X, df, model_name
