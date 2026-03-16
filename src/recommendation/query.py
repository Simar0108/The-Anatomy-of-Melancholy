"""
Phase 7.2: Embed user text and return top-k corpus chunks by cosine similarity.
Maps results to book_id, chunk_id, and text for display.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.recommendation.embed_index import load_index, DEFAULT_FEATURES_DIR


def _embed_query(text: str, model_name: str) -> np.ndarray:
    """Encode a single query with the same model as the corpus (normalized)."""
    from src.features.embeddings import get_embeddings
    return get_embeddings([text], model_name=model_name)[0]


def recommend(
    query_text: str,
    k: int = 5,
    features_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Return top-k corpus chunks most similar to query_text (by cosine similarity).
    DataFrame columns: book_id, chunk_id, text, score, [label_suffering_type if present].
    """
    features_dir = features_dir or DEFAULT_FEATURES_DIR
    X, df, model_name = load_index(features_dir)

    q = _embed_query(query_text.strip(), model_name)
    q = q.astype(np.float32).reshape(1, -1)
    # Embeddings are already L2-normalized; dot product = cosine similarity
    scores = np.dot(X, q.T).ravel()

    top_idx = np.argsort(scores)[::-1][:k]
    out = df.iloc[top_idx][["book_id", "chunk_id", "text"]].copy()
    out["score"] = scores[top_idx]
    if "label_suffering_type" in df.columns:
        out["label_suffering_type"] = df.iloc[top_idx]["label_suffering_type"].values
    return out.reset_index(drop=True)
