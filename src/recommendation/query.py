"""
Phase 7.2: Embed user text and return top-k corpus chunks by cosine similarity.
Optional: re-rank top-K candidates with a cross-encoder for better precision.
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
    rerank: bool = False,
    rerank_top_k: int = 20,
    use_faiss: bool | None = None,
) -> pd.DataFrame:
    """
    Return top-k corpus chunks most similar to query_text (by cosine similarity).
    If rerank=True, retrieve rerank_top_k with bi-encoder then re-rank with cross-encoder and return top k.
    If use_faiss=True (or FAISS index exists and use_faiss is None), use FAISS for retrieval.
    DataFrame columns: book_id, chunk_id, text, score, chunk_index, [label_suffering_type if present].
    """
    features_dir = features_dir or DEFAULT_FEATURES_DIR
    X, df, model_name = load_index(features_dir)
    query_text = query_text.strip()
    retrieve_k = rerank_top_k if rerank else k

    q = _embed_query(query_text, model_name)
    q = q.astype(np.float32).reshape(1, -1)

    use_faiss_idx = (use_faiss is True) or (use_faiss is None and (features_dir / "embeddings.faiss").exists())
    if use_faiss_idx:
        try:
            from src.recommendation.faiss_index import load_index as load_faiss, search as faiss_search
            index = load_faiss(features_dir)
            scores_arr, top_idx = faiss_search(index, q, retrieve_k)
            # FAISS IndexFlatIP returns ascending order (smallest inner product first); we want best first
            order = np.argsort(scores_arr)[::-1]
            scores_arr = np.array(scores_arr, dtype=np.float64)[order]
            top_idx = top_idx.astype(np.int64)[order]
            scores = scores_arr
        except Exception:
            use_faiss_idx = False
    if not use_faiss_idx:
        scores = np.dot(X, q.T).ravel()
        top_idx = np.argsort(scores)[::-1][:retrieve_k]

    if rerank and retrieve_k > k:
        from src.recommendation.rerank import rerank as cross_rerank
        candidate_texts = df.iloc[top_idx]["text"].astype(str).tolist()
        order, rerank_scores = cross_rerank(query_text, candidate_texts, top_k=k)
        top_idx = top_idx[np.array(order)]
        scores = np.array(rerank_scores, dtype=np.float64)
    else:
        top_idx = top_idx[:k]
        scores = scores[:k] if hasattr(scores, "__getitem__") else np.full(k, scores)

    out = df.iloc[top_idx][["book_id", "chunk_id", "text"]].copy()
    out["score"] = scores
    out["chunk_index"] = top_idx
    if "label_suffering_type" in df.columns:
        out["label_suffering_type"] = df.iloc[top_idx]["label_suffering_type"].values
    return out.reset_index(drop=True)
