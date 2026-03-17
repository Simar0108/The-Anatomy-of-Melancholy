"""
Cross-encoder re-ranker: take (query, top-K candidate chunks) and return reordered
indices by relevance score. Uses sentence-transformers cross-encoder (e.g. ms-marco).
"""
from __future__ import annotations

from pathlib import Path

# Default: small cross-encoder for query-document scoring
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank(
    query: str,
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    top_k: int = 5,
) -> tuple[list[int], list[float]]:
    """
    Score (query, text) pairs with a cross-encoder; return indices and scores for top_k.
    texts: list of candidate chunk texts (same order as corpus rows).
    Returns (indices, scores) where indices are the top_k positions in texts (0-based).
    """
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name)
    pairs = [(query, t) for t in texts]
    scores = model.predict(pairs)
    scores = scores if hasattr(scores, "__iter__") else [float(scores)]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return order, [float(scores[i]) for i in order]
