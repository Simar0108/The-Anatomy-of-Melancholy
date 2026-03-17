"""
FAISS index for scalable similarity search over chunk embeddings.
Build index from embeddings.npy (normalized); query returns top-k indices.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "data" / "features"


def build_index(features_dir: Path | None = None) -> Path:
    """Build FAISS IndexFlatIP from embeddings.npy (must be L2-normalized). Returns path to saved index."""
    import faiss
    features_dir = features_dir or DEFAULT_FEATURES_DIR
    X = np.load(features_dir / "embeddings.npy").astype(np.float32)
    if X.flags.c_contiguous is False:
        X = np.ascontiguousarray(X)
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)
    out_path = features_dir / "embeddings.faiss"
    faiss.write_index(index, str(out_path))
    return out_path


def load_index(features_dir: Path | None = None):
    """Load FAISS index from features_dir/embeddings.faiss. Returns faiss.Index."""
    import faiss
    features_dir = features_dir or DEFAULT_FEATURES_DIR
    path = features_dir / "embeddings.faiss"
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}. Run: python scripts/build_faiss_index.py")
    return faiss.read_index(str(path))


def search(index, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Query FAISS index. query_vec: (1, dim) float32 normalized. Returns (distances, indices)."""
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    query_vec = np.ascontiguousarray(query_vec.astype(np.float32))
    distances, indices = index.search(query_vec, k)
    return distances[0], indices[0]
