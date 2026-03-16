"""
Phase 2.2: Sentence embeddings per chunk using sentence-transformers (e.g. all-MiniLM-L6-v2).
Saves embedding matrix for clustering, UMAP, and recommendation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "corpus.parquet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "features"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32


def get_embeddings(
    texts: list[str],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Encode texts with sentence-transformers; return (n_chunks, dim) float32 array."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def run(
    corpus_path: Path | None = None,
    out_dir: Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Load corpus, compute chunk embeddings, save matrix. Returns (n_chunks, dim) array."""
    corpus_path = corpus_path or DEFAULT_CORPUS_PATH
    out_dir = out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(corpus_path)
    texts = df["text"].astype(str).tolist()

    X = get_embeddings(texts, model_name=model_name, batch_size=batch_size)

    np.save(out_dir / "embeddings.npy", X)
    with open(out_dir / "embeddings_meta.txt", "w", encoding="utf-8") as f:
        f.write(f"model={model_name}\n")
        f.write(f"shape={X.shape[0]},{X.shape[1]}\n")

    print(f"Embeddings: shape {X.shape} -> {out_dir}")
    return X


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Extract sentence embeddings from corpus")
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    args = p.parse_args()
    run(corpus_path=args.corpus, out_dir=args.out_dir, model_name=args.model)
