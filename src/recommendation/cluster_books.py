"""
Cluster-based book recommendation: use K-means cluster centroids to recommend
whole books (e.g. from Project Gutenberg) that match each cluster's philosophical
style. Enables "books like this cluster" and discovering new books to add to the study.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "data" / "features"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
# Sample this many characters from each candidate book for embedding (avoid full download)
SAMPLE_CHARS = 20_000
# Split sample into chunks of this many words for mean embedding
SAMPLE_CHUNK_WORDS = 500


def load_cluster_centroids(
    features_dir: Path | None = None,
    results_dir: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Load chunk embeddings and K-means labels; compute L2-normalized centroid per cluster.
    Returns (centroids, cluster_descriptions) where centroids is (n_clusters, dim).
    """
    features_dir = features_dir or DEFAULT_FEATURES_DIR
    results_dir = results_dir or DEFAULT_RESULTS_DIR

    X = np.load(features_dir / "embeddings.npy").astype(np.float32)
    labels_df = pd.read_csv(results_dir / "labels_kmeans.csv")
    labels = labels_df["cluster_kmeans"].values
    if len(labels) != len(X):
        raise ValueError(f"labels length {len(labels)} != embeddings {len(X)}")

    clusters = sorted(set(labels))
    centroids = []
    for c in clusters:
        mask = labels == c
        cen = np.mean(X[mask], axis=0).astype(np.float32)
        norm = np.linalg.norm(cen)
        if norm > 0:
            cen = cen / norm
        centroids.append(cen)
    centroids = np.stack(centroids)

    # Optional: short description from cluster_descriptions.csv
    desc_path = results_dir / "cluster_descriptions.csv"
    descriptions = [f"cluster_{c}" for c in clusters]
    if desc_path.exists():
        desc_df = pd.read_csv(desc_path)
        for _, row in desc_df.iterrows():
            c = row.get("cluster", None)
            if c is not None and c in clusters:
                idx = clusters.index(c)
                terms = row.get("top_terms", "")[:60]
                descriptions[idx] = terms or descriptions[idx]

    return centroids, descriptions


def _embed_text_sample(text: str, model_name: str) -> np.ndarray:
    """Split text into chunks, embed each, return L2-normalized mean vector."""
    from src.features.embeddings import get_embeddings

    text = text[: SAMPLE_CHARS].strip()
    if not text or len(text) < 200:
        return None
    words = text.split()
    chunks = []
    for i in range(0, len(words), SAMPLE_CHUNK_WORDS):
        chunk = " ".join(words[i : i + SAMPLE_CHUNK_WORDS])
        if len(chunk) > 100:
            chunks.append(chunk)
    if not chunks:
        return None
    emb = get_embeddings(chunks, model_name=model_name)
    mean = np.mean(emb, axis=0).astype(np.float32)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    return mean


def _get_model_name(features_dir: Path) -> str:
    meta = features_dir / "embeddings_meta.txt"
    name = "all-MiniLM-L6-v2"
    if meta.exists():
        for line in meta.read_text(encoding="utf-8").strip().splitlines():
            if line.startswith("model="):
                name = line.split("=", 1)[1].strip()
                break
    return name


def recommend_books_for_clusters(
    candidates: pd.DataFrame,
    features_dir: Path | None = None,
    results_dir: Path | None = None,
    top_per_cluster: int = 5,
    fetch_sample: bool = True,
) -> pd.DataFrame:
    """
    For each cluster, rank candidate books by cosine similarity to cluster centroid.
    candidates must have columns: gutenberg_id, title. Optional: sample_text (if not set, we fetch from Gutenberg when fetch_sample=True).
    Returns DataFrame: cluster, cluster_desc, gutenberg_id, title, score, rank_in_cluster.
    """
    features_dir = features_dir or DEFAULT_FEATURES_DIR
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    model_name = _get_model_name(features_dir)
    centroids, descriptions = load_cluster_centroids(features_dir, results_dir)

    if "sample_text" not in candidates.columns and fetch_sample:
        import time
        from src.download_gutenberg import fetch_text, DELAY_SEC
        samples = []
        for _, row in candidates.iterrows():
            gid = row["gutenberg_id"]
            try:
                full = fetch_text(int(gid))
                samples.append(full[:SAMPLE_CHARS] if len(full) > SAMPLE_CHARS else full)
            except Exception as e:
                samples.append("")
                print(f"  Warning: could not fetch Gutenberg {gid}: {e}")
            time.sleep(DELAY_SEC)
        candidates = candidates.copy()
        candidates["sample_text"] = samples
        fetch_sample = False

    if "sample_text" not in candidates.columns:
        raise ValueError("candidates need 'sample_text' or fetch_sample=True with gutenberg_id to fetch")

    book_vectors = []
    for _, row in candidates.iterrows():
        text = row.get("sample_text", "")
        v = _embed_text_sample(str(text), model_name)
        book_vectors.append(v if v is not None else np.zeros(centroids.shape[1], dtype=np.float32))
    book_matrix = np.stack(book_vectors)

    # (n_candidates, n_clusters) scores
    scores = np.dot(book_matrix, centroids.T)

    rows = []
    for c_idx in range(centroids.shape[0]):
        order = np.argsort(scores[:, c_idx])[::-1][:top_per_cluster]
        for rank, idx in enumerate(order, 1):
            rows.append({
                "cluster": c_idx,
                "cluster_desc": descriptions[c_idx][:80] if c_idx < len(descriptions) else f"cluster_{c_idx}",
                "gutenberg_id": candidates.iloc[idx]["gutenberg_id"],
                "title": candidates.iloc[idx]["title"],
                "score": float(scores[idx, c_idx]),
                "rank_in_cluster": rank,
            })
    return pd.DataFrame(rows)


def get_default_candidates_path() -> Path:
    return PROJECT_ROOT / "data" / "candidates_gutenberg.csv"
