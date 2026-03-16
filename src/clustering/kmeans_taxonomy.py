"""
Phase 3: Suffering taxonomy — K-Means and hierarchical clustering on chunk embeddings,
with cluster descriptions (top TF-IDF terms) and cross-tabs (cluster vs book / label).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "data" / "features"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

# Clustering defaults
K_RANGE = (2, 11)  # try k from 2 to 10 for elbow/silhouette
DEFAULT_K = 5
LINKAGE_METHOD = "ward"
TOP_TERMS_PER_CLUSTER = 15


def load_embedding_matrix(features_dir: Path) -> np.ndarray:
    """Load (n_chunks, dim) embedding matrix."""
    path = features_dir / "embeddings.npy"
    if not path.exists():
        raise FileNotFoundError(f"Run Phase 2 first: missing {path}")
    return np.load(path).astype(np.float32)


def load_tfidf_and_vocab(features_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load TF-IDF matrix and vocabulary list."""
    mat_path = features_dir / "tfidf_matrix.npy"
    vocab_path = features_dir / "tfidf_vocab.json"
    if not mat_path.exists() or not vocab_path.exists():
        raise FileNotFoundError(f"Run Phase 2 first: missing {mat_path} or {vocab_path}")
    X = np.load(mat_path).astype(np.float32)
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    return X, vocab


def load_corpus_metadata(features_dir: Path) -> pd.DataFrame:
    """Load corpus_features.parquet (book_id, chunk_id, label_suffering_type, etc.)."""
    path = features_dir / "corpus_features.parquet"
    if not path.exists():
        path = PROJECT_ROOT / "data" / "processed" / "corpus.parquet"
    if not path.exists():
        raise FileNotFoundError("Missing corpus.parquet or corpus_features.parquet")
    return pd.read_parquet(path)


def select_k_elbow_silhouette(
    X: np.ndarray,
    k_range: tuple[int, int] = K_RANGE,
    results_dir: Path | None = None,
) -> int:
    """
    Compute inertia and silhouette for k in k_range; return k with best silhouette.
    Optionally save a small summary to results_dir.
    """
    k_min, k_max = k_range
    inertias = []
    silhouettes = []
    ks = list(range(k_min, k_max))

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        if k > 1:
            silhouettes.append(silhouette_score(X, labels))
        else:
            silhouettes.append(0.0)

    best_idx = np.argmax(silhouettes)
    best_k = ks[best_idx]
    print(f"K selection: best silhouette at k={best_k} (tried {k_min}..{k_max-1})")

    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "k": ks,
            "inertia": inertias,
            "silhouette": silhouettes,
        }).to_csv(results_dir / "k_selection.csv", index=False)

    return best_k


def run_kmeans(X: np.ndarray, k: int = DEFAULT_K, random_state: int = 42) -> np.ndarray:
    """Fit K-Means and return cluster labels (0 .. k-1)."""
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return km.fit_predict(X)


def run_hierarchical(
    X: np.ndarray,
    k: int = DEFAULT_K,
    method: str = LINKAGE_METHOD,
) -> np.ndarray:
    """Run hierarchical clustering; cut to k clusters. Returns labels 0 .. k-1."""
    # Ward expects Euclidean distances; use condensed distance matrix
    cond = pdist(X, metric="euclidean")
    Z = linkage(cond, method=method)
    # fcluster uses 1-based labels; convert to 0-based
    labels = fcluster(Z, k, criterion="maxclust").astype(int) - 1
    return labels


def top_tfidf_terms_per_cluster(
    tfidf_matrix: np.ndarray,
    vocab: list[str],
    labels: np.ndarray,
    top_n: int = TOP_TERMS_PER_CLUSTER,
) -> pd.DataFrame:
    """For each cluster, get top_n terms by mean TF-IDF in that cluster."""
    clusters = sorted(set(labels))
    rows = []
    for c in clusters:
        mask = labels == c
        if mask.sum() == 0:
            continue
        means = tfidf_matrix[mask].mean(axis=0)
        top_idx = np.argsort(-means)[:top_n]
        top_terms = [vocab[i] for i in top_idx if i < len(vocab)]
        rows.append({"cluster": c, "top_terms": " | ".join(top_terms)})
    return pd.DataFrame(rows)


def cross_tab_cluster_book(
    labels: np.ndarray,
    corpus: pd.DataFrame,
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-tabulate cluster vs book_id and cluster vs label_suffering_type."""
    n = len(labels)
    if len(corpus) != n:
        raise ValueError(f"Labels length {n} != corpus length {len(corpus)}")
    df = corpus.copy()
    df["cluster"] = labels

    ct_book = pd.crosstab(df["cluster"], df["book_id"])
    ct_book.to_csv(results_dir / "cluster_vs_book.csv")

    if "label_suffering_type" in df.columns:
        ct_label = pd.crosstab(df["cluster"], df["label_suffering_type"])
        ct_label.to_csv(results_dir / "cluster_vs_label.csv")
    else:
        ct_label = pd.DataFrame()

    return ct_book, ct_label


def run(
    features_dir: Path | None = None,
    results_dir: Path | None = None,
    k: int | None = None,
    use_embeddings: bool = True,
    save_dendrogram: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load features, select k (or use provided k), run K-Means and hierarchical,
    write cluster_descriptions.csv, cluster_vs_book.csv, cluster_vs_label.csv,
    and optional dendrogram. Returns (kmeans_labels, hierarchical_labels).
    """
    features_dir = features_dir or DEFAULT_FEATURES_DIR
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    if use_embeddings:
        X = load_embedding_matrix(features_dir)
    else:
        X, _ = load_tfidf_and_vocab(features_dir)

    tfidf_matrix, vocab = load_tfidf_and_vocab(features_dir)
    corpus = load_corpus_metadata(features_dir)
    n = len(corpus)
    if len(X) != n or len(tfidf_matrix) != n:
        raise ValueError("Row count mismatch: corpus, embeddings, and TF-IDF must have same length.")

    if k is None:
        k = select_k_elbow_silhouette(X, K_RANGE, results_dir)
    print(f"Clustering with k={k}")

    # 3.1 K-Means
    labels_kmeans = run_kmeans(X, k=k)
    pd.DataFrame({"chunk_index": range(n), "cluster_kmeans": labels_kmeans}).to_csv(
        results_dir / "labels_kmeans.csv", index=False
    )

    # 3.2 Hierarchical
    labels_hier = run_hierarchical(X, k=k)
    pd.DataFrame({"chunk_index": range(n), "cluster_hierarchical": labels_hier}).to_csv(
        results_dir / "labels_hierarchical.csv", index=False
    )

    if save_dendrogram:
        from matplotlib import pyplot as plt
        cond = pdist(X, metric="euclidean")
        Z = linkage(cond, method=LINKAGE_METHOD)
        plt.figure(figsize=(10, 5))
        dendrogram(Z, truncate_mode="lastp", p=min(50, n))
        plt.title("Hierarchical clustering (Ward) — truncated dendrogram")
        plt.savefig(results_dir / "dendrogram.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {results_dir / 'dendrogram.png'}")

    # 3.3 Cluster descriptions (top TF-IDF terms per cluster)
    desc_kmeans = top_tfidf_terms_per_cluster(tfidf_matrix, vocab, labels_kmeans, TOP_TERMS_PER_CLUSTER)
    desc_kmeans.to_csv(results_dir / "cluster_descriptions.csv", index=False)
    print(f"Saved {results_dir / 'cluster_descriptions.csv'}")

    # 3.4 Cross-tabs
    ct_book, ct_label = cross_tab_cluster_book(labels_kmeans, corpus, results_dir)
    print(f"Saved {results_dir / 'cluster_vs_book.csv'}")
    if not ct_label.empty:
        print(f"Saved {results_dir / 'cluster_vs_label.csv'}")

    # Brief interpretation note
    with open(results_dir / "notes.md", "a", encoding="utf-8") as f:
        f.write("\n## Phase 3: Clustering\n\n")
        f.write(f"K-Means and hierarchical (Ward) with k={k} on chunk embeddings. ")
        f.write("Cluster descriptions use top TF-IDF terms per cluster. ")
        f.write("Check cluster_vs_book.csv and cluster_vs_label.csv to see whether ")
        f.write("clusters align more with suffering type (label) or with book/author.\n\n")

    return labels_kmeans, labels_hier


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Phase 3: clustering taxonomy")
    p.add_argument("--features-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--k", type=int, default=None, help="Fixed k; if not set, choose by silhouette")
    p.add_argument("--tfidf", action="store_true", help="Cluster on TF-IDF instead of embeddings")
    p.add_argument("--no-dendrogram", action="store_true", help="Skip saving dendrogram")
    args = p.parse_args()
    run(
        features_dir=args.features_dir,
        results_dir=args.results_dir,
        k=args.k,
        use_embeddings=not args.tfidf,
        save_dendrogram=not args.no_dendrogram,
    )
