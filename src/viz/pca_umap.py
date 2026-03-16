"""
Phase 4: Dimensionality reduction (PCA, UMAP) and lexical anchors.
PCA on TF-IDF for interpretable loadings; UMAP on embeddings for visualization.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "data" / "features"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

N_PCA_COMPONENTS = 3
N_ANCHOR_WORDS = 15
UMAP_2D_N_NEIGHBORS = 15
UMAP_2D_MIN_DIST = 0.1
UMAP_3D_N_NEIGHBORS = 15
UMAP_3D_MIN_DIST = 0.1


def _load_embeddings(features_dir: Path) -> np.ndarray:
    path = features_dir / "embeddings.npy"
    if not path.exists():
        raise FileNotFoundError(f"Run Phase 2 first: missing {path}")
    return np.load(path).astype(np.float32)


def _load_tfidf_vocab(features_dir: Path) -> tuple[np.ndarray, list[str]]:
    mat_path = features_dir / "tfidf_matrix.npy"
    vocab_path = features_dir / "tfidf_vocab.json"
    if not mat_path.exists() or not vocab_path.exists():
        raise FileNotFoundError(f"Run Phase 2 first: missing {mat_path} or {vocab_path}")
    X = np.load(mat_path).astype(np.float32)
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    return X, vocab


def _load_corpus(features_dir: Path) -> pd.DataFrame:
    path = features_dir / "corpus_features.parquet"
    if not path.exists():
        path = PROJECT_ROOT / "data" / "processed" / "corpus.parquet"
    if not path.exists():
        raise FileNotFoundError("Missing corpus.parquet or corpus_features.parquet")
    return pd.read_parquet(path)


def _load_cluster_labels(results_dir: Path) -> np.ndarray:
    path = results_dir / "labels_kmeans.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run Phase 3 first: missing {path}")
    df = pd.read_csv(path)
    return df["cluster_kmeans"].values


def run_pca(
    features_dir: Path,
    results_dir: Path,
    n_components: int = N_PCA_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fit PCA on TF-IDF; save projected coords and loadings. Returns (coords, loadings_matrix, vocab)."""
    X, vocab = _load_tfidf_vocab(features_dir)
    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X)  # (n_chunks, n_components)
    loadings = pca.components_.T  # (n_features, n_components)

    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / "pca_coords.npy", coords.astype(np.float32))
    np.save(results_dir / "pca_loadings.npy", loadings.astype(np.float32))
    pd.DataFrame({"explained_variance_ratio": pca.explained_variance_ratio_}).to_csv(
        results_dir / "pca_variance.csv", index=False
    )
    print(f"PCA: {n_components} components, coords {coords.shape}, loadings {loadings.shape} -> {results_dir}")
    return coords, loadings, vocab


def run_umap(
    features_dir: Path,
    results_dir: Path,
    n_components_2d: int = 2,
    n_components_3d: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Run UMAP 2D and 3D on embeddings; save coordinates. Returns (coords_2d, coords_3d)."""
    import umap
    X = _load_embeddings(features_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    reducer_2d = umap.UMAP(n_components=n_components_2d, n_neighbors=UMAP_2D_N_NEIGHBORS, min_dist=UMAP_2D_MIN_DIST, random_state=42, metric="cosine")
    coords_2d = reducer_2d.fit_transform(X)
    np.save(results_dir / "umap_2d.npy", coords_2d.astype(np.float32))

    reducer_3d = umap.UMAP(n_components=n_components_3d, n_neighbors=UMAP_3D_N_NEIGHBORS, min_dist=UMAP_3D_MIN_DIST, random_state=43, metric="cosine")
    coords_3d = reducer_3d.fit_transform(X)
    np.save(results_dir / "umap_3d.npy", coords_3d.astype(np.float32))

    print(f"UMAP: 2D {coords_2d.shape}, 3D {coords_3d.shape} -> {results_dir}")
    return coords_2d, coords_3d


def anchor_words_from_pca(
    loadings: np.ndarray,
    vocab: list[str],
    results_dir: Path,
    top_n: int = N_ANCHOR_WORDS,
) -> pd.DataFrame:
    """Top absolute loadings per PCA component -> anchor_words.csv."""
    n_components = loadings.shape[1]
    rows = []
    for c in range(n_components):
        col = loadings[:, c]
        idx = np.argsort(-np.abs(col))[:top_n]
        for r, i in enumerate(idx):
            if i < len(vocab):
                rows.append({"component": c, "rank": r + 1, "term": vocab[i], "loading": float(col[i])})
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "anchor_words.csv", index=False)
    print(f"Anchor words: {len(df)} rows -> {results_dir / 'anchor_words.csv'}")
    return df


def plot_umap_by_cluster_and_book(
    results_dir: Path,
    features_dir: Path,
) -> None:
    """Scatter UMAP 2D colored by cluster and by book; save figures."""
    import matplotlib.pyplot as plt
    coords_2d = np.load(results_dir / "umap_2d.npy")
    corpus = _load_corpus(features_dir)
    labels = _load_cluster_labels(results_dir)
    n = len(corpus)
    if len(coords_2d) != n or len(labels) != n:
        raise ValueError("Row count mismatch for plotting.")
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=labels, cmap="tab10", alpha=0.7, s=20)
    ax.set_title("UMAP 2D by cluster (K-Means)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    plt.savefig(results_dir / "umap_by_cluster.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {results_dir / 'umap_by_cluster.png'}")

    book_id = corpus["book_id"].astype(str)
    books = book_id.unique().tolist()
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, b in enumerate(books):
        mask = book_id == b
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], label=b, alpha=0.7, s=20, color=plt.cm.tab10(i % 10))
    ax.set_title("UMAP 2D by book")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(results_dir / "umap_by_book.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {results_dir / 'umap_by_book.png'}")


def run(
    features_dir: Path | None = None,
    results_dir: Path | None = None,
    skip_umap: bool = False,
    skip_plots: bool = False,
) -> None:
    """Run PCA, UMAP, anchor words, and plots. Optionally skip UMAP or plots."""
    features_dir = features_dir or DEFAULT_FEATURES_DIR
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    coords_pca, loadings_pca, vocab = run_pca(features_dir, results_dir, N_PCA_COMPONENTS)
    anchor_words_from_pca(loadings_pca, vocab, results_dir, N_ANCHOR_WORDS)

    if not skip_umap:
        run_umap(features_dir, results_dir)
    if not skip_plots:
        plot_umap_by_cluster_and_book(results_dir, features_dir)

    with open(results_dir / "notes.md", "a", encoding="utf-8") as f:
        f.write("\n## Phase 4: Dimensionality reduction & anchors\n\n")
        f.write("PCA on TF-IDF (3 components) and UMAP 2D/3D on embeddings. ")
        f.write("Anchor words = top |loading| per PCA component (anchor_words.csv). ")
        f.write("Hypothesis 2: if anchors align with intuitive suffering types (e.g. moral, existential), ")
        f.write("lexical separation supports the taxonomy; compare with cluster_descriptions.csv and cross-tabs.\n\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Phase 4: PCA, UMAP, anchor words, plots")
    p.add_argument("--features-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--skip-umap", action="store_true", help="Only run PCA and anchors")
    p.add_argument("--skip-plots", action="store_true", help="Do not generate scatter plots")
    args = p.parse_args()
    run(
        features_dir=args.features_dir,
        results_dir=args.results_dir,
        skip_umap=args.skip_umap,
        skip_plots=args.skip_plots,
    )
