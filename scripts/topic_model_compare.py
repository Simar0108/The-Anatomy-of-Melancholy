#!/usr/bin/env python3
"""
Topic model (NMF on TF-IDF) as a second view of the corpus; compare to K-means embedding clusters.
Outputs: results/topic_top_terms.csv, results/topic_vs_book.csv, results/topic_vs_cluster.csv,
         results/topic_model_comparison.md.

Run from project root (after Phase 2 and Phase 3):
  python scripts/topic_model_compare.py
  python scripts/topic_model_compare.py --n-topics 5
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    import argparse
    import json
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import NMF

    p = argparse.ArgumentParser(description="NMF topic model on TF-IDF; compare to K-means")
    p.add_argument("--features-dir", type=Path, default=PROJECT_ROOT / "data" / "features")
    p.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    p.add_argument("--n-topics", type=int, default=3, help="Number of NMF topics (default 3 to match k=3)")
    args = p.parse_args()

    features_dir = args.features_dir
    results_dir = args.results_dir

    # Load TF-IDF and vocab
    mat_path = features_dir / "tfidf_matrix.npy"
    vocab_path = features_dir / "tfidf_vocab.json"
    if not mat_path.exists() or not vocab_path.exists():
        print("Run Phase 2 first (tfidf_matrix.npy, tfidf_vocab.json).")
        sys.exit(1)
    X = np.load(mat_path).astype(np.float32)
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    # Avoid zeros for NMF
    X = np.maximum(X, 1e-8)

    n_topics = args.n_topics
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=300)
    W = nmf.fit_transform(X)  # (n_chunks, n_topics)
    H = nmf.components_      # (n_topics, n_vocab)
    topic_assign = np.argmax(W, axis=1)  # per-chunk topic

    # Top terms per topic
    top_n = 15
    topic_terms = []
    for t in range(n_topics):
        order = np.argsort(H[t])[::-1][:top_n]
        terms = [vocab[i] for i in order]
        for r, term in enumerate(terms):
            topic_terms.append({"topic": t, "rank": r + 1, "term": term, "weight": float(H[t, order[r]])})
    terms_df = pd.DataFrame(topic_terms)
    results_dir.mkdir(parents=True, exist_ok=True)
    terms_df.to_csv(results_dir / "topic_top_terms.csv", index=False)
    print(f"Topic top terms -> {results_dir / 'topic_top_terms.csv'}")

    # Corpus metadata (book_id, label) for cross-tabs
    corpus_path = features_dir / "corpus_features.parquet"
    if not corpus_path.exists():
        corpus_path = PROJECT_ROOT / "data" / "processed" / "corpus.parquet"
    if not corpus_path.exists():
        print("Missing corpus. Run Phase 1–2.")
        sys.exit(1)
    corpus = pd.read_parquet(corpus_path)
    if len(corpus) != len(topic_assign):
        print("Corpus length != TF-IDF rows; check Phase 2.")
        sys.exit(1)
    corpus = corpus.copy()
    corpus["topic_nmf"] = topic_assign

    # Topic vs book
    topic_vs_book = pd.crosstab(corpus["topic_nmf"], corpus["book_id"])
    topic_vs_book.to_csv(results_dir / "topic_vs_book.csv")
    print(f"Topic vs book -> {results_dir / 'topic_vs_book.csv'}")

    # Topic vs K-means cluster (if available)
    labels_path = results_dir / "labels_kmeans.csv"
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        if len(labels_df) == len(topic_assign):
            corpus["cluster_kmeans"] = labels_df["cluster_kmeans"].values
            topic_vs_cluster = pd.crosstab(corpus["topic_nmf"], corpus["cluster_kmeans"])
            topic_vs_cluster.to_csv(results_dir / "topic_vs_cluster.csv")
            print(f"Topic vs cluster -> {results_dir / 'topic_vs_cluster.csv'}")

    # Short comparison note
    md = [
        "# Topic model (NMF) vs K-means embedding clusters",
        "",
        f"NMF with **{n_topics} topics** on TF-IDF (same chunks as embedding pipeline).",
        "",
        "## Top terms per topic",
        "",
    ]
    for t in range(n_topics):
        row_terms = terms_df[terms_df["topic"] == t]["term"].tolist()[:12]
        md.append(f"- **Topic {t}:** " + ", ".join(row_terms))
    md.extend([
        "",
        "## Interpretation",
        "",
        "Compare with K-means cluster descriptions (results/cluster_descriptions.csv). "
        "If NMF topics align with embedding clusters (e.g. archaic vs narrative), that reinforces "
        "that the corpus separates by **style/lexical register** rather than by suffering type. "
        "See topic_vs_book.csv and topic_vs_cluster.csv for cross-tabs.",
        "",
    ])
    with open(results_dir / "topic_model_comparison.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Wrote {results_dir / 'topic_model_comparison.md'}")


if __name__ == "__main__":
    main()
