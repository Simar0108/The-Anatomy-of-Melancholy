#!/usr/bin/env python3
"""
Cluster-based book recommendation: recommend whole books (from a candidate list,
e.g. Project Gutenberg) that match each K-means cluster's philosophical style.
Use this to find "books like this cluster" and to discover new books to add to the study.

Usage (from project root with venv activated):
  python recommend_books.py
  python recommend_books.py --candidates data/candidates_gutenberg.csv --top 5
  python recommend_books.py --cluster 0   # only show recommendations for cluster 0

Requires: Phase 2 (features) and Phase 3 (labels_kmeans.csv) must be run first.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="Recommend books per cluster (by similarity to cluster centroid)."
    )
    p.add_argument("--candidates", type=Path,
                   default=PROJECT_ROOT / "data" / "candidates_gutenberg.csv",
                   help="CSV with gutenberg_id, title [, subject]")
    p.add_argument("--top", type=int, default=5, help="Top N books per cluster")
    p.add_argument("--cluster", type=int, default=None, metavar="K",
                   help="Only show recommendations for cluster K (default: all)")
    p.add_argument("--no-fetch", action="store_true",
                   help="Do not fetch from Gutenberg; candidates must have sample_text column")
    args = p.parse_args()

    features_dir = PROJECT_ROOT / "data" / "features"
    results_dir = PROJECT_ROOT / "results"
    if not (features_dir / "embeddings.npy").exists():
        print("Missing embeddings. Run: python run_phase2.py", file=sys.stderr)
        sys.exit(1)
    if not (results_dir / "labels_kmeans.csv").exists():
        print("Missing cluster labels. Run: python run_phase3.py", file=sys.stderr)
        sys.exit(1)
    if not args.candidates.exists():
        print(f"Missing candidates file: {args.candidates}", file=sys.stderr)
        sys.exit(1)

    import pandas as pd
    from src.recommendation.cluster_books import recommend_books_for_clusters

    candidates = pd.read_csv(args.candidates)
    if "title" not in candidates.columns:
        candidates["title"] = "Gutenberg " + candidates["gutenberg_id"].astype(str)

    print("Computing cluster centroids and candidate book embeddings...")
    if not args.no_fetch:
        print("(Fetching text samples from Gutenberg for each candidate; this may take a minute.)")
    result = recommend_books_for_clusters(
        candidates,
        features_dir=features_dir,
        results_dir=results_dir,
        top_per_cluster=args.top,
        fetch_sample=not args.no_fetch,
    )

    if args.cluster is not None:
        result = result[result["cluster"] == args.cluster]

    print()
    for cluster in result["cluster"].unique():
        subset = result[result["cluster"] == cluster].sort_values("rank_in_cluster")
        desc = subset["cluster_desc"].iloc[0]
        print(f"--- Cluster {cluster}: {desc} ---")
        for _, row in subset.iterrows():
            print(f"  {row['rank_in_cluster']}. [{row['score']:.3f}] {row['title']} (Gutenberg {row['gutenberg_id']})")
        print()
    print("Done. Use these as 'books like this cluster' or as candidates to add to the study (run_phase1 corpus).")


if __name__ == "__main__":
    main()
