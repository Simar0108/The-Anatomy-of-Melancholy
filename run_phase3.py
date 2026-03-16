#!/usr/bin/env python3
"""
Phase 3 pipeline: suffering taxonomy — K-Means + hierarchical clustering,
cluster descriptions (top TF-IDF terms), and cross-tabs (cluster vs book / label).

Run from project root with venv activated:
  source .venv/bin/activate   # or: venv\Scripts\activate on Windows
  python run_phase3.py

Requires: data/features/ from Phase 2 (embeddings.npy, tfidf_matrix.npy, tfidf_vocab.json,
          corpus_features.parquet).

Outputs (in results/):
  k_selection.csv, labels_kmeans.csv, labels_hierarchical.csv,
  cluster_descriptions.csv, cluster_vs_book.csv, cluster_vs_label.csv,
  dendrogram.png, and a short note in notes.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
RESULTS_DIR = PROJECT_ROOT / "results"


def main() -> None:
    if not (FEATURES_DIR / "embeddings.npy").exists():
        print(f"Missing {FEATURES_DIR / 'embeddings.npy'}. Run: python run_phase2.py")
        sys.exit(1)

    from src.clustering.kmeans_taxonomy import run
    run(features_dir=FEATURES_DIR, results_dir=RESULTS_DIR)
    print("Phase 3 done. See results/ for cluster tables and dendrogram.")


if __name__ == "__main__":
    main()
