#!/usr/bin/env python3
"""
Phase 4 pipeline: PCA, UMAP, anchor words, and scatter plots (by cluster / by book).

Run from project root with venv activated:
  python run_phase4.py

Requires: data/features/ (Phase 2) and results/labels_kmeans.csv (Phase 3).

Outputs in results/:
  pca_coords.npy, pca_loadings.npy, pca_variance.csv,
  umap_2d.npy, umap_3d.npy,
  anchor_words.csv, umap_by_cluster.png, umap_by_book.png,
  and an appended note in notes.md.
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
    if not (RESULTS_DIR / "labels_kmeans.csv").exists():
        print(f"Missing {RESULTS_DIR / 'labels_kmeans.csv'}. Run: python run_phase3.py")
        sys.exit(1)

    from src.viz.pca_umap import run
    run(features_dir=FEATURES_DIR, results_dir=RESULTS_DIR)
    print("Phase 4 done. See results/ for PCA/UMAP outputs and plots.")


if __name__ == "__main__":
    main()
