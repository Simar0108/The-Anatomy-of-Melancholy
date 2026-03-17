#!/usr/bin/env python3
"""
Compute how well K-means clusters align with book-level "suffering type" labels.
Outputs NMI and ARI to results/cluster_label_agreement.json (and .txt).
No training — just a consistency check between unsupervised clusters and our labels.

Run from project root (after Phase 2 and 3):
  python scripts/cluster_label_agreement.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    features_dir = PROJECT_ROOT / "data" / "features"
    results_dir = PROJECT_ROOT / "results"
    labels_path = results_dir / "labels_kmeans.csv"
    corpus_path = features_dir / "corpus_features.parquet"

    if not labels_path.exists():
        print(f"Missing {labels_path}. Run Phase 3 first.")
        sys.exit(1)
    if not corpus_path.exists():
        print(f"Missing {corpus_path}. Run Phase 2 first.")
        sys.exit(1)

    labels_df = pd.read_csv(labels_path)
    corpus = pd.read_parquet(corpus_path)
    if "label_suffering_type" not in corpus.columns:
        print("corpus_features has no label_suffering_type.")
        sys.exit(1)

    # Align by chunk index (row position)
    n = len(corpus)
    if len(labels_df) != n:
        print(f"Row count mismatch: labels {len(labels_df)} vs corpus {n}")
        sys.exit(1)

    cluster = labels_df["cluster_kmeans"].values
    label = corpus["label_suffering_type"].astype(str).values

    nmi = normalized_mutual_info_score(cluster, label)
    ari = adjusted_rand_score(cluster, label)

    out = {"NMI": float(nmi), "ARI": float(ari), "n_chunks": n}
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "cluster_label_agreement.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(results_dir / "cluster_label_agreement.txt", "w", encoding="utf-8") as f:
        f.write(f"NMI(cluster, book_label) = {nmi:.4f}\nARI(cluster, book_label) = {ari:.4f}\n")
    print(f"NMI = {nmi:.4f}, ARI = {ari:.4f} -> results/cluster_label_agreement.json")


if __name__ == "__main__":
    main()
