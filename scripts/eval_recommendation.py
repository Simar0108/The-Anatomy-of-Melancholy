#!/usr/bin/env python3
"""
Evaluate chunk recommender: precision@k and optional recall@k using a relevance set.
Relevance set: data/relevance_set.csv with columns query_id, query_text, relevant_chunk_indices
(relevant_chunk_indices = comma-separated corpus row indices that are relevant for that query).

Run from project root:
  python scripts/eval_recommendation.py
  python scripts/eval_recommendation.py --relevance-set data/relevance_set.csv -k 5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Evaluate recommender with relevance set")
    p.add_argument("--relevance-set", type=Path, default=PROJECT_ROOT / "data" / "relevance_set.csv")
    p.add_argument("--features-dir", type=Path, default=PROJECT_ROOT / "data" / "features")
    p.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    p.add_argument("-k", type=int, default=5, help="Top-k for P@k")
    args = p.parse_args()

    import pandas as pd
    from src.recommendation.query import recommend

    if not args.relevance_set.exists():
        print(f"Relevance set not found: {args.relevance_set}")
        print("Create a CSV with columns: query_id, query_text, relevant_chunk_indices")
        print("(relevant_chunk_indices = comma-separated corpus row indices)")
        sys.exit(1)
    if not (args.features_dir / "embeddings.npy").exists():
        print(f"Missing embeddings. Run: python run_phase2.py")
        sys.exit(1)

    rel = pd.read_csv(args.relevance_set)
    if "query_text" not in rel.columns or "relevant_chunk_indices" not in rel.columns:
        print("Relevance set must have columns: query_id, query_text, relevant_chunk_indices")
        sys.exit(1)

    k = args.k
    precisions = []
    recalls = []
    for _, row in rel.iterrows():
        qtext = str(row["query_text"]).strip()
        if not qtext or qtext.startswith("#"):
            continue
        relevant = set()
        for part in str(row["relevant_chunk_indices"]).strip().split(","):
            part = part.strip()
            if part.isdigit():
                relevant.add(int(part))
        if not relevant:
            continue
        rec = recommend(qtext, k=k, features_dir=args.features_dir)
        recommended = set(rec["chunk_index"].tolist())
        hits = len(recommended & relevant)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant) if relevant else 0.0)

    if not precisions:
        print("No valid query rows in relevance set. Add query_text and relevant_chunk_indices.")
        sys.exit(0)

    p_at_k = sum(precisions) / len(precisions)
    r_at_k = sum(recalls) / len(recalls) if recalls else 0.0
    n = len(precisions)
    print(f"Queries: {n}, P@{k}: {p_at_k:.4f}, R@{k}: {r_at_k:.4f}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = {"n_queries": n, f"P@{k}": p_at_k, f"R@{k}": r_at_k, "k": k}
    out_path = args.results_dir / "eval_recommendation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
