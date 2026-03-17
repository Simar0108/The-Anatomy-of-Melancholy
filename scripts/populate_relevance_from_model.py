#!/usr/bin/env python3
"""
Populate data/relevance_set.csv with the recommender's top-k chunk indices for each query.
This gives a non-zero P@k when running scripts/eval_recommendation.py (sanity check).
For real evaluation, replace relevant_chunk_indices with human-judged corpus row indices.

Run from project root (after Phase 2 + FAISS):
  python scripts/populate_relevance_from_model.py
  python scripts/populate_relevance_from_model.py -k 5
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Set relevance_set relevant_chunk_indices from model top-k")
    p.add_argument("--relevance-set", type=Path, default=PROJECT_ROOT / "data" / "relevance_set.csv")
    p.add_argument("--features-dir", type=Path, default=PROJECT_ROOT / "data" / "features")
    p.add_argument("-k", type=int, default=5, help="Number of top indices to store per query")
    args = p.parse_args()

    import pandas as pd
    from src.recommendation.query import recommend

    if not args.relevance_set.exists():
        print(f"Not found: {args.relevance_set}. Create it with query_id, query_text, relevant_chunk_indices.")
        sys.exit(1)
    if not (args.features_dir / "embeddings.npy").exists():
        print("Missing embeddings. Run Phase 2 first.")
        sys.exit(1)

    rel = pd.read_csv(args.relevance_set)
    if "query_text" not in rel.columns:
        print("relevance_set must have column query_text")
        sys.exit(1)

    k = args.k
    updated = []
    for _, row in rel.iterrows():
        qtext = str(row["query_text"]).strip()
        if not qtext or qtext.startswith("#"):
            updated.append(row)
            continue
        rec = recommend(qtext, k=k, features_dir=args.features_dir)
        indices = rec["chunk_index"].tolist()
        idx_str = ",".join(str(i) for i in indices)
        new_row = row.to_dict()
        new_row["relevant_chunk_indices"] = idx_str
        updated.append(new_row)

    out_df = pd.DataFrame(updated)
    out_df.to_csv(args.relevance_set, index=False)
    print(f"Updated {args.relevance_set} with top-{k} chunk indices from recommender.")
    print("Run: python scripts/eval_recommendation.py -k {k}  (expect P@{k} = 1.0 for this sanity check).".format(k=k))


if __name__ == "__main__":
    main()
