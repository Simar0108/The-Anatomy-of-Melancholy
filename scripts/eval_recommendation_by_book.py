#!/usr/bin/env python3
"""
Evaluate recommender by "expected book": for each query we define one expected book (e.g. Myth of Sisyphus).
P@5 = fraction of top-5 results that come from the expected book (no human relevance labels needed).

Input: data/eval_queries_by_book.csv with query_id, query_text, expected_book_id.
Output: results/eval_recommendation_by_book.json and print summary.

Run from project root:
  python scripts/eval_recommendation_by_book.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    import pandas as pd
    from src.recommendation.query import recommend

    queries_path = PROJECT_ROOT / "data" / "eval_queries_by_book.csv"
    features_dir = PROJECT_ROOT / "data" / "features"
    results_dir = PROJECT_ROOT / "results"

    if not queries_path.exists():
        print(f"Create {queries_path} with columns: query_id, query_text, expected_book_id")
        sys.exit(1)
    if not (features_dir / "embeddings.npy").exists():
        print("Missing embeddings. Run Phase 2 first.")
        sys.exit(1)

    queries = pd.read_csv(queries_path)
    if "query_text" not in queries.columns or "expected_book_id" not in queries.columns:
        print("Need columns: query_text, expected_book_id")
        sys.exit(1)

    k = 5
    precisions = []
    recall_book = []  # 1 if at least one of top-k from expected book
    for _, row in queries.iterrows():
        qtext = str(row["query_text"]).strip()
        if not qtext or qtext.startswith("#"):
            continue
        expected_book = str(row["expected_book_id"]).strip()
        rec = recommend(qtext, k=k, features_dir=features_dir)
        recommended_books = rec["book_id"].astype(str).tolist()
        hits = sum(1 for b in recommended_books if b == expected_book)
        precisions.append(hits / k)
        recall_book.append(1 if hits > 0 else 0)

    if not precisions:
        print("No valid query rows.")
        sys.exit(0)

    p_at_k = sum(precisions) / len(precisions)
    recall_at_book = sum(recall_book) / len(recall_book)  # fraction of queries with ≥1 from expected book
    n = len(precisions)
    print(f"Queries: {n}, P@5 (from expected book): {p_at_k:.4f}, Recall@book (≥1 in top-5): {recall_at_book:.4f}")
    results_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "n_queries": n,
        "P@5_from_expected_book": p_at_k,
        "Recall_at_least_one_from_expected_book": recall_at_book,
        "k": k,
    }
    out_path = results_dir / "eval_recommendation_by_book.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
