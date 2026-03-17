#!/usr/bin/env python3
"""
Reranker ablation: run expected-book eval with bi-encoder only vs bi-encoder + cross-encoder rerank.
Output: results/eval_reranker_ablation.json and results/eval_reranker_ablation.md (comparison table).

Run from project root:
  python scripts/eval_reranker_ablation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_eval(rerank: bool, features_dir: Path, queries_path: Path, k: int = 5) -> tuple[float, float, int]:
    import pandas as pd
    from src.recommendation.query import recommend

    queries = pd.read_csv(queries_path)
    precisions = []
    recall_book = []
    for _, row in queries.iterrows():
        qtext = str(row["query_text"]).strip()
        if not qtext or qtext.startswith("#"):
            continue
        expected_book = str(row["expected_book_id"]).strip()
        rec = recommend(qtext, k=k, features_dir=features_dir, rerank=rerank, rerank_top_k=20)
        recommended_books = rec["book_id"].astype(str).tolist()
        hits = sum(1 for b in recommended_books if b == expected_book)
        precisions.append(hits / k)
        recall_book.append(1 if hits > 0 else 0)
    if not precisions:
        return 0.0, 0.0, 0
    p_at_k = sum(precisions) / len(precisions)
    recall_at_book = sum(recall_book) / len(recall_book)
    return p_at_k, recall_at_book, len(precisions)


def main() -> None:
    queries_path = PROJECT_ROOT / "data" / "eval_queries_by_book.csv"
    features_dir = PROJECT_ROOT / "data" / "features"
    results_dir = PROJECT_ROOT / "results"

    if not queries_path.exists():
        print(f"Create {queries_path} with columns: query_id, query_text, expected_book_id")
        sys.exit(1)
    if not (features_dir / "embeddings.npy").exists():
        print("Missing embeddings. Run Phase 2 first.")
        sys.exit(1)

    k = 5
    print("Running eval: bi-encoder only...")
    p_bi, r_bi, n_q = run_eval(False, features_dir, queries_path, k)
    print("Running eval: bi-encoder + cross-encoder rerank...")
    p_rerank, r_rerank, _ = run_eval(True, features_dir, queries_path, k)

    out = {
        "n_queries": n_q,
        "k": k,
        "bi_encoder_only": {"P@5_from_expected_book": p_bi, "Recall_at_least_one_from_expected_book": r_bi},
        "bi_encoder_plus_rerank": {"P@5_from_expected_book": p_rerank, "Recall_at_least_one_from_expected_book": r_rerank},
        "delta_rerank": {"P@5": p_rerank - p_bi, "Recall@book": r_rerank - r_bi},
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "eval_reranker_ablation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    md = [
        "# Reranker ablation",
        "",
        "Expected-book eval: P@5 = fraction of top-5 from expected book; Recall@book = fraction of queries with ≥1 hit.",
        "",
        "| Config | P@5 (from expected book) | Recall@book (≥1 in top-5) |",
        "|--------|--------------------------|----------------------------|",
        f"| Bi-encoder only | {p_bi:.4f} | {r_bi:.4f} |",
        f"| Bi-encoder + cross-encoder rerank | {p_rerank:.4f} | {r_rerank:.4f} |",
        "",
        f"**Delta (rerank − bi-encoder):** P@5 = {out['delta_rerank']['P@5']:+.4f}, Recall@book = {out['delta_rerank']['Recall@book']:+.4f}",
    ]
    with open(results_dir / "eval_reranker_ablation.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Bi-encoder only:     P@5 = {p_bi:.4f}, Recall@book = {r_bi:.4f}")
    print(f"Bi-encoder + rerank: P@5 = {p_rerank:.4f}, Recall@book = {r_rerank:.4f}")
    print(f"Wrote {results_dir / 'eval_reranker_ablation.json'} and .md")


if __name__ == "__main__":
    main()
