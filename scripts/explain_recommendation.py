#!/usr/bin/env python3
"""
Interpretability: for each of a few queries, show why we recommended the top result(s).
- Word overlap: which query words appear in the recommended chunk (and in the cluster's top terms).
- Cluster: which K-means cluster the chunk belongs to and that cluster's top terms.

Output: results/explain_recommendation.md

Run from project root:
  python scripts/explain_recommendation.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _tokenize(text: str) -> set[str]:
    """Lowercase, keep only letters, split."""
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    return {w for w in text.split() if len(w) > 1}


def main() -> None:
    import pandas as pd
    from src.recommendation.query import recommend

    features_dir = PROJECT_ROOT / "data" / "features"
    results_dir = PROJECT_ROOT / "results"

    if not (features_dir / "embeddings.npy").exists():
        print("Missing embeddings. Run Phase 2 first.")
        sys.exit(1)

    labels_path = results_dir / "labels_kmeans.csv"
    desc_path = results_dir / "cluster_descriptions.csv"
    if not labels_path.exists() or not desc_path.exists():
        print("Run Phase 3 first (labels_kmeans.csv, cluster_descriptions.csv).")
        sys.exit(1)

    labels_df = pd.read_csv(labels_path)
    desc_df = pd.read_csv(desc_path)
    cluster_terms = {}
    for _, row in desc_df.iterrows():
        c = int(row["cluster"])
        cluster_terms[c] = (row["top_terms"] if "top_terms" in row else str(row.get("terms", ""))).replace(" | ", " ").split()

    # chunk_index -> cluster
    chunk_to_cluster = dict(zip(labels_df["chunk_index"], labels_df["cluster_kmeans"]))

    queries = [
        "I am tormented by the idea of meaninglessness",
        "Accept what is in your control and let go of the rest",
        "We must imagine Sisyphus happy",
    ]
    top_k = 3
    lines = [
        "# Why we recommended this",
        "",
        "For each query we show the top recommended passage, **word overlap** (query words that appear in the chunk or in the cluster's defining terms), and the **cluster** (K-means) with its top terms.",
        "",
    ]

    for q in queries:
        rec = recommend(q, k=top_k, features_dir=features_dir)
        q_tokens = _tokenize(q)
        lines.append(f"## Query: \"{q}\"")
        lines.append("")

        for i, (_, row) in enumerate(rec.iterrows()):
            chunk_idx = int(row["chunk_index"])
            book = row["book_id"]
            chunk_id = row["chunk_id"]
            text = (row["text"] or "")[:400]
            if len(str(row["text"])) > 400:
                text += "..."

            cluster = chunk_to_cluster.get(chunk_idx, -1)
            terms_set = set(cluster_terms.get(cluster, []))
            chunk_tokens = _tokenize(text)
            overlap_chunk = q_tokens & chunk_tokens
            overlap_terms = q_tokens & terms_set
            overlap_any = overlap_chunk | overlap_terms

            lines.append(f"### Top-{i+1}: {book} (chunk {chunk_id})")
            lines.append("")
            lines.append("**Snippet:**")
            lines.append("> " + text.replace("\n", " ").strip())
            lines.append("")
            lines.append(f"**Cluster:** {cluster} — top terms: *" + ", ".join(cluster_terms.get(cluster, [])[:10]) + "*")
            lines.append("")
            if overlap_any:
                lines.append(f"**Overlap with query (in chunk or cluster terms):** " + ", ".join(sorted(overlap_any)))
            else:
                lines.append("**Overlap with query:** (none; match is semantic/similarity rather than lexical)")
            lines.append("")
            lines.append("---")
            lines.append("")

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "explain_recommendation.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
