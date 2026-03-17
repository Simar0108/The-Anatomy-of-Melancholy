#!/usr/bin/env python3
"""
Phase 7.3: CLI for the "Resonance" recommendation engine.
Input a quote or reflection; get top-k matching chunks from the corpus (book, chapter, text).

Usage (from project root with venv activated):
  python recommend.py "I am tormented by the idea of meaninglessness"
  python recommend.py "virtue and the soul" -k 10
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="Recommend corpus chunks by philosophical texture (cosine similarity on sentence embeddings)."
    )
    p.add_argument("query", type=str, nargs="?", default=None,
                   help="Quote or reflection to find similar passages for")
    p.add_argument("-k", type=int, default=5, help="Number of chunks to return (default: 5)")
    p.add_argument("--rerank", action="store_true", help="Re-rank top-20 with cross-encoder for better precision")
    p.add_argument("--features-dir", type=Path, default=None,
                   help="Path to data/features (default: project data/features)")
    args = p.parse_args()

    if not args.query:
        p.print_help()
        print("\nExample: python recommend.py \"I am tormented by the idea of meaninglessness\" -k 5")
        sys.exit(0)

    features_dir = args.features_dir or (PROJECT_ROOT / "data" / "features")
    if not (features_dir / "embeddings.npy").exists():
        print(f"Missing {features_dir / 'embeddings.npy'}. Run: python run_phase2.py", file=sys.stderr)
        sys.exit(1)

    from src.recommendation.query import recommend
    results = recommend(args.query, k=args.k, features_dir=features_dir, rerank=args.rerank)

    print(f"Top {len(results)} chunks for: \"{args.query[:60]}{'...' if len(args.query) > 60 else ''}\"\n")
    for i, row in results.iterrows():
        book = row["book_id"]
        chunk = row["chunk_id"]
        score = row["score"]
        text = (row["text"][:200] + "…") if len(str(row["text"])) > 200 else row["text"]
        label = f" [{row['label_suffering_type']}]" if "label_suffering_type" in row else ""
        print(f"  [{i+1}] {book} (chunk {chunk}) score={score:.4f}{label}")
        print(f"      {text}")
        print()


if __name__ == "__main__":
    main()
