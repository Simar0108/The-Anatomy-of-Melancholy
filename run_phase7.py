#!/usr/bin/env python3
"""
Phase 7 pipeline: "Resonance" recommendation engine.
Builds on Phase 2 embeddings: load index, run test query, optionally run cross-genre (poetry) check.

Run from project root with venv activated:
  python run_phase7.py

Requires: data/features/ from Phase 2 (embeddings.npy, corpus_features.parquet).

Outputs:
  - Verifies recommendation pipeline and runs sample queries.
  - Optional: results/recommendation_poetry_notes.md (if poetry samples are added).

After Phase 7, use: python recommend.py "your quote or reflection" -k 5
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
    if not (FEATURES_DIR / "corpus_features.parquet").exists():
        print(f"Missing {FEATURES_DIR / 'corpus_features.parquet'}. Run: python run_phase2.py")
        sys.exit(1)

    from src.recommendation.embed_index import load_index
    from src.recommendation.query import recommend

    print("Loading embedding index...")
    X, df, model_name = load_index(FEATURES_DIR)
    print(f"  Index: {X.shape[0]} chunks, dim={X.shape[1]}, model={model_name}")

    # Test queries (existential / stoic / moral flavor)
    test_queries = [
        "I am tormented by the idea of meaninglessness",
        "Accept what is in your control and let go of the rest",
    ]
    print("\nRunning sample recommendations:\n")
    for q in test_queries:
        results = recommend(q, k=3, features_dir=FEATURES_DIR)
        print(f"  Query: \"{q[:50]}...\" " if len(q) > 50 else f"  Query: \"{q}\"")
        for _, row in results.iterrows():
            print(f"    -> {row['book_id']} (chunk {row['chunk_id']}) score={row['score']:.4f}")
        print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    notes_path = RESULTS_DIR / "recommendation_poetry_notes.md"
    if not notes_path.exists():
        notes_path.write_text(
            "# Phase 7: Recommendation engine\n\n"
            "Cross-genre test: run poetry or other short samples through the CLI and check whether "
            "recommended chunks match the intended philosophical texture. Example:\n"
            "  python recommend.py \"O heart, heart of mine, the bitter and the sweet\" -k 5\n\n"
            "Add brief notes here after testing.\n",
            encoding="utf-8",
        )
        print(f"Created {notes_path} for cross-genre notes.")

    print("Phase 7 done. Use: python recommend.py \"your quote or reflection\" -k 5")


if __name__ == "__main__":
    main()
