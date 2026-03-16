#!/usr/bin/env python3
"""
Phase 6 pipeline: syntactic complexity by book (Hypothesis 3).

Run from project root with venv activated:
  python run_phase6.py

Requires: data/features/corpus_features.parquet (Phase 2) with syntactic columns.

Outputs in results/:
  syntactic_by_book.csv, syntactic_by_book.png,
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
    if not (FEATURES_DIR / "corpus_features.parquet").exists():
        print(f"Missing {FEATURES_DIR / 'corpus_features.parquet'}. Run: python run_phase2.py")
        sys.exit(1)

    from src.analysis.syntactic_complexity import run
    run(features_dir=FEATURES_DIR, results_dir=RESULTS_DIR)
    print("Phase 6 done. See results/ for syntactic table and figure.")


if __name__ == "__main__":
    main()
