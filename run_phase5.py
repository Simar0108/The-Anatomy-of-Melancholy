#!/usr/bin/env python3
"""
Phase 5 pipeline: sentiment trajectory and volatility (Hypothesis 1).

Run from project root with venv activated:
  python run_phase5.py

Requires: data/features/corpus_features.parquet (Phase 2) with sentiment columns.

Outputs in results/:
  volatility_by_book.csv, sentiment_trajectory_by_book.png,
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

    from src.analysis.sentiment_trajectory import run
    run(features_dir=FEATURES_DIR, results_dir=RESULTS_DIR)
    print("Phase 5 done. See results/ for volatility table and trajectory plot.")


if __name__ == "__main__":
    main()
