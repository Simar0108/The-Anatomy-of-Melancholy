#!/usr/bin/env python3
"""
Chunk-level hypothesis tests: stoic vs existential on mean_sent_len (and optional sentiment).
Uses corpus_features.parquet so N = hundreds of chunks per group → much more power than book-level.

Run from project root:
  python scripts/stats_chunk_level.py
  python scripts/stats_chunk_level.py --groups stoic existential philosophy
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Chunk-level stats: stoic vs existential (high N)")
    p.add_argument("--features-dir", type=Path, default=PROJECT_ROOT / "data" / "features")
    p.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    p.add_argument("--groups", type=str, nargs="+", default=["stoic", "existential"])
    args = p.parse_args()

    import pandas as pd
    import numpy as np

    path = args.features_dir / "corpus_features.parquet"
    if not path.exists():
        print(f"Missing {path}. Run Phase 2 first.")
        sys.exit(1)

    df = pd.read_parquet(path)
    if "label_suffering_type" not in df.columns:
        print("corpus_features has no label_suffering_type.")
        sys.exit(1)
    if "mean_sent_len" not in df.columns:
        print("corpus_features has no mean_sent_len. Run Phase 2 with syntactic.")
        sys.exit(1)

    groups = args.groups
    lines_out = []
    lines_out.append("Chunk-level hypothesis tests (high N → more power)")
    lines_out.append("")

    # mean_sent_len
    vals = {}
    for g in groups:
        subset = df[df["label_suffering_type"] == g]
        if len(subset) == 0:
            print(f"No chunks for label '{g}'")
            continue
        vals[g] = subset["mean_sent_len"].dropna().values
    if len(vals) >= 2:
        a, b = vals[groups[0]], vals[groups[1]]
        try:
            from scipy import stats
            stat, p_value = stats.ttest_ind(a, b)
        except ImportError:
            stat, p_value = 0.0, 1.0
        mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
        line1 = "Metric: mean_sent_len (per chunk)"
        line2 = f"  {groups[0]}: n={len(a)}, mean={mean_a:.4f}"
        line3 = f"  {groups[1]}: n={len(b)}, mean={mean_b:.4f}"
        line4 = f"  p-value (t-test): {p_value:.4f}"
        line5 = f"  Significant at alpha=0.05: {'Yes' if p_value < 0.05 else 'No'}"
        print(line1)
        print(line2)
        print(line3)
        print(line4)
        print(line5)
        lines_out.extend([line1, line2, line3, line4, line5, ""])

    # Optional: sentiment compound per chunk
    if "sentiment_compound" in df.columns:
        col = "sentiment_compound"
        vals = {}
        for g in groups:
            subset = df[df["label_suffering_type"] == g]
            if len(subset) == 0:
                continue
            vals[g] = subset[col].dropna().values
        if len(vals) >= 2:
            a, b = vals[groups[0]], vals[groups[1]]
            try:
                from scipy import stats
                stat, p_value = stats.ttest_ind(a, b)
            except ImportError:
                p_value = 1.0
            mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
            line1 = f"Metric: {col} (per chunk)"
            line2 = f"  {groups[0]}: n={len(a)}, mean={mean_a:.4f}"
            line3 = f"  {groups[1]}: n={len(b)}, mean={mean_b:.4f}"
            line4 = f"  p-value (t-test): {p_value:.4f}"
            line5 = f"  Significant at alpha=0.05: {'Yes' if p_value < 0.05 else 'No'}"
            print(line1)
            print(line2)
            print(line3)
            print(line4)
            print(line5)
            lines_out.extend([line1, line2, line3, line4, line5, ""])

    if not lines_out:
        print("No comparisons run. Check --groups and corpus labels.")
        sys.exit(1)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / "stats_chunk_level.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
