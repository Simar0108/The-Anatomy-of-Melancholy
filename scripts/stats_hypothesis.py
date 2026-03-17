#!/usr/bin/env python3
"""
Run a simple statistical test for hypothesis support: e.g. mean sentence length
or volatility for stoic vs existential books (t-test or permutation test).
Writes results to results/stats_hypothesis.txt for inclusion in the report.

Run from project root:
  python scripts/stats_hypothesis.py
  python scripts/stats_hypothesis.py --metric volatility --groups stoic existential
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Statistical test for hypothesis (e.g. stoic vs existential)")
    p.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    p.add_argument("--metric", type=str, choices=["mean_sent_len", "volatility"], default="mean_sent_len")
    p.add_argument("--groups", type=str, nargs="+", default=["stoic", "existential"])
    args = p.parse_args()

    import pandas as pd
    import numpy as np

    # Load per-book metrics
    if args.metric == "volatility":
        path = args.results_dir / "volatility_by_book.csv"
        col = "volatility"
    else:
        path = args.results_dir / "syntactic_by_book.csv"
        col = "mean_sent_len"
    if not path.exists():
        print(f"Missing {path}. Run Phase 5 (volatility) or Phase 6 (syntactic) first.")
        sys.exit(1)

    df = pd.read_csv(path)
    if "label_suffering_type" not in df.columns or col not in df.columns:
        print(f"Expected columns label_suffering_type and {col}")
        sys.exit(1)

    groups = args.groups
    vals = {}
    for g in groups:
        subset = df[df["label_suffering_type"] == g]
        if len(subset) == 0:
            print(f"No rows for label '{g}'")
            continue
        vals[g] = subset[col].dropna().values
    if len(vals) < 2:
        print("Need at least two groups with data.")
        sys.exit(1)

    a, b = vals[groups[0]], vals[groups[1]]
    try:
        from scipy import stats
        stat, p_value = stats.ttest_ind(a, b)
    except ImportError:
        # Permutation test (no scipy)
        def permute_diff(a, b, n_perm=5000):
            combined = np.concatenate([a, b])
            n_a, diff_obs = len(a), np.mean(a) - np.mean(b)
            count = 0
            for _ in range(n_perm):
                np.random.shuffle(combined)
                diff = np.mean(combined[:n_a]) - np.mean(combined[n_a:])
                if abs(diff) >= abs(diff_obs):
                    count += 1
            return diff_obs, count / n_perm
        stat, p_value = permute_diff(a, b)
        stat = float(stat)
        p_value = float(p_value)

    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    line1 = f"Metric: {col}"
    line2 = f"  {groups[0]}: n={len(a)}, mean={mean_a:.4f}"
    line3 = f"  {groups[1]}: n={len(b)}, mean={mean_b:.4f}"
    line4 = f"  p-value (t-test): {p_value:.4f}"
    line5 = f"  Significant at alpha=0.05: {'Yes' if p_value < 0.05 else 'No'}"
    print(line1)
    print(line2)
    print(line3)
    print(line4)
    print(line5)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / "stats_hypothesis.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join([line1, line2, line3, line4, line5]))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
