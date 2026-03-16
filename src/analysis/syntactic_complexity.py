"""
Phase 6: Syntactic complexity by book — test Hypothesis 3 (yearning vs Stoic style).
Aggregate mean_sent_len, std_sent_len, punc_density by book; table + figure.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "data" / "features"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

SYNTACTIC_COLS = ["mean_sent_len", "std_sent_len", "punc_density"]


def _load_corpus_with_syntactic(features_dir: Path) -> pd.DataFrame:
    path = features_dir / "corpus_features.parquet"
    if not path.exists():
        path = PROJECT_ROOT / "data" / "processed" / "corpus.parquet"
    if not path.exists():
        raise FileNotFoundError("Missing corpus.parquet or corpus_features.parquet")
    df = pd.read_parquet(path)
    missing = [c for c in SYNTACTIC_COLS if c not in df.columns]
    if missing:
        syn_path = features_dir / "syntactic.parquet"
        if not syn_path.exists():
            raise FileNotFoundError("Run Phase 2 first: need syntactic features in corpus_features or syntactic.parquet")
        syn = pd.read_parquet(syn_path)
        df = pd.concat([df, syn], axis=1)
    return df


def aggregate_syntactic_by_book(df: pd.DataFrame) -> pd.DataFrame:
    """Per-book means of mean_sent_len, std_sent_len, punc_density; optional label."""
    agg_cols = [c for c in SYNTACTIC_COLS if c in df.columns]
    if not agg_cols:
        raise ValueError("No syntactic columns found.")
    by_book = df.groupby("book_id", sort=False)[agg_cols].mean().reset_index()
    by_book["n_chunks"] = df.groupby("book_id", sort=False).size().values
    if "label_suffering_type" in df.columns:
        labels = df.groupby("book_id")["label_suffering_type"].first()
        by_book = by_book.merge(labels.reset_index(), on="book_id")
    return by_book


def plot_syntactic_by_book(by_book: pd.DataFrame, results_dir: Path) -> None:
    """Bar chart: mean sentence length and punctuation density by book."""
    import matplotlib.pyplot as plt
    books = by_book["book_id"].astype(str).tolist()
    x = range(len(books))
    width = 0.35
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    if "mean_sent_len" in by_book.columns:
        ax1.bar([i - width / 2 for i in x], by_book["mean_sent_len"], width, label="Mean sent. length", color="steelblue")
        ax1.set_ylabel("Mean sentence length")
        ax1.set_title("Syntactic complexity by book")
        ax1.legend(loc="upper right")
    if "punc_density" in by_book.columns:
        ax2.bar([i + width / 2 for i in x], by_book["punc_density"], width, label="Punctuation density", color="coral", alpha=0.8)
        ax2.set_ylabel("Punctuation density")
        ax2.legend(loc="upper right")
    ax2.set_xticks(x)
    ax2.set_xticklabels(books, rotation=15, ha="right")
    ax2.set_xlabel("Book")
    plt.tight_layout()
    plt.savefig(results_dir / "syntactic_by_book.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {results_dir / 'syntactic_by_book.png'}")


def run(
    features_dir: Path | None = None,
    results_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Load corpus with syntactic features, aggregate by book, save syntactic_by_book.csv
    and figure; append Hypothesis 3 note. Returns syntactic-by-book table.
    """
    features_dir = features_dir or DEFAULT_FEATURES_DIR
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    df = _load_corpus_with_syntactic(features_dir)
    by_book = aggregate_syntactic_by_book(df)

    out_path = results_dir / "syntactic_by_book.csv"
    by_book.to_csv(out_path, index=False)
    print(f"Syntactic by book: {out_path}")

    plot_syntactic_by_book(by_book, results_dir)

    with open(results_dir / "notes.md", "a", encoding="utf-8") as f:
        f.write("\n## Phase 6: Syntactic complexity & Hypothesis 3\n\n")
        f.write("Per-book mean sentence length and punctuation density. ")
        f.write("Hypothesis 3: yearning-focused texts (e.g. Dostoevsky, Steinbeck) show higher ")
        f.write("syntactic complexity than aphoristic Stoic style (e.g. Meditations). ")
        f.write("Compare syntactic_by_book.csv and syntactic_by_book.png across books.\n\n")

    return by_book


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Phase 6: syntactic complexity by book")
    p.add_argument("--features-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    args = p.parse_args()
    run(features_dir=args.features_dir, results_dir=args.results_dir)
