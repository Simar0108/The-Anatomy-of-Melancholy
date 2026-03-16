"""
Phase 5: Sentiment trajectory ("emotional pulse") per book and volatility for Hypothesis 1.
Rolling-window mean sentiment along book order; volatility = std of sentiment per book.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "data" / "features"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

WINDOW_SIZE = 8
STRIDE = 1
SENTIMENT_COL = "sentiment_compound"


def _load_corpus_with_sentiment(features_dir: Path) -> pd.DataFrame:
    path = features_dir / "corpus_features.parquet"
    if not path.exists():
        path = PROJECT_ROOT / "data" / "processed" / "corpus.parquet"
    if not path.exists():
        raise FileNotFoundError("Missing corpus.parquet or corpus_features.parquet")
    df = pd.read_parquet(path)
    if SENTIMENT_COL not in df.columns:
        sent_path = features_dir / "sentiment.parquet"
        if not sent_path.exists():
            raise FileNotFoundError("Run Phase 2 first: need sentiment in corpus_features or sentiment.parquet")
        sent = pd.read_parquet(sent_path)
        df = pd.concat([df, sent], axis=1)
    return df


def rolling_sentiment_per_book(
    df: pd.DataFrame,
    window: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> pd.DataFrame:
    """
    For each book, compute rolling mean of sentiment_compound along chunk order.
    Returns DataFrame with book_id, position (chunk index), rolling_mean.
    """
    rows = []
    for book_id, g in df.groupby("book_id", sort=False):
        g = g.sort_values("chunk_id").reset_index(drop=True)
        if SENTIMENT_COL not in g.columns:
            continue
        vals = g[SENTIMENT_COL].values.astype(np.float64)
        n = len(vals)
        for i in range(0, n - window + 1, stride):
            win = vals[i : i + window]
            rows.append({"book_id": book_id, "position": i + window // 2, "rolling_mean": float(np.mean(win))})
        if n < window and n > 0:
            rows.append({"book_id": book_id, "position": n // 2, "rolling_mean": float(np.mean(vals))})
    return pd.DataFrame(rows)


def volatility_per_book(df: pd.DataFrame) -> pd.DataFrame:
    """Per-book volatility = std(sentiment_compound). Optionally mean sentiment."""
    if SENTIMENT_COL not in df.columns:
        raise ValueError(f"Missing column {SENTIMENT_COL}")
    agg = df.groupby("book_id", sort=False)[SENTIMENT_COL].agg(["std", "mean", "count"]).reset_index()
    agg.columns = ["book_id", "volatility", "mean_sentiment", "n_chunks"]
    if "label_suffering_type" in df.columns:
        labels = df.groupby("book_id")["label_suffering_type"].first()
        agg = agg.merge(labels.reset_index(), on="book_id")
    agg["volatility"] = agg["volatility"].fillna(0)
    return agg


def plot_trajectory(
    rolling_df: pd.DataFrame,
    results_dir: Path,
) -> None:
    """Plot sentiment (rolling mean) vs position per book; faceted or one figure per book."""
    import matplotlib.pyplot as plt
    books = rolling_df["book_id"].unique().tolist()
    n_books = len(books)
    fig, axes = plt.subplots(n_books, 1, figsize=(8, 2.5 * n_books), sharex=True)
    if n_books == 1:
        axes = [axes]
    for ax, book_id in zip(axes, books):
        sub = rolling_df[rolling_df["book_id"] == book_id]
        ax.plot(sub["position"], sub["rolling_mean"], color="steelblue", alpha=0.8)
        ax.set_ylabel("Rolling mean sentiment")
        ax.set_title(str(book_id))
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("Chunk position")
    plt.tight_layout()
    plt.savefig(results_dir / "sentiment_trajectory_by_book.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {results_dir / 'sentiment_trajectory_by_book.png'}")


def run(
    features_dir: Path | None = None,
    results_dir: Path | None = None,
    window: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> pd.DataFrame:
    """
    Load corpus+sentiment, compute rolling trajectory and volatility per book,
    save volatility_by_book.csv, trajectory plot, and Hypothesis 1 note.
    Returns volatility table.
    """
    features_dir = features_dir or DEFAULT_FEATURES_DIR
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    df = _load_corpus_with_sentiment(features_dir)

    rolling_df = rolling_sentiment_per_book(df, window=window, stride=stride)
    if not rolling_df.empty:
        plot_trajectory(rolling_df, results_dir)

    vol_df = volatility_per_book(df)
    vol_df.to_csv(results_dir / "volatility_by_book.csv", index=False)
    print(f"Volatility: {results_dir / 'volatility_by_book.csv'}")

    with open(results_dir / "notes.md", "a", encoding="utf-8") as f:
        f.write("\n## Phase 5: Sentiment trajectory & Hypothesis 1\n\n")
        f.write("Rolling-window mean sentiment and per-book volatility (std of sentiment). ")
        f.write("Hypothesis 1: Stoic texts (e.g. Meditations) show lower emotional volatility ")
        f.write("than existential texts (e.g. Myth of Sisyphus). Compare volatility_by_book.csv ")
        f.write("across label_suffering_type (stoic vs existential).\n\n")

    return vol_df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Phase 5: sentiment trajectory and volatility")
    p.add_argument("--features-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--window", type=int, default=WINDOW_SIZE)
    p.add_argument("--stride", type=int, default=STRIDE)
    args = p.parse_args()
    run(features_dir=args.features_dir, results_dir=args.results_dir, window=args.window, stride=args.stride)
