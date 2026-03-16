"""
Phase 2.4: Sentiment/affect per chunk (e.g. VADER) for trajectory and volatility analysis.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "corpus.parquet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "features"


def score_chunk_vader(text: str) -> dict:
    """Return compound score and optional pos/neg/neu for one chunk."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return {
        "sentiment_compound": scores["compound"],
        "sentiment_neg": scores["neg"],
        "sentiment_neu": scores["neu"],
        "sentiment_pos": scores["pos"],
    }


def run(
    corpus_path: Path | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Load corpus, score each chunk with VADER, save sentiment table. Returns DataFrame."""
    corpus_path = corpus_path or DEFAULT_CORPUS_PATH
    out_dir = out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(corpus_path)
    texts = df["text"].astype(str).tolist()

    rows = [score_chunk_vader(t) for t in texts]
    sent_df = pd.DataFrame(rows)

    out_path = out_dir / "sentiment.parquet"
    sent_df.to_parquet(out_path, index=False)
    print(f"Sentiment: {len(sent_df)} rows -> {out_path}")
    return sent_df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Extract sentiment (VADER) from corpus")
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    run(corpus_path=args.corpus, out_dir=args.out_dir)
