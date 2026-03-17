"""
Transformer-based sentiment per chunk (Hugging Face pipeline).
Output columns match VADER for compatibility: sentiment_compound, sentiment_pos, sentiment_neg, sentiment_neu.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "corpus.parquet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "features"
# Model: good balance of speed and quality; outputs pos/neg/label
DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
BATCH_SIZE = 32
MAX_LENGTH = 512


def _truncate(text: str, max_chars: int = 512 * 4) -> str:
    """Rough truncation to avoid token limit (approx 4 chars per token)."""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(maxsplit=1)[0] + " " if " " in text[:max_chars] else text[:max_chars]


def run(
    corpus_path: Path | None = None,
    out_dir: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """Load corpus, score each chunk with HF sentiment pipeline, save. Returns DataFrame."""
    from transformers import pipeline

    corpus_path = corpus_path or DEFAULT_CORPUS_PATH
    out_dir = out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(corpus_path)
    texts = df["text"].astype(str).tolist()

    pipe = pipeline("sentiment-analysis", model=model_name, device=-1, return_all_scores=True)
    rows = []
    for i in range(0, len(texts), batch_size):
        batch = [_truncate(t) for t in texts[i : i + batch_size]]
        # Pad empty so pipeline gets valid input
        batch = [t if t and len(t) > 10 else "neutral text" for t in batch]
        out = pipe(batch, truncation=True, max_length=MAX_LENGTH)
        for j, item in enumerate(out):
            if not item:
                rows.append({"sentiment_compound": 0.0, "sentiment_pos": 0.5, "sentiment_neg": 0.5, "sentiment_neu": 0.0})
                continue
            scores = {x["label"].lower(): x["score"] for x in item}
            pos = scores.get("positive", 0.5)
            neg = scores.get("negative", 0.5)
            compound = float(pos - neg)
            neu = max(0.0, 1.0 - pos - neg)
            rows.append({"sentiment_compound": compound, "sentiment_pos": pos, "sentiment_neg": neg, "sentiment_neu": neu})
        if len(texts) > 50 and (i + batch_size) % (batch_size * 4) == 0:
            print(f"  Sentiment: {min(i + batch_size, len(texts))}/{len(texts)} chunks")
    sent_df = pd.DataFrame(rows)

    out_path = out_dir / "sentiment.parquet"
    sent_df.to_parquet(out_path, index=False)
    print(f"Sentiment (transformer): {len(sent_df)} rows -> {out_path}")
    return sent_df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Extract sentiment (transformer) from corpus")
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = p.parse_args()
    run(corpus_path=args.corpus, out_dir=args.out_dir, model_name=args.model)
