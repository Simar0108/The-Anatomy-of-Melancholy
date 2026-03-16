"""
Phase 1.4–1.6: Chunk raw books into passages and build corpus.parquet.
Uses chapter/section headers when present, else fixed-word passages.

Usage (from project root with venv activated):
  python src/chunk_books.py
  python src/chunk_books.py --raw-dir /path/to/raw/books   # optional: custom raw folder
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# book_id -> label_suffering_type (for evaluation)
# Must match .txt filenames in raw dir and CORPUS in download_gutenberg.py
BOOK_LABELS = {
    "brothers_karamazov": "moral",
    "myth_of_sisyphus": "existential",
    "meditations": "stoic",
    "east_of_eden": "intergenerational",
    "enchiridion": "stoic",
    "crime_and_punishment": "moral",
    "the_stranger": "existential",
    "notes_from_underground": "existential",
}

# When no chapter headers found, split by this many words per passage
FALLBACK_WORDS = 800
# Minimum chunk size (words) to keep; merge smaller into next
MIN_CHUNK_WORDS = 100


def _word_count(text: str) -> int:
    return len(text.split())


def _split_by_headers(text: str) -> list[str]:
    """
    Split text on lines that look like chapter/section headers.
    Returns list of chunk texts (header line included in chunk).
    """
    # Patterns: "Chapter 1", "Book I", "Part One", "CHAPTER IV", "Book 1", etc.
    pattern = re.compile(
        r"^(?:(?:Book|Part|Section|Chapter|Epilogue|Prologue)\s+"
        r"(?:\d+|I{1,3}|IV|V|VI{0,3}|IX|X{1,3}|One|Two|First|Second)|"
        r"CHAPTER\s+[IVXLCDM\d]+)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    parts = []
    last_end = 0
    for m in pattern.finditer(text):
        start = m.start()
        if start > last_end:
            chunk = text[last_end:start].strip()
            if chunk and _word_count(chunk) >= MIN_CHUNK_WORDS:
                parts.append(chunk)
        last_end = start
    if last_end < len(text):
        chunk = text[last_end:].strip()
        if chunk and _word_count(chunk) >= MIN_CHUNK_WORDS:
            parts.append(chunk)
    return parts


def _split_fixed_words(text: str, words_per_chunk: int = FALLBACK_WORDS) -> list[str]:
    """Split text into chunks of roughly words_per_chunk words."""
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), words_per_chunk):
        chunk = " ".join(tokens[i : i + words_per_chunk])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def chunk_book(book_id: str, text: str, label: str) -> pd.DataFrame:
    """Split one book into chunks; return DataFrame with book_id, chunk_id, text, word_count, label."""
    # Try header-based split first
    parts = _split_by_headers(text)
    if len(parts) < 2:
        parts = _split_fixed_words(text)
    rows = []
    for i, chunk in enumerate(parts):
        wc = _word_count(chunk)
        if wc < MIN_CHUNK_WORDS and rows:
            # Merge into previous chunk
            rows[-1]["text"] = rows[-1]["text"] + "\n\n" + chunk
            rows[-1]["word_count"] = _word_count(rows[-1]["text"])
        else:
            rows.append(
                {
                    "book_id": book_id,
                    "chunk_id": len(rows) if rows else 0,
                    "text": chunk,
                    "word_count": wc,
                    "label_suffering_type": label,
                }
            )
    # Re-index chunk_id after possible merges
    for i, r in enumerate(rows):
        r["chunk_id"] = i
    return pd.DataFrame(rows)


def main(raw_dir: Path | None = None) -> None:
    raw_dir = raw_dir or DEFAULT_RAW_DIR
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for book_id, label in BOOK_LABELS.items():
        raw_path = raw_dir / f"{book_id}.txt"
        if not raw_path.exists():
            print(f"Skip {book_id}: missing {raw_path}. Run download_gutenberg.py or pass --raw-dir.")
            continue
        text = raw_path.read_text(encoding="utf-8")
        # Normalize whitespace
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        df = chunk_book(book_id, text, label)
        print(f"{book_id}: {len(df)} chunks, {df['word_count'].sum():,} words")
        all_dfs.append(df)
    if not all_dfs:
        raise SystemExit("No books processed. Run download_gutenberg.py first.")
    corpus = pd.concat(all_dfs, ignore_index=True)
    out_path = PROCESSED_DIR / "corpus.parquet"
    corpus.to_parquet(out_path, index=False)
    print(f"Wrote {len(corpus)} rows -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Chunk books into corpus.parquet")
    p.add_argument("--raw-dir", type=Path, default=None, help="Folder with book_id.txt files (default: data/raw)")
    args = p.parse_args()
    main(raw_dir=args.raw_dir)
