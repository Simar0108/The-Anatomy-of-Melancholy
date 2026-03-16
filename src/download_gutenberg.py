"""
Phase 1.3: Download primary corpus from Project Gutenberg.
Saves plain-text files to data/raw/ for chunking.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import requests

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Primary corpus: book_id, Gutenberg ID, optional label for later
# Expanded for better representation across suffering types / philosophy.
CORPUS = [
    ("brothers_karamazov", 28054, "moral"),
    ("myth_of_sisyphus", 52881, "existential"),
    ("meditations", 2680, "stoic"),
    ("east_of_eden", 1327, "intergenerational"),
    # Additional books for larger representation
    ("enchiridion", 871, "stoic"),           # Epictetus
    ("crime_and_punishment", 2554, "moral"), # Dostoevsky
    ("the_stranger", 11954, "existential"),  # Camus
    ("notes_from_underground", 22728, "existential"),  # Dostoevsky
]

# Be nice to Gutenberg: identify the app, avoid hammering
HEADERS = {
    "User-Agent": "TheAnatomyOfMelancholy/1.0 (CS252; educational)",
}
DELAY_SEC = 1.0
# Large books need long timeouts and retries
TIMEOUT_CONNECT = 30
TIMEOUT_READ = 300  # 5 min for multi-MB files
MAX_RETRIES = 3
RETRY_DELAY_SEC = 5
CHUNK_SIZE = 64 * 1024  # 64 KB


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Remove typical Gutenberg header and footer."""
    start_markers = (
        r"\*\*\* START OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK",
        r"\*\*\* START OF (?:THE |THIS )?DIGITAL LIBRARY",
        r"\*\*\* BEGIN OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK",
    )
    end_markers = (
        r"\*\*\* END OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK",
        r"\*\*\* END OF (?:THE |THIS )?DIGITAL LIBRARY",
        r"\*\*\* END OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK",
    )
    for pat in start_markers:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            text = text[m.end() :]
            break
    for pat in end_markers:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            text = text[: m.start()]
            break
    return text.strip()


def _fetch_streamed(url: str) -> str:
    """Download with streaming so large files don't time out on a single read."""
    r = requests.get(
        url,
        headers=HEADERS,
        stream=True,
        timeout=(TIMEOUT_CONNECT, TIMEOUT_READ),
    )
    r.raise_for_status()
    chunks = []
    for chunk in r.iter_content(chunk_size=CHUNK_SIZE, decode_unicode=False):
        if chunk:
            chunks.append(chunk)
    raw = b"".join(chunks)
    return raw.decode("utf-8", errors="replace")


def fetch_text(gutenberg_id: int) -> str:
    """Fetch plain text for a Gutenberg ID. Tries common URL patterns with retries."""
    urls = [
        f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-h/{gutenberg_id}-h.htm",
    ]
    last_error = None
    for url in urls:
        for attempt in range(MAX_RETRIES):
            try:
                text = _fetch_streamed(url)
                if url.endswith(".htm"):
                    text = re.sub(r"<[^>]+>", " ", text)
                    text = re.sub(r"\s+", " ", text)
                if len(text) > 500:
                    return _strip_gutenberg_boilerplate(text)
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    print(f"  Retry {attempt + 1}/{MAX_RETRIES} in {RETRY_DELAY_SEC}s: {e}")
                    time.sleep(RETRY_DELAY_SEC)
                else:
                    print(f"  Skip {url}: {e}")
        continue
    raise FileNotFoundError(f"Could not fetch Gutenberg ID {gutenberg_id}: {last_error}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for book_id, gid, label in CORPUS:
        out_path = RAW_DIR / f"{book_id}.txt"
        if out_path.exists() and out_path.stat().st_size > 1000:
            print(f"Already have {book_id}, skip download.")
            continue
        print(f"Downloading {book_id} (Gutenberg {gid})...")
        try:
            text = fetch_text(gid)
            out_path.write_text(text, encoding="utf-8")
            print(f"  Wrote {len(text):,} chars -> {out_path}")
        except FileNotFoundError as e:
            print(f"  Failed: {e}")
            failed.append(book_id)
        time.sleep(DELAY_SEC)
    if failed:
        print(f"\nSkipped {len(failed)} book(s): {failed}. You can re-run later or add .txt files manually to {RAW_DIR}.")
    print("Done. Raw files in", RAW_DIR)


if __name__ == "__main__":
    main()
