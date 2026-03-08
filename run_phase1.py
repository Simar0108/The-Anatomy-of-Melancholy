#!/usr/bin/env python3
"""
Phase 1 pipeline: download (if needed) + chunk -> data/processed/corpus.parquet.

Run from project root with your venv activated:
  source .venv/bin/activate
  python run_phase1.py

Optional: use already-downloaded books in another folder:
  python run_phase1.py --raw-dir /path/to/your/raw/books
  (Expects files: brothers_karamazov.txt, myth_of_sisyphus.txt, meditations.txt, east_of_eden.txt)
"""
import argparse
import sys
from pathlib import Path

# So we can run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

def main():
    p = argparse.ArgumentParser(description="Phase 1: build corpus.parquet")
    p.add_argument("--raw-dir", type=Path, default=None, help="Folder with .txt books (default: data/raw); if set, skips download")
    p.add_argument("--skip-download", action="store_true", help="Do not run download; only chunk")
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent
    raw_dir = args.raw_dir or (project_root / "data" / "raw")
    skip_download = args.skip_download or (args.raw_dir is not None)

    if not skip_download:
        from src.download_gutenberg import main as download_main
        download_main()
    else:
        print("Skipping download; using raw dir:", raw_dir)

    from src.chunk_books import main as chunk_main
    chunk_main(raw_dir=raw_dir)


if __name__ == "__main__":
    main()
