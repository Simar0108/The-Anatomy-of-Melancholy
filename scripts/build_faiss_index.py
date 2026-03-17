#!/usr/bin/env python3
"""Build FAISS index from data/features/embeddings.npy. Run after Phase 2."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.recommendation.faiss_index import build_index
if __name__ == "__main__":
    path = build_index()
    print(f"Built FAISS index -> {path}")
