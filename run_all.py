#!/usr/bin/env python3
"""
Single command to run the full pipeline and post-processing from project root.

  python run_all.py

What it does (in order, only running steps whose prerequisites exist):
  1. Phase 1: build corpus (if data/processed/corpus.parquet missing)
  2. Phase 2: features (if data/features/embeddings.npy missing)
  3. Phase 3: clustering
  4. Phase 4: PCA, UMAP, anchors
  5. Phase 5: sentiment trajectory, volatility
  6. Phase 6: syntactic by book
  7. Phase 7: recommendation check + sample queries
  8. Build FAISS index (if embeddings exist, index missing)
  9. Eval recommendation P@k (if data/relevance_set.csv and features exist)
 10. Stats hypothesis test (if results/volatility or syntactic exist)
 11. Cluster-label agreement NMI/ARI (if Phase 3 + corpus_features exist)
 12. Zero-shot eval: accuracy vs book label, NMI vs clusters (no fine-tuning)

Optional:
  python run_all.py --through 4        # stop after phase 4
  python run_all.py --skip-post        # skip FAISS, eval, stats
  python run_all.py --phase2-args "--embedding-model all-mpnet-base-v2 --sentiment transformer"

Requires: venv activated, pip install -r requirements.txt, spacy download en_core_web_sm.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS = PROJECT_ROOT / "data" / "processed" / "corpus.parquet"
FEATURES = PROJECT_ROOT / "data" / "features"
EMBEDDINGS = FEATURES / "embeddings.npy"
FAISS_INDEX = FEATURES / "embeddings.faiss"
RESULTS = PROJECT_ROOT / "results"
RELEVANCE_SET = PROJECT_ROOT / "data" / "relevance_set.csv"


def run(cmd: list[str], env=None, strict: bool = True) -> bool:
    """Run command; if strict, exit on non-zero. Returns True iff returncode == 0."""
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    if r.returncode != 0 and strict:
        sys.exit(r.returncode)
    return r.returncode == 0


def main() -> None:
    p = argparse.ArgumentParser(description="Run full pipeline + FAISS + eval + stats")
    p.add_argument("--through", type=int, default=7, help="Last phase to run (1-7)")
    p.add_argument("--skip-post", action="store_true", help="Skip FAISS build, eval, stats")
    p.add_argument("--phase2-args", type=str, default="", help="Extra args for run_phase2.py")
    args = p.parse_args()

    # Phase 1
    if not CORPUS.exists():
        print("\n" + "=" * 60 + "\nPhase 1: corpus\n" + "=" * 60)
        run([sys.executable, "run_phase1.py"])
    else:
        print("[ Phase 1: corpus already present ]")

    if args.through < 1:
        print("Stopped (--through 0).")
        return

    # Phase 2
    if not EMBEDDINGS.exists():
        print("\n" + "=" * 60 + "\nPhase 2: features\n" + "=" * 60)
        cmd = [sys.executable, "run_phase2.py"]
        if args.phase2_args:
            cmd.extend(args.phase2_args.split())
        run(cmd)
    else:
        print("[ Phase 2: features already present ]")

    if args.through < 2:
        return

    # Phases 3–7
    for phase in range(3, args.through + 1):
        print("\n" + "=" * 60 + f"\nPhase {phase}\n" + "=" * 60)
        run([sys.executable, f"run_phase{phase}.py"])

    if args.skip_post:
        print("\nDone (post-processing skipped).")
        return

    # FAISS
    if EMBEDDINGS.exists() and not FAISS_INDEX.exists():
        print("\n" + "=" * 60 + "\nBuilding FAISS index\n" + "=" * 60)
        run([sys.executable, "scripts/build_faiss_index.py"])
    elif FAISS_INDEX.exists():
        print("[ FAISS index already present ]")

    # Eval (optionally populate relevance set from model top-k if still placeholder)
    if RELEVANCE_SET.exists() and EMBEDDINGS.exists():
        try:
            import pandas as pd
            rel = pd.read_csv(RELEVANCE_SET)
            if "relevant_chunk_indices" in rel.columns:
                first = str(rel["relevant_chunk_indices"].iloc[0]).strip()
                # Placeholder: single small digits (e.g. 1, 2, 3) -> populate from model for sanity-check P@k
                if first in ("1", "2", "3") and len(rel) >= 1:
                    print("\n" + "=" * 60 + "\nPopulate relevance set from model (sanity check)\n" + "=" * 60)
                    run([sys.executable, "scripts/populate_relevance_from_model.py", "-k", "5"], strict=False)
        except Exception:
            pass
        print("\n" + "=" * 60 + "\nEval: recommendation P@k\n" + "=" * 60)
        run([sys.executable, "scripts/eval_recommendation.py"], strict=False)

    # Stats
    if (RESULTS / "volatility_by_book.csv").exists() or (RESULTS / "syntactic_by_book.csv").exists():
        print("\n" + "=" * 60 + "\nStats: hypothesis test (stoic vs existential)\n" + "=" * 60)
        run([sys.executable, "scripts/stats_hypothesis.py", "--all-metrics"], strict=False)

    # Cluster-label agreement (NMI/ARI)
    if (RESULTS / "labels_kmeans.csv").exists() and (FEATURES / "corpus_features.parquet").exists():
        print("\n" + "=" * 60 + "\nCluster-label agreement (NMI, ARI)\n" + "=" * 60)
        run([sys.executable, "scripts/cluster_label_agreement.py"], strict=False)

    # Zero-shot eval (no fine-tuning)
    if (FEATURES / "corpus_features.parquet").exists():
        print("\n" + "=" * 60 + "\nZero-shot classification vs book label & clusters\n" + "=" * 60)
        run([sys.executable, "scripts/zero_shot_eval.py", "--max-chunks", "400"], strict=False)

    print("\n" + "=" * 60 + "\nAll steps complete.\n" + "=" * 60)


if __name__ == "__main__":
    main()
