#!/usr/bin/env python3
"""
Zero-shot classification: assign each chunk (or a sample) a "suffering type" using
a pre-trained model with no fine-tuning. Compare to book-level labels and to K-means clusters.
Outputs: results/zero_shot_metrics.json, optional results/zero_shot_predictions.csv.

Run from project root (after Phase 2; Phase 3 optional for NMI vs cluster):
  python scripts/zero_shot_eval.py
  python scripts/zero_shot_eval.py --max-chunks 500   # cap for speed
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

# Suppress leaked-semaphore warning from loky/joblib at shutdown (Python 3.13 + HF)
warnings.filterwarnings("ignore", message=".*resource_tracker.*leaked semaphore.*", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Labels we ask the model to choose from (must be interpretable as "topic")
CANDIDATE_LABELS = [
    "stoic philosophy",
    "existential philosophy",
    "moral or ethical fiction",
    "intergenerational family narrative",
    "philosophy and metaphysics",
    "poetry and lyric",
]
# Map our book-level label to a short key for comparison (model returns full string)
LABEL_TO_KEY = {
    "stoic": "stoic philosophy",
    "existential": "existential philosophy",
    "moral": "moral or ethical fiction",
    "intergenerational": "intergenerational family narrative",
    "philosophy": "philosophy and metaphysics",
    "poetry_sentimental": "poetry and lyric",
    "poetry_existential": "poetry and lyric",
}
# Widely available zero-shot NLI model (no auth required)
DEFAULT_MODEL = "facebook/bart-large-mnli"
MAX_CHUNK_CHARS = 400  # truncate so we don't exceed model max length


def main() -> None:
    import pandas as pd
    import numpy as np

    p = __import__("argparse").ArgumentParser(description="Zero-shot eval: compare to book label and clusters")
    p.add_argument("--max-chunks", type=int, default=400, help="Max chunks to run (default 400)")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = p.parse_args()

    features_dir = PROJECT_ROOT / "data" / "features"
    results_dir = PROJECT_ROOT / "results"
    corpus_path = features_dir / "corpus_features.parquet"
    labels_path = results_dir / "labels_kmeans.csv"

    if not corpus_path.exists():
        print("Missing corpus_features.parquet. Run Phase 2 first.")
        sys.exit(1)

    corpus = pd.read_parquet(corpus_path)
    if "label_suffering_type" not in corpus.columns:
        print("corpus_features has no label_suffering_type.")
        sys.exit(1)

    texts = corpus["text"].astype(str).tolist()
    book_labels = corpus["label_suffering_type"].astype(str).tolist()
    n_total = len(texts)
    if args.max_chunks < n_total:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, size=args.max_chunks, replace=False)
        idx = np.sort(idx)
        texts = [texts[i] for i in idx]
        book_labels = [book_labels[i] for i in idx]
        chunk_indices = idx.tolist()
    else:
        chunk_indices = list(range(n_total))

    # Truncate for model
    texts = [t[:MAX_CHUNK_CHARS] if len(t) > MAX_CHUNK_CHARS else t for t in texts]
    texts = [t if t.strip() else "neutral text" for t in texts]

    print(f"Running zero-shot on {len(texts)} chunks (model={args.model})...")
    try:
        from transformers import pipeline
        pipe = pipeline("zero-shot-classification", model=args.model, device=-1)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    predictions = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        out = pipe(batch, candidate_labels=CANDIDATE_LABELS, multi_label=False)
        if not isinstance(out, list):
            out = [out]
        for o in out:
            pred = o["labels"][0] if o.get("labels") else CANDIDATE_LABELS[0]
            predictions.append(pred)
        if (i + batch_size) % 128 == 0:
            print(f"  {min(i + batch_size, len(texts))}/{len(texts)}")
    assert len(predictions) == len(book_labels)

    # Map book label to same schema for accuracy (book label is e.g. "stoic" -> "stoic philosophy")
    expected_keys = []
    for bl in book_labels:
        key = LABEL_TO_KEY.get(bl, bl)
        if key not in CANDIDATE_LABELS:
            key = "philosophy and metaphysics"  # fallback
        expected_keys.append(key)

    accuracy = sum(1 for p, e in zip(predictions, expected_keys) if p == e) / len(predictions)
    metrics = {"accuracy_vs_book_label": float(accuracy), "n_chunks": len(predictions), "model": args.model}

    # NMI with K-means cluster if available
    if labels_path.exists():
        from sklearn.metrics import normalized_mutual_info_score
        labels_df = pd.read_csv(labels_path)
        cluster = labels_df["cluster_kmeans"].values
        # For chunks we evaluated, get cluster id
        cluster_sub = cluster[chunk_indices]
        # We need to compare cluster (int) to prediction (string) — use prediction string as categorical
        nmi = normalized_mutual_info_score(cluster_sub, predictions)
        metrics["NMI_zero_shot_vs_cluster"] = float(nmi)

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "zero_shot_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Accuracy (vs book label) = {accuracy:.4f}")
    if "NMI_zero_shot_vs_cluster" in metrics:
        print(f"NMI(zero_shot_pred, cluster) = {metrics['NMI_zero_shot_vs_cluster']:.4f}")
    print(f"Wrote results/zero_shot_metrics.json")

    # Optional: save predictions for inspection
    pred_df = pd.DataFrame({
        "chunk_index": chunk_indices,
        "book_label": book_labels,
        "zero_shot_pred": predictions,
    })
    pred_df.to_csv(results_dir / "zero_shot_predictions.csv", index=False)
    print(f"Wrote results/zero_shot_predictions.csv")


if __name__ == "__main__":
    main()
