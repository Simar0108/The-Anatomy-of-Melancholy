#!/usr/bin/env python3
"""
Sentiment arc by label: per-book "arc" = mean(sentiment in second half) - mean(sentiment in first half).
Compare stoic vs existential (and other labels) to see if trajectory shape differs by type.

Outputs: results/arc_by_label.csv, results/arc_by_label.png, results/sentiment_arc_summary.md.

Run from project root (after Phase 2):
  python scripts/sentiment_arc_by_label.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    import numpy as np
    import pandas as pd

    features_dir = PROJECT_ROOT / "data" / "features"
    results_dir = PROJECT_ROOT / "results"
    path = features_dir / "corpus_features.parquet"
    if not path.exists():
        print("Run Phase 2 first (corpus_features.parquet).")
        sys.exit(1)

    df = pd.read_parquet(path)
    if "sentiment_compound" not in df.columns or "book_id" not in df.columns:
        print("Need sentiment_compound and book_id in corpus_features.")
        sys.exit(1)

    # Per-book: first half vs second half mean sentiment
    arcs = []
    for book_id, grp in df.groupby("book_id"):
        grp = grp.sort_values("chunk_id")
        n = len(grp)
        if n < 2:
            arcs.append({"book_id": book_id, "arc": 0.0, "n_chunks": n, "label_suffering_type": grp["label_suffering_type"].iloc[0] if "label_suffering_type" in grp.columns else ""})
            continue
        mid = n // 2
        first_half = grp["sentiment_compound"].iloc[:mid].mean()
        second_half = grp["sentiment_compound"].iloc[mid:].mean()
        arc = second_half - first_half
        label = grp["label_suffering_type"].iloc[0] if "label_suffering_type" in grp.columns else ""
        arcs.append({"book_id": book_id, "arc": float(arc), "n_chunks": n, "label_suffering_type": label})

    arc_df = pd.DataFrame(arcs)
    results_dir.mkdir(parents=True, exist_ok=True)
    arc_df.to_csv(results_dir / "arc_by_label.csv", index=False)
    print(f"Wrote {results_dir / 'arc_by_label.csv'}")

    # By-label summary (mean arc, count)
    if "label_suffering_type" in arc_df.columns and arc_df["label_suffering_type"].notna().any():
        by_label = arc_df.groupby("label_suffering_type").agg(mean_arc=("arc", "mean"), n_books=("book_id", "count")).reset_index()
        by_label.to_csv(results_dir / "arc_by_label_summary.csv", index=False)
        print(f"Wrote {results_dir / 'arc_by_label_summary.csv'}")

    # Plot: arc by book, colored by label
    try:
        import matplotlib.pyplot as plt
        labels = arc_df["label_suffering_type"].fillna("unknown")
        uniq = labels.unique()
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(uniq), 1)))
        label_to_color = dict(zip(uniq, colors[: len(uniq)]))
        fig, ax = plt.subplots(figsize=(10, 4))
        x = range(len(arc_df))
        bars = ax.bar(x, arc_df["arc"], color=[label_to_color.get(l, "gray") for l in labels])
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(arc_df["book_id"], rotation=45, ha="right")
        ax.set_ylabel("Sentiment arc (2nd half − 1st half)")
        ax.set_title("Sentiment arc by book")
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=label_to_color[l], label=l) for l in uniq]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(results_dir / "arc_by_label.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {results_dir / 'arc_by_label.png'}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    # Summary markdown
    md = [
        "# Sentiment arc by label",
        "",
        "**Arc** = mean(sentiment in second half of book) − mean(sentiment in first half). "
        "Positive = book tends more positive toward the end; negative = more negative toward the end.",
        "",
    ]
    if "label_suffering_type" in arc_df.columns and arc_df["label_suffering_type"].notna().any():
        by_label = arc_df.groupby("label_suffering_type")["arc"].agg(["mean", "count"])
        md.append("## Mean arc by label")
        md.append("")
        md.append("| Label | Mean arc | n_books |")
        md.append("|-------|----------|--------|")
        for label, row in by_label.iterrows():
            md.append(f"| {label} | {row['mean']:.4f} | {int(row['count'])} |")
        md.append("")
    md.append("See arc_by_label.csv for per-book values and arc_by_label.png for the bar chart.")
    with open(results_dir / "sentiment_arc_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Wrote {results_dir / 'sentiment_arc_summary.md'}")


if __name__ == "__main__":
    main()
