# Key Results — The Anatomy of Melancholy

One-page summary of what we found. Full details: **report.md** and **results/**.

---

## 1. We don’t get a “suffering type” taxonomy — we get style and register

- Clustering and PCA separate texts by **lexical style and author** (narrative vs. archaic philosophical vs. book-specific vocabulary), not by the intended labels (stoic, existential, moral, intergenerational).
- **NMI/ARI** between K-means clusters and book labels are low (~0.26 / ~0.14). That’s the main **negative result**: the embedding space is not organizing by suffering type; it’s organizing by **how** the text is written.

---

## 2. One clear statistical result: chunk-level sentence length (H3)

- **Stoic vs. existential chunks** (n=135 vs 695): existential passages have **longer mean sentence length** than stoic (29.2 vs 25.4 words; **p = 0.0002**).
- So **H3 is supported** as a **stylistic** signal: aphoristic/compact (stoic) vs. more expansive prose (existential). Run: `python scripts/stats_chunk_level.py` → `results/stats_chunk_level.txt`.

---

## 3. Lexical anchors (H2)

- A small set of words drives PCA separation: **archaic** (*thou, thy, thee, unto*), **narrative** (*said, replied, don*), **setting-specific** (*king, france, duke*). Interpretable and reproducible. See `results/anchor_words.csv` and `results/cluster_descriptions.csv`.

---

## 4. Recommendation: expected book in the mix

- **Recall@book = 1.0**: for every query, at least one of the top-5 results comes from the “expected” book (e.g. meaninglessness → Myth of Sisyphus).
- **P@5 from expected book** is lower (~0.2–0.4) because the model also surfaces other philosophically related books — plausible for a “resonance” recommender. Eval: `data/eval_queries_by_book.csv` + `python scripts/eval_recommendation_by_book.py`.

---

## 5. Reranker ablation (B)

- Compare **bi-encoder only** vs **bi-encoder + cross-encoder rerank**. Results in `results/eval_reranker_ablation.json` and `eval_reranker_ablation.md`. Run: `python scripts/eval_reranker_ablation.py`.

---

## 6. Interpretability: why we recommended this (C)

- For each query we show **word overlap** (query words in the chunk or in the cluster’s top terms) and the **cluster** (with its defining terms). Run: `python scripts/explain_recommendation.py` → `results/explain_recommendation.md`.

---

## 7. Topic model vs. embeddings (D)

- **NMF on TF-IDF** (same chunks) gives a second view: topics and top terms. Compare with K-means clusters via `results/topic_vs_cluster.csv` and `topic_model_comparison.md`. Run: `python scripts/topic_model_compare.py`.

---

## 8. Sentiment arc by label (E)

- **Arc** = mean(sentiment in second half) − mean(sentiment in first half) per book. By-label summary and plot: `results/arc_by_label.csv`, `arc_by_label.png`, `sentiment_arc_summary.md`. Run: `python scripts/sentiment_arc_by_label.py`.

---

## How to reproduce

1. **Full pipeline:** `python run_all.py` (Phases 1–7 + FAISS + evals + stats).
2. **Extra analyses (B–E):**
   - `python scripts/eval_reranker_ablation.py`
   - `python scripts/explain_recommendation.py`
   - `python scripts/topic_model_compare.py`
   - `python scripts/sentiment_arc_by_label.py`

See **RUN.md** for step-by-step and **report.md** for hypotheses, methods, and limitations.
