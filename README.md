# The Anatomy of Melancholy

CS252A Final Project: a **taxonomy of suffering** in classical and philosophical literature—beyond sentiment, toward existential, moral, and stoic texture.

## Key results

- **We don’t recover a “suffering type” taxonomy** — clusters and PCA separate by **style and register** (archaic vs. narrative, author identity), not by stoic/existential/moral labels. NMI/ARI with book labels are low; that negative result is informative.
- **Chunk-level H3 (sentence length) is significant:** existential chunks have longer mean sentence length than stoic (p = 0.0002). Lexical anchors (H2) are interpretable and drive PCA separation.
- **Recommendation:** Recall@book = 1.0 (expected book appears in top-5 for every query); P@5 from expected book is modest. Optional reranker ablation, interpretability notes, topic model (NMF), and sentiment arc by label are in **KEY_RESULTS.md** and `results/`.

→ **One-page summary:** [KEY_RESULTS.md](KEY_RESULTS.md). **Findings, interpretation & architecture:** [FINDINGS_AND_ARCHITECTURE.md](FINDINGS_AND_ARCHITECTURE.md). Full report: [report.md](report.md).

## Hypotheses

1. **Semantic divergence** — Stoic texts show lower “emotional volatility” than existential texts in embedding space.
2. **Lexical anchors** — A small set of words drives cluster separation in 3D PCA.
3. **“Yearning” signature** — Authors focused on human yearning show higher syntactic complexity than the aphoristic Stoic style.

## Dataset

- **Primary (21 books):** 14 prose (philosophy & novels) + 7 sentimental/existential poetry (Dickinson, Whitman, Poe, Byron, Browning, Keats, Leaves of Grass). See `data/README.md` for Gutenberg IDs and labels. Built for robust, diverse clusters and recommendation.
- **Features:** TF-IDF, sentence embeddings (all-MiniLM-L6-v2), syntactic metrics, sentiment.

## Tasks (summary)

1. **Suffering taxonomy** — Chunk books → K-Means + hierarchical clustering → cluster by type of suffering.
2. **Visualization** — UMAP (and PCA) for “distance of thought.”
3. **Sentiment trajectory** — Rolling-window “emotional pulse” per book.
4. **“Resonance” recommendation** — (a) Chunk-level: `recommend.py "quote"` → similar passages. (b) **Cluster-based book recommendation:** `recommend_books.py` → recommend whole books (from a Gutenberg candidate list) that match each K-means cluster’s philosophical style, and use this to find more books to add to the study.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Download spaCy model (for syntactic features):

```bash
python -m spacy download en_core_web_sm
```

## How to run everything

**Single command (full pipeline + FAISS + eval + stats):**
```bash
python run_all.py
```
Runs phases 1–7 (skipping 1–2 if corpus/features already exist), then builds the FAISS index, runs recommendation evaluation (P@k), and the hypothesis test. See **[TECHNICAL_BREAKDOWN.md](TECHNICAL_BREAKDOWN.md)** for what this does and why.

**Step-by-step:** See **[RUN.md](RUN.md)** for a full walkthrough: prerequisites, Step 1 (build 21-book corpus) through Step 7, and a quick-reference table.

## Project plan

See **[PLAN.md](PLAN.md)** for the full phased plan (data pipeline → features → clustering → viz → trajectory → recommendation → report) and target file layout.

## Reproducing

1. **Phase 1:** `python run_phase1.py` → `data/processed/corpus.parquet` (see `data/README.md`).
2. **Phase 2:** `python run_phase2.py` → features in `data/features/`.
3. **Phase 3:** `python run_phase3.py` → clustering taxonomy in `results/` (K-Means, hierarchical, descriptions, cross-tabs).
4. **Phase 4:** `python run_phase4.py` → PCA, UMAP, anchor words, scatter plots in `results/`.
5. **Phase 5:** `python run_phase5.py` → sentiment trajectory plot and volatility_by_book.csv (Hypothesis 1).
6. **Phase 6:** `python run_phase6.py` → syntactic_by_book.csv + figure (Hypothesis 3).
7. **Phase 7:** `python run_phase7.py` → verifies recommendation pipeline; then `python recommend.py "quote or reflection"` for top-k chunk recommendations.
8. **Phase 8:** Report and packaging → `report.md`, this README, and `run_pipeline.py`.

**Full pipeline (reproduce everything in order):**
```bash
python run_pipeline.py
```
Optional: `python run_pipeline.py --through 4` (run only phases 1–4); `python run_pipeline.py --skip 1,2` (skip phases 1 and 2 if you already have corpus and features).

## Layout

```
├── README.md
├── KEY_RESULTS.md  # one-page summary of key results (for sharing)
├── RUN.md          # step-by-step walkthrough
├── run_all.py      # full pipeline + FAISS + eval + stats + ablation + explain + topic model + arc
├── TECHNICAL_BREAKDOWN.md
├── PLAN.md
├── requirements.txt
├── report.md       # final report (hypotheses, methods, results, limitations)
├── run_pipeline.py # run Phases 1–7 in order
├── recommend.py    # CLI: recommend chunks by quote (optional --rerank)
├── recommend_books.py
├── data/           # raw, processed, features, eval_queries_by_book.csv, relevance_set.csv
├── scripts/        # eval_reranker_ablation, explain_recommendation, topic_model_compare, sentiment_arc_by_label, ...
├── src/
├── notebooks/
└── results/        # tables, figures, eval_reranker_ablation, explain_recommendation.md, topic_*, arc_*, ...
```

## License

For course use; texts from Project Gutenberg follow their terms.
