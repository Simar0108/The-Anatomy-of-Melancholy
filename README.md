# The Anatomy of Melancholy

CS252A Final Project: a **taxonomy of suffering** in classical and philosophical literature—beyond sentiment, toward existential, moral, and stoic texture.

## Hypotheses

1. **Semantic divergence** — Stoic texts show lower “emotional volatility” than existential texts in embedding space.
2. **Lexical anchors** — A small set of words drives cluster separation in 3D PCA.
3. **“Yearning” signature** — Authors focused on human yearning show higher syntactic complexity than the aphoristic Stoic style.

## Dataset

- **Primary (8 books):** Brothers Karamazov, Myth of Sisyphus, Meditations, East of Eden, Enchiridion (Epictetus), Crime and Punishment, The Stranger, Notes from the Underground (Project Gutenberg). See `data/README.md` for IDs and labels.
- **Secondary:** Sentimental poetry for cross-genre recommendation tests.
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
├── PLAN.md
├── requirements.txt
├── report.md       # final report (hypotheses, methods, results, limitations)
├── run_pipeline.py # run Phases 1–7 in order
├── recommend.py    # CLI: recommend chunks by quote
├── data/           # raw, processed, features (see data/README.md)
├── src/            # download, chunk, features, clustering, viz, analysis, recommendation
├── notebooks/      # 01–06 for pipeline and analysis
└── results/        # tables, figures, notes
```

## License

For course use; texts from Project Gutenberg follow their terms.
