# The Anatomy of Melancholy

CS252A Final Project: a **taxonomy of suffering** in classical and philosophical literature—beyond sentiment, toward existential, moral, and stoic texture.

## Hypotheses

1. **Semantic divergence** — Stoic texts show lower “emotional volatility” than existential texts in embedding space.
2. **Lexical anchors** — A small set of words drives cluster separation in 3D PCA.
3. **“Yearning” signature** — Authors focused on human yearning show higher syntactic complexity than the aphoristic Stoic style.

## Dataset

- **Primary:** The Brothers Karamazov, The Myth of Sisyphus, Meditations, East of Eden (Project Gutenberg), plus others TBD.
- **Secondary:** Sentimental poetry for cross-genre recommendation tests.
- **Features:** TF-IDF, Word2Vec/sentence embeddings, syntactic metrics (sentence length, punctuation density), sentiment.

## Tasks (summary)

1. **Suffering taxonomy** — Chunk books → K-Means + hierarchical clustering → cluster by type of suffering.
2. **Visualization** — UMAP (and PCA) for “distance of thought.”
3. **Sentiment trajectory** — Rolling-window “emotional pulse” per book.
4. **“Resonance” recommendation** — User input → cosine similarity → recommend chapter/book (and optionally new texts).

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

1. **Phase 1:** Download corpus and chunk into `data/processed/corpus.parquet` (see `data/README.md`).
2. **Phase 2:** Extract features (TF-IDF, embeddings, syntactic, sentiment).
3. **Phases 3–6:** Run clustering, UMAP, sentiment trajectory, syntactic analysis.
4. **Phase 7:** Run recommendation engine (CLI or app).
5. **Phase 8:** Report and packaging.

## Layout

```
├── README.md
├── PLAN.md
├── requirements.txt
├── data/           # raw, processed, features (see data/README.md)
├── src/            # download, chunk, features, clustering, viz, analysis, recommendation
├── notebooks/      # 01–06 for pipeline and analysis
├── results/        # tables, figures, notes
└── report.md       # final report (to be added)
```

## License

For course use; texts from Project Gutenberg follow their terms.
