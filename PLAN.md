# The Anatomy of Melancholy — Project Plan

A phased plan for building the suffering taxonomy, visualizations, trajectory analysis, and recommendation engine.

---

## Phase 1: Environment & Data Pipeline

**Goal:** Reproducible environment and a clean corpus of chunked texts with metadata.

| Task | Description | Deliverable |
|------|-------------|-------------|
| **1.1** | Set up Python env: `requirements.txt`, optional `pyproject.toml` or venv | `requirements.txt`, `.python-version` or instructions |
| **1.2** | Create `data/raw/` and `data/processed/`; document where to get Gutenberg texts (URLs/IDs) | Folder structure, `data/README.md` |
| **1.3** | Download script or notebook: fetch primary corpus (Brothers Karamazov, Myth of Sisyphus, Meditations, East of Eden) from Project Gutenberg | `src/download_gutenberg.py` or `notebooks/01_download_corpus.ipynb` |
| **1.4** | Chunking pipeline: split each book into chapters (or fixed-length passages with overlap) and save as rows with `book_id`, `chapter_or_passage_id`, `text`, `word_count` | `src/chunk_books.py` or in notebook `02_chunk_corpus.ipynb` |
| **1.5** | Add secondary corpus: small set of sentimental poetry (same chunking convention if applicable) | Same pipeline or `data/raw/poetry/` + processed table |
| **1.6** | Build a single corpus table: CSV/Parquet with columns `[book_id, chunk_id, text, word_count, label_suffering_type]` (label optional, for later evaluation) | `data/processed/corpus.parquet` (or `.csv`) |

**Exit criterion:** Running one script (or notebook) produces `data/processed/corpus.parquet` and a short `data/README.md` describing columns and sources.

---

## Phase 2: Feature Extraction

**Goal:** Each chunk has vector representations and syntactic features for clustering, volatility, and recommendations.

| Task | Description | Deliverable |
|------|-------------|-------------|
| **2.1** | TF-IDF: fit on corpus vocabulary, transform chunks; save matrix and vocabulary | `src/features/tfidf.py`, output in `data/processed/` or `data/features/` |
| **2.2** | Embeddings: Word2Vec or sentence-transformers (e.g. `all-MiniLM-L6-v2`) per chunk; save matrix and metadata | `src/features/embeddings.py` |
| **2.3** | Syntactic features: sentence length (mean, std), punctuation density, optional clause count; use spaCy or regex | `src/features/syntactic.py` |
| **2.4** | Sentiment/affect: choose model (e.g. VADER, or transformer-based sentiment); score per chunk for trajectory and volatility | `src/features/sentiment.py` |
| **2.5** | Feature registry: one script or notebook that runs 2.1–2.4 and produces a feature set (e.g. `data/features/corpus_features.parquet` with embedding columns or paths) | `notebooks/03_extract_features.ipynb` or `src/pipeline/extract_all.py` |

**Exit criterion:** One command or notebook run produces all features needed for Phase 3–5 (TF-IDF, embeddings, syntactic, sentiment).

---

## Phase 3: Suffering Taxonomy (Clustering)

**Goal:** Test whether chunks cluster by “type of suffering” rather than only by author.

| Task | Description | Deliverable |
|------|-------------|-------------|
| **3.1** | K-Means: fit on embedding matrix (or TF-IDF); choose k (e.g. 4–6) via elbow/silhouette; save labels and model | `src/clustering/kmeans_taxonomy.py` or section in `04_clustering.ipynb` |
| **3.2** | Hierarchical clustering: linkage + dendrogram; optionally cut to get flat labels; compare with K-Means | Same module or notebook |
| **3.3** | Cluster description: for each cluster, report top TF-IDF terms and/or anchor words (Phase 4 can refine anchors) | Table or plot in notebook, saved to `results/cluster_descriptions.csv` |
| **3.4** | Cross-tab: cluster vs. book (and vs. optional `label_suffering_type`); brief interpretation | `results/cluster_vs_book.csv` + short write-up in notebook or `results/notes.md` |

**Exit criterion:** Replicable clustering pipeline; cluster assignments and cross-tabs saved; one narrative on “do we see suffering types vs. author identity?”

---

## Phase 4: Dimensionality Reduction & Lexical Anchors

**Goal:** Visualize “distance of thought” and identify anchor words that drive separation.

| Task | Description | Deliverable |
|------|-------------|-------------|
| **4.1** | PCA: fit on embedding (or TF-IDF) matrix; 3 components; save loadings and projected coordinates | `src/viz/pca_umap.py` or `05_dimensionality_reduction.ipynb` |
| **4.2** | UMAP: 2D and 3D; same matrix; save coordinates and (if possible) model for reuse | Same module/notebook |
| **4.3** | Plot: scatter by cluster and by book; save figures (e.g. `results/umap_by_cluster.png`, `results/umap_by_book.png`) | Script or notebook cells |
| **4.4** | Anchor words: from PCA loadings (top absolute weight per component) or differential TF-IDF (top terms per cluster); table of anchors per dimension/cluster | `results/anchor_words.csv`, optional plot |
| **4.5** | Hypothesis 2 write-up: do anchors align with intuitive “suffering types”? | Short section in report or `results/notes.md` |

**Exit criterion:** UMAP/PCA plots and anchor-word table; one paragraph interpreting anchors vs. hypothesis.

---

## Phase 5: Sentiment Trajectory & Volatility

**Goal:** “Emotional pulse” per book and a volatility metric for Hypothesis 1.

| Task | Description | Deliverable |
|------|-------------|-------------|
| **5.1** | Rolling window: define window (e.g. 5–10 chunks) and stride; compute mean sentiment per window along book order | `src/analysis/sentiment_trajectory.py` or `06_sentiment_trajectory.ipynb` |
| **5.2** | Plot: sentiment (y) vs. position (x) per book; one plot per book or faceted | `results/sentiment_trajectory_*.png` |
| **5.3** | Volatility metric: e.g. std of rolling sentiment, or variance of embedding position along book; compute per book | Same module/notebook |
| **5.4** | Hypothesis 1: compare volatility (Stoic vs. Existential); table and short interpretation | `results/volatility_by_book.csv` + write-up |

**Exit criterion:** Trajectory plots and volatility table; one paragraph on semantic/emotional divergence (Hypothesis 1).

---

## Phase 6: Syntactic Complexity (“Yearning” Signature)

**Goal:** Test Hypothesis 3: higher syntactic complexity for “yearning” vs. Stoic texts.

| Task | Description | Deliverable |
|------|-------------|-------------|
| **6.1** | Aggregate syntactic features by book (mean sentence length, punctuation density, etc.); optional: by cluster | `src/analysis/syntactic_complexity.py` or section in notebook |
| **6.2** | Compare books: e.g. Dostoevsky/Steinbeck vs. Marcus Aurelius; table and simple plot | `results/syntactic_by_book.csv` + figure |
| **6.3** | Hypothesis 3 write-up: do yearning-heavy books show higher complexity? | Short section in report or `results/notes.md` |

**Exit criterion:** Syntactic table and one paragraph on Hypothesis 3.

---

## Phase 7: “Resonance” Recommendation Engine

**Goal:** User input (reflection/poem) → recommend chunk and/or book by philosophical texture.

| Task | Description | Deliverable |
|------|-------------|-------------|
| **7.1** | Embedding index: same model as Phase 2; corpus chunk embeddings in a searchable structure (e.g. FAISS, or just matrix + cosine in numpy) | `src/recommendation/embed_index.py` |
| **7.2** | Query pipeline: embed user text → cosine similarity to corpus → return top-k chunks (and map to book/chapter) | `src/recommendation/query.py` or `recommend.py` |
| **7.3** | Interface: CLI (e.g. `python recommend.py "quote or reflection"`) or minimal Streamlit/Gradio app | `src/recommendation/cli.py` or `app.py` |
| **7.4** | Optional: extend to “recommend new books” (e.g. aggregate book-level embedding and compare to external list; API if available) | `src/recommendation/recommend_books.py` (stub or full) |
| **7.5** | Cross-genre test: run poetry samples through engine; do recommended chunks make sense? Brief note | `results/recommendation_poetry_notes.md` |

**Exit criterion:** User can input text and get top-k chunk/book recommendations; optional book-level extension; short cross-genre note.

---

## Phase 8: Report & Packaging

**Goal:** Single place that states hypotheses, methods, results, and repo layout.

| Task | Description | Deliverable |
|------|-------------|-------------|
| **8.1** | Final report or README section: hypotheses, dataset, methods (chunking, features, clustering, UMAP, sentiment, recommendation), results (tables/figures), limitations | `report.md` or `README.md` (expanded) or PDF |
| **8.2** | Repo hygiene: `requirements.txt` pinned (or lock file); `data/README.md`; clear instructions to run pipeline end-to-end (e.g. “Phase 1 → 2 → 3” in order) | Updated `README.md`, `requirements.txt` |
| **8.3** | Optional: single entrypoint script that runs Phases 1–2–3 (and optionally 4–5–6) so TA/grader can reproduce | `run_pipeline.py` or `Makefile` |

**Exit criterion:** Report written; someone else can clone, install, run pipeline, and reproduce main results.

---

## Suggested Order of Execution

1. **Phase 1** → **Phase 2** (data + features first).
2. **Phase 3** and **Phase 4** can run in parallel after Phase 2 (clustering + PCA/UMAP).
3. **Phase 5** and **Phase 6** next (trajectory + syntactic).
4. **Phase 7** (recommendation) once embedding pipeline is stable.
5. **Phase 8** last (report and packaging).

---

## File Layout (Target)

```
The-Anatomy-of-Melancholy/
├── README.md
├── PLAN.md
├── requirements.txt
├── data/
│   ├── README.md
│   ├── raw/              # Downloaded Gutenberg + poetry
│   ├── processed/        # corpus.parquet, chunk metadata
│   └── features/         # Optional: saved matrices/embeddings
├── src/
│   ├── download_gutenberg.py  (or in notebooks)
│   ├── chunk_books.py
│   ├── features/
│   │   ├── tfidf.py
│   │   ├── embeddings.py
│   │   ├── syntactic.py
│   │   └── sentiment.py
│   ├── clustering/
│   │   └── kmeans_taxonomy.py
│   ├── viz/
│   │   └── pca_umap.py
│   ├── analysis/
│   │   ├── sentiment_trajectory.py
│   │   └── syntactic_complexity.py
│   └── recommendation/
│       ├── embed_index.py
│       └── query.py
├── notebooks/
│   ├── 01_download_corpus.ipynb
│   ├── 02_chunk_corpus.ipynb
│   ├── 03_extract_features.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_dimensionality_reduction.ipynb
│   └── 06_sentiment_trajectory.ipynb
├── results/
│   ├── cluster_descriptions.csv
│   ├── cluster_vs_book.csv
│   ├── anchor_words.csv
│   ├── volatility_by_book.csv
│   ├── syntactic_by_book.csv
│   └── *.png
└── report.md
```

---

## Next Step

Start with **Phase 1**: create folder structure, `requirements.txt`, and the Gutenberg download + chunking pipeline so `data/processed/corpus.parquet` exists. Then we can proceed to Phase 2.
