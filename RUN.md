# How to run the full pipeline

This guide walks you through building a **robust, diverse** corpus (20+ books including sentimental/existential poetry), then running the full pipeline to get clusters and both recommendation systems.

---

## Prerequisites

- **Python 3.9+** and a virtual environment (recommended).
- **Network** access (to download from Project Gutenberg and to load embedding models the first time).

From the project root:

```bash
# Create and activate venv (one-time)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies (one-time)
pip install -r requirements.txt

# SpaCy model for syntactic features (one-time)
python -m spacy download en_core_web_sm
```

---

## Step 1: Build the corpus (20+ books)

This downloads **21 books** from Project Gutenberg (14 prose + 7 poetry) and chunks them into `data/processed/corpus.parquet`.

```bash
python run_phase1.py
```

- **What it does:** Runs `src/download_gutenberg.py` (saves `.txt` files to `data/raw/`), then `src/chunk_books.py` (builds `data/processed/corpus.parquet`).
- **Time:** Several minutes (downloads are rate-limited; some books are large).
- **If a download fails:** The script skips that book and continues. You can re-run later; existing files are not re-downloaded. Add missing `.txt` files manually to `data/raw/` if needed (see `data/README.md` for exact filenames).

**Corpus contents:**

| Type        | Count | Examples |
|------------|-------|----------|
| Prose      | 14    | Meditations, Myth of Sisyphus, Brothers Karamazov, East of Eden, Enchiridion, Crime and Punishment, The Stranger, Notes from the Underground, Seneca Letters, Zarathustra, Sickness Unto Death, Republic, Man's Search for Meaning |
| Poetry     | 7     | Dickinson, Whitman, Poe, Byron (Childe Harold), Browning (Sonnets), Keats, Leaves of Grass |

Labels used for evaluation: `moral`, `existential`, `stoic`, `intergenerational`, `philosophy`, `poetry_sentimental`, `poetry_existential`.

---

## Step 2: Extract features

```bash
python run_phase2.py
```

- **What it does:** Builds TF-IDF, sentence embeddings (sentence-transformers), syntactic features (spaCy), and sentiment (VADER) for every chunk. Writes everything to `data/features/` (including `corpus_features.parquet` and `embeddings.npy`).
- **Time:** 10–30+ minutes depending on machine (embedding model download on first run, then encoding all chunks).
- **Requires:** `data/processed/corpus.parquet` from Step 1.

---

## Step 3: Build the taxonomy (clustering)

```bash
python run_phase3.py
```

- **What it does:** Runs K-Means and hierarchical clustering on chunk embeddings, picks k (e.g. by silhouette), writes cluster labels, descriptions (top TF-IDF terms per cluster), and cross-tabs (cluster vs book, cluster vs label) to `results/`.
- **Requires:** `data/features/` from Step 2.
- **Outputs:** `results/labels_kmeans.csv`, `results/cluster_descriptions.csv`, `results/cluster_vs_book.csv`, `results/cluster_vs_label.csv`, `results/dendrogram.png`, etc.

These clusters are what drive **cluster-based book recommendation** (Step 7b).

---

## Step 4: Dimensionality reduction and anchor words

```bash
python run_phase4.py
```

- **What it does:** PCA on TF-IDF, UMAP 2D/3D on embeddings, anchor words from PCA loadings, scatter plots by cluster and by book.
- **Requires:** Step 2 and Step 3.
- **Outputs:** `results/pca_*.npy`, `results/umap_*.npy`, `results/anchor_words.csv`, `results/umap_by_cluster.png`, `results/umap_by_book.png`.

---

## Step 5: Sentiment trajectory and volatility

```bash
python run_phase5.py
```

- **What it does:** Rolling-window sentiment along book order, per-book volatility (std of sentiment), trajectory plot.
- **Outputs:** `results/sentiment_trajectory_by_book.png`, `results/volatility_by_book.csv`.

---

## Step 6: Syntactic complexity

```bash
python run_phase6.py
```

- **What it does:** Per-book mean sentence length, punctuation density; table and figure.
- **Outputs:** `results/syntactic_by_book.csv`, `results/syntactic_by_book.png`.

---

## Step 7: Recommendation systems

You now have **two** recommendation modes.

### 7a. Chunk-level (passage) recommendation

```bash
python run_phase7.py
```
*(Optional: verifies the pipeline and runs sample queries.)*

Then, anytime:

```bash
python recommend.py "I am tormented by the idea of meaninglessness" -k 5
```

- **What it does:** Embeds your quote, finds the top-k **chunks** (passages) in the corpus by cosine similarity, prints book, chunk id, score, and a snippet.
- **Use case:** “Find passages that resonate with this thought.”

### 7b. Cluster-based book recommendation (whole books, by philosophy branch)

```bash
python recommend_books.py --top 5
```

- **What it does:** Loads K-means cluster centroids and a **candidate list** of books (`data/candidates_gutenberg.csv`). For each candidate, fetches a short text sample from Gutenberg, embeds it, and ranks candidates by similarity to each cluster. Prints top-N **books per cluster** (same philosophical/style branch).
- **Options:** `--candidates path/to.csv`, `--top 10`, `--cluster 0` (only cluster 0).
- **Use case:** “What whole books (from Gutenberg) match each of my clusters?” and “Which books should I add next to the study?”

To **add more candidate books:** Edit `data/candidates_gutenberg.csv` (add rows: `gutenberg_id`, `title`, optional `subject`), then run `recommend_books.py` again.

---

## Run everything in one go

To reproduce the full pipeline from scratch (corpus → features → clustering → viz → trajectory → syntactic → recommendation check):

```bash
python run_pipeline.py
```

- **Skip phases:** e.g. if you already have corpus and features:  
  `python run_pipeline.py --skip 1,2`
- **Run only through a phase:** e.g. through Phase 4:  
  `python run_pipeline.py --through 4`

---

## Quick reference: order of operations

| Step | Command | Produces |
|------|---------|----------|
| 1 | `python run_phase1.py` | `data/raw/*.txt`, `data/processed/corpus.parquet` |
| 2 | `python run_phase2.py` | `data/features/*` (embeddings, TF-IDF, syntactic, sentiment) |
| 3 | `python run_phase3.py` | `results/labels_kmeans.csv`, cluster descriptions, cross-tabs |
| 4 | `python run_phase4.py` | PCA, UMAP, anchor words, scatter plots |
| 5 | `python run_phase5.py` | Sentiment trajectory, volatility_by_book |
| 6 | `python run_phase6.py` | Syntactic by book |
| 7a | `python recommend.py "quote"` | Top-k similar **chunks** |
| 7b | `python recommend_books.py` | Top **books** per cluster (from candidates) |

**Data flow:** Phase 1 → Phase 2 → Phase 3. Phases 4–6 can run after 3. Recommendation (7a and 7b) uses outputs of Phase 2 and (for 7b) Phase 3.

---

## Are we doing this right now?

- **20+ books:** Yes. The corpus is defined as **21 books** (14 prose + 7 poetry) in `src/download_gutenberg.py` and `src/chunk_books.py`. Run `run_phase1.py` to download and chunk them.
- **Sentimental / existential poetry:** Yes. Seven poetry works are included (Dickinson, Whitman, Poe, Byron, Browning, Keats, Leaves of Grass) with labels `poetry_sentimental` and `poetry_existential`.
- **Robust, diverse data for clusters and recommendation:** Yes. With 21 books and mixed prose/poetry and labels, the pipeline builds richer clusters and more diverse chunk-level and cluster-based book recommendations. For maximum robustness, run the full pipeline (Steps 1–7) as above.
