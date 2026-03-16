# Data

## Layout

- **`raw/`** — Downloaded full texts from Project Gutenberg and any poetry files. Not committed (add to `.gitignore` if large).
- **`processed/`** — Chunked corpus: `corpus.parquet` (or `.csv`) with columns:
  - `book_id` — e.g. `brothers_karamazov`, `myth_of_sisyphus`, `meditations`, `east_of_eden`
  - `chunk_id` — chapter or passage index
  - `text` — full text of the chunk
  - `word_count` — optional
  - `label_suffering_type` — optional (for evaluation): moral, existential, stoic, intergenerational
- **`features/`** — Produced by Phase 2 (`python run_phase2.py`):
  - `tfidf_matrix.npy`, `tfidf_vocab.json` — TF-IDF matrix (n_chunks × n_features) and vocabulary
  - `embeddings.npy`, `embeddings_meta.txt` — sentence embeddings (n_chunks × dim)
  - `syntactic.parquet` — mean_sent_len, std_sent_len, punc_density per chunk
  - `sentiment.parquet` — VADER compound/pos/neg/neu per chunk
  - `corpus_features.parquet` — corpus columns + syntactic + sentiment (row index matches .npy matrices)

## Primary corpus (Project Gutenberg)

| Book | Gutenberg ID | Intended label |
|------|--------------|----------------|
| The Brothers Karamazov (Dostoevsky) | 28054 | moral |
| The Myth of Sisyphus (Camus) | 52881 | existential |
| Meditations (Marcus Aurelius) | 2680 | stoic |
| East of Eden (Steinbeck) | 1327 | intergenerational |
| The Enchiridion (Epictetus) | 871 | stoic |
| Crime and Punishment (Dostoevsky) | 2554 | moral |
| The Stranger (Camus) | 11954 | existential |
| Notes from the Underground (Dostoevsky) | 22728 | existential |

Download from: `https://www.gutenberg.org/files/<id>/<id>-0.txt` or use `python run_phase1.py`.

## Candidate books for cluster-based recommendation

- **`candidates_gutenberg.csv`** — Gutenberg ID, title, optional subject. Used by `python recommend_books.py` to recommend **whole books** per K-means cluster (same philosophical branch). You can add rows to discover more books; the script fetches a short text sample per candidate and ranks by similarity to each cluster centroid. This is how you can **use the recommender to find more books to add**: run `recommend_books.py`, then add top candidates to `CORPUS` in `src/download_gutenberg.py` and `BOOK_LABELS` in `src/chunk_books.py`, then re-run Phase 1–2–3.

**Finding candidates by "online" or metadata:** To discover books by style/philosophy (e.g. sentiment or genre), you can build the candidate list from external sources first, then rank by cluster. For example: (1) Use [Project Gutenberg’s catalog](https://www.gutenberg.org/ebooks/) or API (e.g. by subject: "Philosophy", "Stoicism", "Fiction") to get a list of Gutenberg IDs. (2) Add them to `candidates_gutenberg.csv`. (3) Run `recommend_books.py` to see which cluster each candidate matches. That way "online" or metadata gives you the candidate pool; the embedding+cluster similarity gives you the branch of philosophy (cluster) for each book.

## Secondary corpus

- Sentimental poetry: to be added (e.g. from Gutenberg poetry collection or a small curated list).

## Reproducing (with venv)

From project root with your venv activated (`source .venv/bin/activate`):

1. **Option A — download then chunk**
   ```bash
   python run_phase1.py
   ```
   This runs `src/download_gutenberg.py` then `src/chunk_books.py`.

2. **Option B — you already have the .txt files**
   - Put them in `data/raw/` with these exact names (at least the first four for minimal corpus):
     - `brothers_karamazov.txt`, `myth_of_sisyphus.txt`, `meditations.txt`, `east_of_eden.txt`
     - Optional (expanded corpus): `enchiridion.txt`, `crime_and_punishment.txt`, `the_stranger.txt`, `notes_from_underground.txt`
   - Then either:
     ```bash
     python run_phase1.py --skip-download
     ```
     or:
     ```bash
     python src/chunk_books.py
     ```
   - If your books are in another folder:
     ```bash
     python src/chunk_books.py --raw-dir /path/to/folder/with/txt/files
     ```

3. Output: `data/processed/corpus.parquet`.

4. **Phase 2 — features**
   ```bash
   python run_phase2.py
   ```
   Requires `corpus.parquet`. Produces all of `data/features/` (TF-IDF, embeddings, syntactic, sentiment, `corpus_features.parquet`).
