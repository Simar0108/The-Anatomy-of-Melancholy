# Data

## Layout

- **`raw/`** — Downloaded full texts from Project Gutenberg and any poetry files. Not committed (add to `.gitignore` if large).
- **`processed/`** — Chunked corpus: `corpus.parquet` (or `.csv`) with columns:
  - `book_id` — e.g. `brothers_karamazov`, `myth_of_sisyphus`, `meditations`, `east_of_eden`
  - `chunk_id` — chapter or passage index
  - `text` — full text of the chunk
  - `word_count` — optional
  - `label_suffering_type` — optional (for evaluation): moral, existential, stoic, intergenerational, philosophy, poetry_sentimental, poetry_existential
- **`features/`** — Produced by Phase 2 (`python run_phase2.py`):
  - `tfidf_matrix.npy`, `tfidf_vocab.json` — TF-IDF matrix (n_chunks × n_features) and vocabulary
  - `embeddings.npy`, `embeddings_meta.txt` — sentence embeddings (n_chunks × dim)
  - `syntactic.parquet` — mean_sent_len, std_sent_len, punc_density per chunk
  - `sentiment.parquet` — VADER compound/pos/neg/neu per chunk
  - `corpus_features.parquet` — corpus columns + syntactic + sentiment (row index matches .npy matrices)
- **`relevance_set.csv`** — For recommendation P@k evaluation: columns `query_id`, `query_text`, `relevant_chunk_indices` (comma-separated **corpus row indices** 0..N-1). If you use placeholders (e.g. 1, 2, 3), P@k will be 0. To get a sanity-check non-zero P@k, run once: `python scripts/populate_relevance_from_model.py` (overwrites with the model’s top-5 per query). For real evaluation, replace `relevant_chunk_indices` with human-judged corpus row indices.

## Primary corpus (Project Gutenberg) — 21 books

**Prose (14):** Brothers Karamazov (28054, moral), Myth of Sisyphus (52881, existential), Meditations (2680, stoic), East of Eden (1327, intergenerational), Enchiridion (871, stoic), Crime and Punishment (2554, moral), The Stranger (11954, existential), Notes from the Underground (22728, existential), Seneca Letters (47078, stoic), Zarathustra (1998, existential), Sickness Unto Death (16643, existential), Republic (730, philosophy), Man's Search for Meaning (50316, existential).

**Poetry (7):** Dickinson Poems (12242, poetry_sentimental), Whitman Poems (8388, poetry_sentimental), Poe Poems (2148, poetry_existential), Byron Childe Harold (2171, poetry_sentimental), Browning Sonnets (1260, poetry_sentimental), Keats Poems (18855, poetry_sentimental), Leaves of Grass (1322, poetry_sentimental).

Download: `python run_phase1.py`. Full walkthrough: **`RUN.md`** in the project root.

## Candidate books for cluster-based recommendation

- **`candidates_gutenberg.csv`** — Gutenberg ID, title, optional subject. Used by `python recommend_books.py` to recommend **whole books** per K-means cluster (same philosophical branch). You can add rows to discover more books; the script fetches a short text sample per candidate and ranks by similarity to each cluster centroid. This is how you can **use the recommender to find more books to add**: run `recommend_books.py`, then add top candidates to `CORPUS` in `src/download_gutenberg.py` and `BOOK_LABELS` in `src/chunk_books.py`, then re-run Phase 1–2–3.

**Finding candidates by "online" or metadata:** To discover books by style/philosophy (e.g. sentiment or genre), you can build the candidate list from external sources first, then rank by cluster. For example: (1) Use [Project Gutenberg’s catalog](https://www.gutenberg.org/ebooks/) or API (e.g. by subject: "Philosophy", "Stoicism", "Fiction") to get a list of Gutenberg IDs. (2) Add them to `candidates_gutenberg.csv`. (3) Run `recommend_books.py` to see which cluster each candidate matches. That way "online" or metadata gives you the candidate pool; the embedding+cluster similarity gives you the branch of philosophy (cluster) for each book.

## Secondary corpus

- Additional poetry or themed collections can be added to `CORPUS` in `src/download_gutenberg.py` and `BOOK_LABELS` in `src/chunk_books.py`; then re-run Phase 1–2–3.

## Reproducing (with venv)

From project root with your venv activated (`source .venv/bin/activate`):

1. **Option A — download then chunk**
   ```bash
   python run_phase1.py
   ```
   This runs `src/download_gutenberg.py` then `src/chunk_books.py`.

2. **Option B — you already have the .txt files**
   - Put them in `data/raw/` with exact names matching `book_id` in CORPUS (e.g. `brothers_karamazov.txt`, `meditations.txt`, `dickinson_poems.txt`, …). See `src/download_gutenberg.py` for the full list of 21 `book_id` values.
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
