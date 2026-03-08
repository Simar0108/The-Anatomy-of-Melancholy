# Data

## Layout

- **`raw/`** — Downloaded full texts from Project Gutenberg and any poetry files. Not committed (add to `.gitignore` if large).
- **`processed/`** — Chunked corpus: `corpus.parquet` (or `.csv`) with columns:
  - `book_id` — e.g. `brothers_karamazov`, `myth_of_sisyphus`, `meditations`, `east_of_eden`
  - `chunk_id` — chapter or passage index
  - `text` — full text of the chunk
  - `word_count` — optional
  - `label_suffering_type` — optional (for evaluation): moral, existential, stoic, intergenerational
- **`features/`** — Optional: saved TF-IDF matrix, embedding matrix, or paths to them.

## Primary corpus (Project Gutenberg)

| Book | Gutenberg ID | Intended label |
|------|--------------|----------------|
| The Brothers Karamazov (Dostoevsky) | 28054 | moral |
| The Myth of Sisyphus (Camus) | 52881 | existential |
| Meditations (Marcus Aurelius) | 2680 | stoic |
| East of Eden (Steinbeck) | 1327 | intergenerational |

Download from: `https://www.gutenberg.org/files/<id>/<id>-0.txt` or use the `gutenberg` package.

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
   - Put them in `data/raw/` with these exact names:
     - `brothers_karamazov.txt`
     - `myth_of_sisyphus.txt`
     - `meditations.txt`
     - `east_of_eden.txt`
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
