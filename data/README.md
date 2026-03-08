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

## Reproducing

1. Run the download script/notebook (Phase 1.3).
2. Run the chunking pipeline (Phase 1.4).
3. Output: `processed/corpus.parquet`.
