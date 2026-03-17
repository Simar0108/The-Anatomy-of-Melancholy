# Execution Plan: Tier 1 + Tier 2 Elevation

Order of implementation and dependencies. Work through in order; each step is testable.

---

## Phase A: Tier 1

### A1. Config + embeddable model choice
- Add optional `config/default.yaml` with `embedding_model` and `sentiment_backend`.
- `run_phase2.py`: CLI flags `--embedding-model`, `--sentiment` (vader | transformer). Default embedding remains `all-MiniLM-L6-v2`; add option `all-mpnet-base-v2`.
- **Deliverable:** `python run_phase2.py --embedding-model all-mpnet-base-v2` runs and writes `embeddings_meta.txt` with that model.

### A2. Transformer sentiment
- Add `src/features/sentiment_transformer.py`: Hugging Face pipeline (e.g. `distilbert-base-uncased-finetuned-sst-2-english`), output columns compatible with existing `sentiment_compound`, `sentiment_pos`, `sentiment_neg`, `sentiment_neu`.
- `run_phase2.py`: when `--sentiment transformer`, call sentiment_transformer.run; else existing VADER. Save which backend was used in features dir (e.g. `sentiment_meta.txt`).
- **Deliverable:** `python run_phase2.py --sentiment transformer` produces sentiment.parquet from transformer; Phase 5 still runs.

### A3. Recommendation evaluation
- Define relevance set format: `data/relevance_set.csv` with columns `query_id`, `query_text`, `chunk_index` (or `book_id`, `chunk_id`). Add a small example set (5–10 rows) as template.
- Add `scripts/eval_recommendation.py`: load relevance set, for each query run `recommend(..., k=5)`, compute precision@5 (and optionally recall@5), print and optionally write `results/eval_recommendation.json`.
- **Deliverable:** Run `python scripts/eval_recommendation.py` and get P@5 reported; document in report how to add more rows for full eval.

### A4. Statistical test
- Add `scripts/stats_hypothesis.py`: load `results/volatility_by_book.csv` or `syntactic_by_book.csv`, group by `label_suffering_type` (e.g. stoic vs existential), run scipy.stats.ttest_ind or permutation test, print p-value and effect; optionally write `results/stats_hypothesis.txt`.
- **Deliverable:** One sentence in report: "Difference in mean sentence length (stoic vs existential) is [significant/not significant] (p=...)."

### A5. Report: framing + evaluation section
- Add one sentence in Limitations or Intro: we use book-level labels as proxies for "suffering"; features measure stylistic/lexical similarity.
- Add short "Evaluation" subsection: qualitative (top-5 per cluster from recommend_books) + quantitative (P@5 from relevance set, how to reproduce).

---

## Phase B: Tier 2

### B1. Cross-encoder re-ranker
- Add `src/recommendation/rerank.py`: load cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`), score (query, chunk) pairs for top-K candidates, return reordered indices.
- `query.py`: add `recommend(..., rerank=True, rerank_top_k=20)`; when True, retrieve 20 with bi-encoder, re-rank with cross-encoder, return top 5.
- `recommend.py`: add `--rerank` flag.
- **Deliverable:** `python recommend.py "quote" --rerank` returns top-5 after re-ranking.

### B2. FAISS retrieval
- Add `faiss-cpu` to requirements. Add `src/recommendation/faiss_index.py`: build FAISS IndexFlatIP from embeddings.npy (normalized), query returns top-k indices. Same API as numpy path.
- `query.py` or `embed_index.py`: when FAISS index exists (e.g. `embeddings.faiss` or flag), use FAISS for retrieval; else numpy. Optional: `run_phase2.py` or a small script builds the FAISS index after saving embeddings.
- **Deliverable:** Retrieval can use FAISS; CLI or env to toggle; document in RUN.md.

### B3. Ablation + report table
- Document in RUN.md or a short ABLATION.md: run Phase 2 with `--embedding-model all-MiniLM-L6-v2`, then with `all-mpnet-base-v2`; re-run Phase 3; compare silhouette and/or one qualitative recommendation example. Add empty table in report (Model | Silhouette | Notes) for reader to fill or we fill with placeholder.
- **Deliverable:** Instructions to run ablation; report section "Ablation (embedding model)".

### B4. Related work
- Add one paragraph in report (before or after Methods): prior work on literary NLP, sentiment in philosophy/literature, or clustering of texts; how this project relates.
- **Deliverable:** Report has "Related work" or "Positioning" subsection.

---

## File checklist

| Item | File(s) |
|------|--------|
| Config | `config/default.yaml` (optional) |
| Embedding option | `run_phase2.py` (--embedding-model), `embeddings.py` (already has model arg) |
| Transformer sentiment | `src/features/sentiment_transformer.py`, `run_phase2.py` (--sentiment) |
| Eval | `data/relevance_set.csv`, `scripts/eval_recommendation.py` |
| Stats | `scripts/stats_hypothesis.py` |
| Re-ranker | `src/recommendation/rerank.py`, `query.py`, `recommend.py` |
| FAISS | `src/recommendation/faiss_index.py`, `query.py` or embed_index, requirements.txt |
| Report | framing sentence, Evaluation subsection, Ablation table, Related work |
| Docs | RUN.md (FAISS, ablation), ELEVATION_PLAN.md (done) |

---

## Token access (Hugging Face)

- For **sentence-transformers** and **transformers** (sentiment, cross-encoder), public models work without a token. Rate limits may apply; if you hit them, set `HF_TOKEN` (create at huggingface.co/settings/tokens).
- No token required to complete this plan; add token only if you see rate-limit errors.
