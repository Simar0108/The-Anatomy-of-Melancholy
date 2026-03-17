# Rework: Better, More Interesting Results

This doc describes changes made to get **stronger and more interpretable results** without changing the core pipeline.

---

## 1. What Was Wrong

- **P@5/R@5 = 1.0** was a sanity check (relevance set = model’s own top-5), not a real evaluation.
- **Book-level stats** (n=3 stoic vs n=6 existential) had almost no power → p-values ~0.30 and 0.98, so we couldn’t claim significance.
- **Cluster–label agreement** (NMI/ARI) was low, which is correct (clusters = style, not suffering type) but felt “negative.”

---

## 2. What We Added

### 2.1 Chunk-level hypothesis tests (more power)

- **Script:** `scripts/stats_chunk_level.py`
- **Idea:** Compare **stoic vs existential chunks** (not books). N ≈ 135 stoic chunks vs 600+ existential chunks → much more power.
- **Metrics:** `mean_sent_len` (and optional sentiment `compound`) per chunk from `corpus_features.parquet`.
- **Output:** `results/stats_chunk_level.txt` — t-test and “Significant at alpha=0.05: Yes/No.”
- **Run:** `python scripts/stats_chunk_level.py` or via `run_all.py`.

**Why it helps:** We can now test H3 (sentence length) at chunk level; a significant p-value is plausible and gives a clear, reportable result.

### 2.2 Recommendation eval by “expected book” (no human labels)

- **Script:** `scripts/eval_recommendation_by_book.py`
- **Data:** `data/eval_queries_by_book.csv` — columns `query_id`, `query_text`, `expected_book_id` (e.g. “myth_of_sisyphus” for the meaninglessness query).
- **Metric:** For each query, get top-5 recommendations; **P@5 = fraction of top-5 that come from the expected book**. Also **Recall@book = fraction of queries with ≥1 hit** from the expected book.
- **Output:** `results/eval_recommendation_by_book.json`.

**Why it helps:** You get a **meaningful automatic metric** (“Does the recommender return passages from the right book?”) without labeling individual chunks. Good for the report and for ablation (e.g. with/without reranker).

### 2.3 Pipeline integration

- `run_all.py` now runs:
  - **Eval by expected book** (if `data/eval_queries_by_book.csv` exists).
  - **Chunk-level stats** (if `corpus_features.parquet` exists).

---

## 3. How to Reframe the Report

### Narrative shift

1. **Lead with what we find, not what we wanted**
   - “We set out to detect a *taxonomy of suffering*; the structure we recover is instead a **taxonomy of style and register** (archaic vs narrative, aphoristic vs expansive). That is a clear and useful result.”

2. **H2 (lexical anchors)**  
   - Keep as a **positive result**: interpretable words drive PCA separation.

3. **H3 (sentence length / “yearning”)**  
   - **Book-level:** “Partially supported” as a stylistic pattern; report that the **chunk-level** test (many more observations) gives a testable claim and report p-value from `stats_chunk_level.txt` (significant or not).

4. **H1 (volatility)**  
   - Be honest: at **book level** the group means are almost identical (0.53 vs 0.54). The interesting pattern is **which books** are high vs low (style/format). Optionally add chunk-level sentiment comparison in `stats_chunk_level.py` and report that.

5. **Recommendation**  
   - **Qualitative:** “Existential queries → Camus; stoic → Meditations” (keep examples).  
   - **Quantitative:** Report **P@5 from expected book** and **Recall@book** from `eval_recommendation_by_book.json` as the main automatic eval. Mention that human-labeled P@k would be the gold standard for future work.

6. **Clusters ≠ suffering type**  
   - Frame as the **main negative result**: “Clusters separate by style and author, not by suffering type; NMI/ARI with book labels are low. That tells us what the embedding space is actually capturing.”

---

## 4. Quick Commands

```bash
# Chunk-level stats only
python scripts/stats_chunk_level.py

# Eval by expected book only
python scripts/eval_recommendation_by_book.py

# Full pipeline (includes both)
python run_all.py
```

Edit `data/eval_queries_by_book.csv` to add queries or change expected books. Chunk-level groups can be changed with `--groups stoic existential philosophy`.

---

## 5. Optional Next Steps

- **Cross-encoder ablation:** Run eval by expected book with and without `--rerank` and report P@5/Recall@book for both.
- **More queries:** Add 2–3 queries to `eval_queries_by_book.csv` (e.g. moral, intergenerational) for a richer eval.
- **Report sentence:** Add one short “Rework” or “Revised analysis” subsection that points to chunk-level stats and eval-by-book, and reframes the narrative as above.
