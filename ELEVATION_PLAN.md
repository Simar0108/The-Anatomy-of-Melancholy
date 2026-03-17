# Elevation Plan: From Class Project to Resume-Worthy Experiment

**Goal:** Turn "The Anatomy of Melancholy" into a project that demonstrates **industry-relevant NLP/ML**: modern transformer-based pipelines, retrieval best practices, and proper evaluation—so it holds up in a portfolio or interview.

---

## Is this a good idea?

**Yes.** You already have a clear story (suffering taxonomy, hypotheses, 21-book corpus, two recommenders). Adding (1) stronger transformer usage, (2) retrieval best practices, and (3) real evaluation makes it **distinguishable** from a typical class project and gives you concrete things to talk about ("we use bi-encoder + cross-encoder re-ranking," "we evaluated recommendation with precision@k," "we compared transformer vs. VADER sentiment").

---

## Where you are now vs. where "industry-hot" sits

| Layer | Current | Industry-typical / resume-strong |
|-------|---------|-----------------------------------|
| **Embeddings** | sentence-transformers (MiniLM-L6, 384d) — already a transformer | Larger / better ST model; optional fine-tuning or domain adaptation |
| **Sentiment** | VADER (rule-based) | Transformer-based sentiment (HF pipeline or dedicated model) |
| **Retrieval** | NumPy cosine similarity | FAISS (or similar) index; optional **cross-encoder re-ranker** |
| **Evaluation** | Anecdotal ("existential query → Camus") | Labeled relevance set + precision@k; significance tests for hypotheses |
| **Rigor** | Descriptive stats | One or two inferential checks (e.g. t-test by label); ablations (model, k) |
| **Reproducibility** | RUN.md, run_pipeline.py | Config-driven runs (e.g. YAML); optional experiment log |

You don’t need to change everything. A few **high-signal** upgrades are enough to make the project "elevated."

---

## Recommended upgrades (by impact and difficulty)

### Tier 1 — High impact, moderate effort (do these first)

1. **Upgrade embedding model (drop-in)**  
   - **What:** Use a stronger sentence-transformer, e.g. `all-mpnet-base-v2` (768d), behind a config flag so you can keep MiniLM as baseline.  
   - **Why:** Better semantic quality for clustering and retrieval; "we use state-of-the-art sentence embeddings" is a clear resume line.  
   - **Effort:** Low. Change default model name; re-run Phase 2 (and 3–7 if you want full pipeline). Optionally run a **tiny ablation**: same pipeline with MiniLM vs. mpnet, compare silhouette or a few recommendation examples.  
   - **Difficulty:** ⭐

2. **Transformer-based sentiment (replace VADER)**  
   - **What:** Use a Hugging Face sentiment model (e.g. `cardiffnlp/twitter-roberta-base-sentiment-latest` or `distilbert-base-uncased-finetuned-sst-2-english`) per chunk; keep same volatility/trajectory pipeline.  
   - **Why:** Shows "we use transformers for NLP," not only for embeddings; sentiment in literature is a natural fit.  
   - **Effort:** One new module (e.g. `src/features/sentiment_transformer.py`) + switch in Phase 2; report volatility with both or only transformer.  
   - **Difficulty:** ⭐⭐

3. **Recommendation evaluation (qualitative + small quantitative)**  
   - **What:**  
     - **Qualitative:** In the report, document top-5 books per cluster from `recommend_books.py` and 1–2 sentences on fit (e.g. "Cluster 1 top-5 are all philosophical/classical; aligns with cluster description.").  
     - **Quantitative:** Create a small **relevance set** (e.g. 20–30 query–chunk pairs: query + list of relevant chunk ids). Compute **precision@5** (and optionally recall@5) for the chunk recommender.  
   - **Why:** Turns "our recommender works" into "we evaluated our recommender."  
   - **Effort:** Qualitative = 30 min. Quantitative = 1–2 hours to label 20–30 pairs + one script to compute P@5.  
   - **Difficulty:** ⭐⭐

4. **One statistical test for a hypothesis**  
   - **What:** e.g. Compare mean sentence length (or volatility) for books labeled "stoic" vs "existential" with a **t-test** or **permutation test**; report p-value and one sentence in the report.  
   - **Why:** Shows you do **inference**, not only description.  
   - **Effort:** One small script or notebook section + one paragraph in the report.  
   - **Difficulty:** ⭐

5. **Explicit "suffering" framing (limitation)**  
   - **What:** One sentence in the report: we use book-level labels as proxies for "type of suffering"; our features measure stylistic/lexical similarity, which may only partly align with that construct.  
   - **Why:** Shows critical thinking and clarifies scope.  
   - **Effort:** One sentence.  
   - **Difficulty:** ⭐

**Tier 1 total:** Roughly **3–5 days** of part-time work. Gets you: better embeddings, transformer sentiment, evaluated recommendation, one stat test, and clear framing.

---

### Tier 2 — Strong "industry" signal (next step)

6. **Cross-encoder re-ranker for chunk recommendation**  
   - **What:** Keep current bi-encoder (embed query + corpus → cosine). Retrieve top-K (e.g. 20–50), then **re-rank** those with a **cross-encoder** (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` or similar from sentence-transformers). Return top-5 after re-ranking.  
   - **Why:** Bi-encoder retrieve + cross-encoder re-rank is **standard** in search/RAG; easy to explain in an interview.  
   - **Effort:** One module + wiring in `recommend.py`; optional flag `--rerank`.  
   - **Difficulty:** ⭐⭐

7. **FAISS (or similar) for retrieval**  
   - **What:** Build a FAISS index over chunk embeddings; query returns top-k by inner product (with normalized vectors, same as cosine). Same API as now, scalable to much larger corpora.  
   - **Why:** "We use FAISS for scalable similarity search" is a standard line in retrieval systems.  
   - **Effort:** Add `faiss-cpu` (or `faiss-gpu`); one small module that builds index from `embeddings.npy` and queries it; plug into `query.py` or `recommend.py`.  
   - **Difficulty:** ⭐

8. **Ablation: embedding model**  
   - **What:** Run clustering (and optionally recommendation) with two models (e.g. MiniLM vs. mpnet). Report silhouette, maybe one qualitative comparison ("mpnet separates X and Y more clearly").  
   - **Why:** Shows you think about **model choice** and reproducibility.  
   - **Effort:** Re-run pipeline with different config; one table or short section in the report.  
   - **Difficulty:** ⭐⭐

9. **Short related work / positioning**  
   - **What:** One paragraph in the report: literary NLP, sentiment in philosophy/literature, or clustering of philosophical texts; how this project fits.  
   - **Why:** Frames the project as research-aware.  
   - **Effort:** 30–60 min reading + one paragraph.  
   - **Difficulty:** ⭐

**Tier 2 total:** Another **2–4 days** part-time. Adds re-ranker, FAISS, ablation, and positioning.

---

### Tier 3 — If you want to go further (optional)

10. **Zero-shot "suffering type" with an NLI/classification model**  
    - Use a model (e.g. NLI or zero-shot pipeline) to label a sample of chunks with "stoic / existential / moral" and compare agreement with your clusters or book labels. Shows zero-shot NLU.  
    - **Difficulty:** ⭐⭐⭐

11. **Config-driven experiments (YAML or Hydra)**  
    - All main choices (embedding model, k, chunk size, sentiment model) in a config file; run pipeline with `config=exp1.yaml`. Makes ablations and "we ran 4 configurations" easy to describe.  
    - **Difficulty:** ⭐⭐

12. **Fine-tuned embeddings (domain adaptation)**  
    - Fine-tune a small sentence-transformer on cluster labels (or synthetic labels) so embeddings are "philosophy-aware." High impact but needs a bit of training infrastructure and time.  
    - **Difficulty:** ⭐⭐⭐⭐

---

## Suggested order of implementation

**Phase A (honorable experiment, ~1 week part-time):**  
Tier 1: (1) embedding upgrade + optional ablation, (2) transformer sentiment, (3) recommendation eval (qual + small quant), (4) one stat test, (5) suffering framing.  
→ You can stop here and have a **strong, resume-worthy** project.

**Phase B (industry-hot, +a few days):**  
Tier 2: (6) re-ranker, (7) FAISS, (8) ablation table, (9) related work.  
→ Clearly above typical class projects; good for portfolio and interviews.

**Phase C (optional):**  
Tier 3 items if you have time and want to go deeper.

---

## Resume / portfolio phrasing (after elevation)

- **Before:** "Built a taxonomy of philosophical texts using embeddings and clustering; implemented a recommendation system."
- **After (example):** "Designed and evaluated an NLP pipeline for philosophical literature: transformer-based embeddings (sentence-transformers) and sentiment, K-means clustering over 21 books (1.4k chunks), bi-encoder retrieval with optional cross-encoder re-ranking and FAISS. Evaluated recommendations with precision@k; reported significance tests for stylistic hypotheses. Tech: Python, Hugging Face, sentence-transformers, FAISS, scikit-learn."

---

## Difficulty summary

| Tier | Main items | Relative difficulty | Time (rough) |
|------|------------|----------------------|--------------|
| 1 | Embedding upgrade, transformer sentiment, rec eval, 1 stat test, framing | Moderate | 3–5 days part-time |
| 2 | Re-ranker, FAISS, ablation, related work | Moderate | 2–4 days part-time |
| 3 | Zero-shot, config-driven, fine-tuning | Higher | 1–2 weeks+ |

**Bottom line:** Going "full out" on **Tier 1 + Tier 2** is a **good idea** and **doable** in about 1–2 weeks of part-time work. It would make the project clearly resume-worthy and show you understand modern NLP architectures and methodology without rebuilding the whole pipeline.

If you want to proceed, a practical order is: **embedding upgrade → transformer sentiment → recommendation evaluation → one stat test → re-ranker → FAISS**, then polish the report (ablation, related work, framing).
