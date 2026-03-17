# Technical Breakdown: What We Do, Why, and How

A single place that explains the pipeline in plain language, the technical choices, and where the novelty (or lack of it) lies.

---

## One command to run everything

From the project root with your venv activated and dependencies installed:

```bash
python run_all.py
```

This runs, in order, only what’s needed:

1. **Phase 1** — Download 21 books from Gutenberg and chunk them (if `data/processed/corpus.parquet` is missing).
2. **Phase 2** — Extract features: TF-IDF, embeddings, syntactic, sentiment (if `data/features/embeddings.npy` is missing).
3. **Phases 3–7** — Clustering, PCA/UMAP, sentiment trajectory, syntactic by book, recommendation check.
4. **FAISS** — Build the FAISS index for fast retrieval (if embeddings exist and the index is missing).
5. **Eval** — Compute recommendation P@k from `data/relevance_set.csv` (if that file and features exist).
6. **Stats** — Run the hypothesis test (stoic vs. existential) if volatility or syntactic results exist.

So: one command runs the full pipeline and post-processing, creating only what’s missing. Optional flags:

- `python run_all.py --through 4` — Stop after phase 4.
- `python run_all.py --skip-post` — Skip FAISS, eval, and stats.
- `python run_all.py --phase2-args "--embedding-model all-mpnet-base-v2 --sentiment transformer"` — Use a stronger embedding model and transformer sentiment in Phase 2.

---

## What we’re doing (high level)

We’re building a **literary/philosophical taxonomy** and **recommendation system**:

1. **Corpus** — 21 books (philosophy, novels, poetry) from Project Gutenberg, chunked into ~1.4k passages.
2. **Features** — Each passage gets: a vector (sentence embedding), TF-IDF counts, syntactic stats (sentence length, punctuation), and sentiment (VADER or a transformer).
3. **Taxonomy** — We cluster passages by embedding similarity (K-means) and reduce dimensions (PCA on TF-IDF, UMAP on embeddings) to see how books group. We then test three hypotheses: (H1) stoic texts have lower “emotional volatility,” (H2) a small set of words drives separation, (H3) “yearning” texts have higher syntactic complexity.
4. **Recommendation** — Two modes: (a) **chunk-level**: you type a quote, we return the most similar passages (bi-encoder; optional cross-encoder re-rank; optional FAISS). (b) **Cluster-based books**: we recommend whole books (from a Gutenberg candidate list) that match each cluster’s style.
5. **Evaluation** — We measure recommendation quality with precision@k on a small relevance set and run a simple statistical test (e.g. stoic vs. existential sentence length or volatility).

So we’re doing: **ingest → represent → cluster → visualize → test hypotheses → recommend → evaluate**.

---

## Why we’re doing it

- **Research question:** Can we recover a “taxonomy of suffering” (moral, existential, stoic, etc.) from text with off-the-shelf NLP? We use book-level labels as *proxies* for that idea; our features are stylistic and lexical. The answer in our runs: **not as a clean taxonomy** — we get style and register (e.g. archaic philosophical vs. narrative). That negative result is still informative.
- **Product-style goal:** A tool that finds “passages that resonate” with a thought and “books in the same philosophical branch” as a cluster. That’s useful for readers and for expanding the corpus.
- **Portfolio goal:** Show a full NLP pipeline: embeddings, clustering, retrieval (bi-encoder + re-ranker + FAISS), transformer sentiment, and basic evaluation and stats.

---

## How we do it (technical architecture)

### Data flow

```
Gutenberg (21 books)
    → chunk (chapters / fixed-length)
    → corpus.parquet (rows: book_id, chunk_id, text, label_suffering_type)
    → features:
         - TF-IDF (vocabulary from corpus) → matrix + vocab
         - sentence-transformers (MiniLM or mpnet) → embeddings.npy (L2-normalized)
         - spaCy → syntactic.parquet (mean/std sentence length, punctuation)
         - VADER or DistilBERT SST-2 → sentiment.parquet
    → corpus_features.parquet (corpus + syntactic + sentiment; row index = chunk index)
```

### Clustering and viz

- **K-means** on the embedding matrix (row = chunk). We choose k by silhouette (e.g. k=3). Labels go to `labels_kmeans.csv`.
- **Hierarchical** (Ward) on the same matrix; we can cut to k clusters for comparison.
- **PCA** on the TF-IDF matrix (3 components); top |loadings| per component → “anchor words” (interpretable axes).
- **UMAP** on embeddings (2D and 3D) for scatter plots by cluster and by book.

So: **same chunks, multiple views** — TF-IDF for lexical structure, embeddings for semantic structure.

### Recommendation

- **Chunk recommender:**  
  - Encode query with the same sentence-transformer → vector q.  
  - Similarity = dot(embeddings, q) (cosine, since vectors are normalized).  
  - Top-k indices = best chunks. Optionally: retrieve top-20, then **re-rank** with a cross-encoder (query, chunk) and return top-5.  
  - If `embeddings.faiss` exists, we use **FAISS** (IndexFlatIP) instead of a raw dot product for speed.

- **Cluster-based book recommender:**  
  - Compute **cluster centroids** (mean embedding per cluster, L2-normalized).  
  - For each candidate book (from a CSV of Gutenberg IDs), fetch a short sample, embed it, and score similarity to each centroid.  
  - For each cluster, return the top-N candidate books. So “books in this philosophical branch” = books whose sample is closest to that cluster’s centroid.

### Evaluation and stats

- **Relevance set:** CSV with query_id, query_text, and relevant_chunk_indices (corpus row indices). We run the chunk recommender for each query, get top-k chunk indices, and compute **precision@k** (and optionally recall@k).
- **Hypothesis test:** We take per-book metrics (e.g. mean sentence length or volatility), group by label (e.g. stoic vs. existential), and run a **t-test** (or permutation test). Result goes into the report (e.g. “difference significant at α=0.05 or not”).

---

## Novelty: has this been done before?

**Short answer:** The **individual pieces** are standard (sentence-transformers, K-means, PCA/UMAP, bi-encoder retrieval, cross-encoder re-ranker, FAISS, sentiment). The **combination** — philosophical/literary taxonomy + “suffering” framing + cluster-based book recommendation from Gutenberg + hypothesis-driven analysis (volatility, syntax) — is a specific application that isn’t a well-known template. So there is **application-level novelty** (what we study and how we frame it), but **method novelty** is limited (we use established architectures).

### What is novel or uncommon

1. **Framing** — Treating “taxonomy of suffering” (moral, existential, stoic, intergenerational) as something to test with embeddings and clustering is a clear, interpretable research angle. Many NLP projects cluster by genre or author; fewer explicitly test whether “suffering type” is recoverable and report that it’s largely **not** (style dominates).
2. **Cluster-based book recommendation** — Using cluster centroids to recommend **whole books** from a Gutenberg candidate list (and to suggest what to add to the study) is a concrete, reproducible workflow. It’s a simple but clear use of the taxonomy.
3. **Pipeline completeness** — One repo: data → features (with transformer options) → clustering → viz → trajectory → syntax → two recommenders → evaluation (P@k) and a statistical test. That’s a full “experiment” rather than a single script.

4. **Cluster–label agreement (NMI/ARI)** — We measure how well unsupervised clusters align with book-level labels (`scripts/cluster_label_agreement.py`). Low agreement supports that style, not "suffering type," drives the taxonomy.
5. **Zero-shot check (no fine-tuning)** — We run a pre-trained zero-shot classifier on chunks and compare to book labels and K-means clusters (`scripts/zero_shot_eval.py`). No training or LoRA.

### What is not novel

- Sentence embeddings, K-means, PCA, UMAP, bi-encoder + cross-encoder re-ranker, FAISS, and transformer sentiment are all standard in NLP and search.
- Clustering literary or philosophical text by embedding similarity has been done in digital humanities and NLP; we’re applying it to a particular corpus and question.

---

## Ways to add more novelty

If you want to push the project toward “something that hasn’t been done before” or “clearly novel,” here are concrete directions:

1. **Domain-adapted embeddings**  
   - **Idea:** Fine-tune a small sentence-transformer on (passage, cluster_id) or (passage, label) so that the embedding space is “philosophy-aware” or “suffering-type-aware.”  
   - **Why novel:** Most work uses off-the-shelf embeddings; fine-tuning for a literary/philosophical taxonomy is a clear, publishable step.  
   - **How:** Collect or synthesize labels (e.g. use cluster assignments as silver labels), then use sentence-transformers’ fine-tuning API (e.g. MultipleNegativesRankingLoss) on (anchor, positive, negatives). Re-run clustering and recommendation with the new embeddings.

2. **Zero-shot “suffering type” and agreement with clusters**  
   - **Idea:** Use an NLI or zero-shot classification model (e.g. “This passage is about [stoic / existential / moral] suffering”) to assign a label per chunk. Compare agreement between these labels and K-means clusters (or book-level labels).  
   - **Why novel:** It directly tests “does the model’s notion of suffering type align with our clusters?” and can be written up as a small experiment.  
   - **How:** Hugging Face pipeline for zero-shot classification; compute accuracy or NMI between zero-shot labels and cluster ids (or label per book).

3. **Cross-genre systematic comparison**  
   - **Idea:** Treat poetry vs. prose as a factor: do poetry chunks sit in different clusters? Do volatility and syntactic complexity differ by genre when controlling for “suffering” label?  
   - **Why novel:** Many literary NLP projects stay within one genre; an explicit cross-genre (prose + poetry) analysis with the same pipeline is a clear angle.  
   - **How:** Add a “genre” column (poetry vs. prose); cross-tab cluster vs. genre; run volatility/syntax by genre (and optionally by label within genre).

4. **Releasing a benchmark**  
   - **Idea:** Release the 21-book corpus (or a subset) with book-level labels, plus the relevance set (query + relevant chunk indices), and report baseline P@k.  
   - **Why novel:** Reproducible benchmarks for “philosophical/literary retrieval” or “suffering-type taxonomy” are scarce; this would be a small but citable contribution.  
   - **How:** Clean and document the corpus and relevance set; add a short data statement; put baseline numbers and run_all.py in the repo.

5. **Human evaluation of recommendations**  
   - **Idea:** Have a few people rate “is this passage relevant to this query?” for a sample of (query, recommended chunk) pairs. Report precision@k with human judgments and compare to the relevance set.  
   - **Why novel:** It grounds the system in human judgment and strengthens the “evaluation” section.  
   - **How:** Export a sample (e.g. 50 pairs); simple form or spreadsheet; aggregate binary relevance; compute P@k and optionally compare to automatic relevance set.

6. **Contrastive or interpretable clustering**  
   - **Idea:** Use a method that encourages clusters to align with labels (e.g. constrained clustering, or a loss that pulls same-label chunks together) and compare to unsupervised K-means.  
   - **Why novel:** It directly addresses “can we get a taxonomy that matches our labels?” with a different method.  
   - **How:** E.g. scikit-learn’s constrained K-means (if you have partial labels) or a simple contrastive loss on top of embeddings.

---

## Summary table

| Layer        | What we do                          | Why                               | Novel?                          |
|-------------|--------------------------------------|-----------------------------------|----------------------------------|
| Data        | 21 books, chunked, book-level labels | Diverse, reproducible corpus      | Application (Gutenberg + labels) |
| Embeddings  | Sentence-transformers (MiniLM/mpnet)| Semantic similarity               | No (standard)                    |
| Sentiment   | VADER or DistilBERT SST-2            | Volatility and trajectory         | No (transformer option is standard) |
| Clustering  | K-means (+ hierarchical)            | Taxonomy over passages            | No                              |
| PCA / UMAP  | TF-IDF PCA; embedding UMAP          | Interpretable axes and viz        | No                              |
| Retrieval   | Bi-encoder + optional re-ranker + FAISS | Scalable, accurate retrieval  | No                              |
| Book rec    | Centroids + candidate list           | “Books like this cluster”         | Application-level (workflow)    |
| Eval        | P@k, t-test                          | Evidence for quality and hypotheses | Standard practice              |
| Framing     | “Taxonomy of suffering”              | Clear research question           | Yes (application)                |

So: **we have application novelty** (framing, corpus, cluster-based book rec, full pipeline) and **we use standard methods**. To add **method or benchmark novelty**, the list above (fine-tuning, zero-shot, cross-genre, benchmark release, human eval, constrained clustering) gives concrete options.

---

## Single run command (recap)

```bash
# From project root, with venv active and deps installed:
python run_all.py
```

This runs the full pipeline and post-processing in one go, only executing steps whose prerequisites are missing or whose outputs don’t yet exist.
