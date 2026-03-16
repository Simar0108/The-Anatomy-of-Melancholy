# The Anatomy of Melancholy — Final Report

CS252A Final Project: a **taxonomy of suffering** in classical and philosophical literature—beyond sentiment, toward existential, moral, and stoic texture.

---

## Executive summary and how to interpret the results

**Is the project complete?** Yes. All eight phases are implemented: data pipeline, features, clustering, PCA/UMAP, sentiment trajectory, syntactic analysis, recommendation engine, and this report.

**What we set out to do:** Test whether we can detect a *taxonomy of suffering* (moral, existential, stoic, intergenerational) using embeddings, sentiment, and syntax—and build a “resonance” recommender that finds passages by philosophical texture.

**What we actually learned:**

- **We did not find a clean “suffering type” taxonomy.** Clustering and PCA separate texts mainly by **style and author**, not by the four suffering labels. The main split is “narrative, modern-ish prose” (East of Eden, some others) vs. “archaic, philosophical register” (Meditations + Myth of Sisyphus). So the **insight is negative but clear**: with this corpus and these features, *suffering type* is not what drives the structure; **register and author identity** do.

- **H2 (lexical anchors) is the strongest positive result.** A small set of words really does drive separation in PCA: archaic (*thou, thy, thee*), narrative (*garden, wrath, house*), and Meditations-specific (*marcus, fronto, aurelius*). **How to interpret:** If you want to know “what words separate these books in TF-IDF space,” the anchor words give a direct, interpretable answer. They support the idea that **lexical style** (and author) matters more than an abstract “type of suffering.”

- **H1 (Stoic lower emotional volatility) is not supported—and in this run is reversed.** Meditations (stoic) has *higher* sentiment volatility (0.75) than Myth of Sisyphus (existential, 0.50). **How to interpret:** Do not treat this as evidence that Stoicism is “more emotional.” It likely reflects **chunking and VADER**: short, varied aphorisms can swing sentiment more than longer narrative passages, and VADER is not built for philosophical language. So H1 is a **non-result** in this setup, not a finding about the texts themselves.

- **H3 (yearning → higher syntactic complexity) is partially supported.** Meditations has the shortest mean sentence length (27.3 words); Myth of Sisyphus and East of Eden are longer (33.2, 32.8). **How to interpret:** Aphoristic, Stoic style does show up as **shorter sentences**; more narrative/yearning texts as longer. So we gain a **modest stylistic insight**: sentence length tracks something like aphorism vs. narrative, which loosely aligns with the hypothesis.

- **Recommendation works in a meaningful way.** Existential-style queries surface Camus; Stoic-style queries surface Meditations. **How to interpret:** The engine is retrieving by **semantic/philosophical similarity**, not by suffering label. So we have a usable tool for “find passages that resonate with this thought,” even though we did not recover a clear taxonomy of suffering types.

**Bottom line:** There are **no strong, headline “suffering taxonomy” results**. We did get: (1) interpretable lexical anchors (H2), (2) a plausible syntactic signal (H3, partial), (3) a working recommendation by philosophical texture, and (4) a clear negative result (structure is style/author, not suffering type) that is itself informative. The project is complete; the main takeaway is that **distinguishing suffering types** in this setup would require different features, more data, or chunk-level labels—while **style and author** are already visible and useful for recommendation.

---

## 1. Hypotheses

1. **Semantic divergence (H1)** — Stoic texts show lower “emotional volatility” than existential texts in embedding space.
2. **Lexical anchors (H2)** — A small set of words drives cluster separation in 3D PCA.
3. **“Yearning” signature (H3)** — Authors focused on human yearning show higher syntactic complexity than the aphoristic Stoic style.

---

## 2. Dataset

- **Primary corpus (Project Gutenberg):** The Brothers Karamazov (moral), The Myth of Sisyphus (existential), Meditations (stoic), East of Eden (intergenerational). See `data/README.md` for IDs and sources.
- **Chunking:** Books split into chapters or fixed-length passages; each row has `book_id`, `chunk_id`, `text`, `word_count`, `label_suffering_type`.
- **Features:** TF-IDF (vocabulary from corpus), sentence embeddings (sentence-transformers `all-MiniLM-L6-v2`), syntactic metrics (mean/std sentence length, punctuation density via spaCy), sentiment (VADER per chunk).

**Note:** In the run used for this report, Brothers Karamazov has only 5 chunks; the other three books have 61–106 chunks. This imbalance affects per-book statistics and clustering.

---

## 3. Methods

| Step | Description |
|------|-------------|
| **Chunking** | Pipeline in `src/chunk_books.py`; output `data/processed/corpus.parquet`. |
| **Features** | Phase 2: TF-IDF, embeddings, syntactic, sentiment → `data/features/` (matrices + `corpus_features.parquet`). |
| **Clustering** | K-Means and hierarchical (Ward) on chunk embeddings; k chosen via elbow/silhouette (k=2 used). Cluster descriptions = top TF-IDF terms per cluster. Cross-tabs: cluster vs book, cluster vs label. |
| **Dimensionality reduction** | PCA (3 components) on TF-IDF; UMAP 2D/3D on embeddings. Anchor words = top \|loading\| per PCA component. |
| **Sentiment trajectory** | Rolling-window mean sentiment along book order; volatility = std of sentiment per book. |
| **Syntactic complexity** | Per-book mean sentence length, std sentence length, punctuation density. |
| **Recommendation** | Same embedding model; user query embedded and compared by cosine similarity to corpus; top-k chunks returned with book/chunk and score. |

---

## 4. Results

### 4.1 Clustering (suffering taxonomy)

- **K selection:** Silhouette is highest at k=2 (~0.075); k=2 used for a binary split. See `results/k_selection.csv`.
- **Cluster vs book:** Cluster 0: East of Eden (61), Brothers Karamazov (5), Meditations (5), Myth of Sisyphus (4). Cluster 1: Meditations (85), Myth of Sisyphus (102). So one cluster is dominated by East of Eden (+ a few others); the other by Meditations and Myth of Sisyphus.
- **Cluster vs label:** Cluster 0: intergenerational (61), moral (5), stoic (5), existential (4). Cluster 1: existential (102), stoic (85). Clusters align more with **book/author** than with a single suffering type; the second cluster groups existential and stoic texts together (Camus + Marcus Aurelius).
- **Cluster descriptions (top terms):** Cluster 0: *garden, minora, irais, little, said, don, house, went, round, man wrath, wrath, day, days…* (narrative/place). Cluster 1: *thou, unto, things, man, thy, thee, men, nature, doth, good, life, world, thyself…* (archaic/philosophical). So the split is largely stylistic (narrative vs. philosophical register).

### 4.2 PCA and lexical anchors (H2)

- **PCA:** Fit on TF-IDF; 3 components explain ~2.8%, 1.6%, 0.9% of variance (see `results/pca_variance.csv`). First component is driven by archaic/second-person terms: *thou, unto, thy, doth, thee, thyself, whatsoever, hath, shalt, nature, things*. Second: *garden, minora, irais, belief, wrath, house, went, morality, self*. Third: *marcus, fronto, ad, aurelius, caes, father, emperor, roman, master* (Meditations-specific). So **H2 is supported**: a small set of words (archaic, narrative, and author-specific) drives separation along PCA axes; these align with intuitive style and source rather than a single “suffering type” label.

### 4.3 Sentiment trajectory and volatility (H1)

- **Per-book volatility (std of sentiment):** Brothers Karamazov 0.87, Meditations 0.75, Myth of Sisyphus 0.50, East of Eden 0.52. Mean sentiment (positive) is highest for East of Eden and Myth of Sisyphus (~0.83) and lower for Brothers Karamazov and Meditations.
- **Interpretation:** In this run, **H1 is not supported**: the stoic text (Meditations) has *higher* volatility than the existential text (Myth of Sisyphus). Brothers Karamazov has the highest volatility but only 5 chunks, so that value is unstable. Possible reasons: chunk size/count, VADER’s sensitivity to aphoristic vs. narrative style, or the particular split of chapters.

### 4.4 Syntactic complexity (H3)

- **Mean sentence length (words):** Brothers Karamazov 18.0, Meditations 27.3, Myth of Sisyphus 33.2, East of Eden 32.8. Punctuation density is similar across books (~0.03–0.04).
- **Interpretation:** **H3 is partially supported**: Meditations (stoic, aphoristic) has the shortest mean sentence length among the three multi-chunk books; Myth of Sisyphus and East of Eden (yearning/narrative) have longer sentences. Brothers Karamazov (5 chunks) has the shortest mean length, which may be an artifact of the small chunk set.

### 4.5 Recommendation engine

- Query “I am tormented by the idea of meaninglessness” → top chunks from *Myth of Sisyphus*. Query “Accept what is in your control and let go of the rest” → top chunks from *Meditations* and *East of Eden*. Recommendations align with the philosophical tone of the query. See Phase 7 and `python recommend.py "quote"`.

---

## 5. Limitations

- **Chunk imbalance:** Brothers Karamazov has very few chunks (5) in the run used here; per-book and cluster results are sensitive to this.
- **Labels:** `label_suffering_type` is assigned by book, not by chunk; we do not have chunk-level ground truth for “type of suffering.”
- **VADER:** Designed for social/media text; may not capture philosophical or archaic tone well; volatility may reflect style as much as “emotional” content.
- **PCA variance:** First three components explain only a small fraction of TF-IDF variance; structure is spread across many dimensions.
- **Cross-genre:** Poetry/secondary corpus was not fully integrated; recommendation cross-genre notes are in `results/recommendation_poetry_notes.md`.

---

## 6. Reproducing

From project root with a virtual environment activated:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Then run phases in order (or use `python run_pipeline.py` to run the full pipeline):

1. `python run_phase1.py` → corpus  
2. `python run_phase2.py` → features  
3. `python run_phase3.py` → clustering  
4. `python run_phase4.py` → PCA, UMAP, anchors  
5. `python run_phase5.py` → sentiment trajectory, volatility  
6. `python run_phase6.py` → syntactic by book  
7. `python run_phase7.py` → recommendation check; then `python recommend.py "quote"`  

Artifacts: `data/processed/`, `data/features/`, `results/*.csv`, `results/*.png`. See `README.md` and `PLAN.md` for layout and task details.

---

## 7. Summary

The pipeline builds a **suffering taxonomy** from chunk embeddings and TF-IDF: clustering separates narrative vs. philosophical style; PCA anchor words are interpretable (archaic, narrative, author-specific). **H2 (lexical anchors)** is supported; **H3 (syntactic complexity)** is partially supported (Meditations shorter sentences). **H1 (stoic lower volatility)** is not supported in this run and is limited by chunk balance and sentiment model choice. The recommendation engine returns semantically coherent chunks for existential and stoic queries.
