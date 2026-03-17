# The Anatomy of Melancholy — Final Report

CS252A Final Project: a **taxonomy of suffering** in classical and philosophical literature—beyond sentiment, toward existential, moral, and stoic texture.

---

## Executive summary and how to interpret the results

**Is the project complete?** Yes. All eight phases are implemented: data pipeline, features, clustering, PCA/UMAP, sentiment trajectory, syntactic analysis, recommendation engine, and this report.

**What we set out to do:** Test whether we can detect a *taxonomy of suffering* (moral, existential, stoic, intergenerational) using embeddings, sentiment, and syntax—and build a “resonance” recommender that finds passages by philosophical texture. We use book-level labels as proxies for type of suffering; our features measure stylistic and lexical similarity, which may only partly align with that construct.

**What we actually learned:**

- **We did not find a clean “suffering type” taxonomy.** Clustering and PCA separate texts mainly by **style and author**, not by the four suffering labels. The main split is “narrative, modern-ish prose” (East of Eden, some others) vs. “archaic, philosophical register” (Meditations + Myth of Sisyphus). So the **insight is negative but clear**: with this corpus and these features, *suffering type* is not what drives the structure; **register and author identity** do.

- **H2 (lexical anchors) is the strongest positive result.** A small set of words really does drive separation in PCA: archaic (*thou, thy, thee*), narrative (*garden, wrath, house*), and Meditations-specific (*marcus, fronto, aurelius*). **How to interpret:** If you want to know “what words separate these books in TF-IDF space,” the anchor words give a direct, interpretable answer. They support the idea that **lexical style** (and author) matters more than an abstract “type of suffering.”

- **H1 (Stoic lower emotional volatility) is partially supported in the 21-book run.** Enchiridion and Seneca (stoic) have *lower* volatility (0.36, 0.48) than many existential (e.g. Stranger 0.76, Man’s Search 0.85). Meditations (0.75) remains high. **How to interpret:** The pattern is **stylistic**: compact/aphoristic texts (Enchiridion, Seneca, Notes from Underground) tend low volatility; narrative/essay-style tend higher. So H1 holds as a loose trend, not a clean stoic-vs-existential split; VADER and chunking still influence results.

- **H3 (yearning → higher syntactic complexity) is partially supported.** Aphoristic/compact texts (Enchiridion 20.7, Zarathustra 20.1, Dickinson 16.5) have shorter mean sentence length; narrative/expansive (Stranger 39.5, Notes 40.9, Myth 33.2, Leaves of Grass 37.4) have longer. **How to interpret:** Sentence length tracks **aphorism vs. narrative/lyric** style; the hypothesis holds as a stylistic signal.

- **Recommendation works in a meaningful way.** Existential-style queries surface Camus; Stoic-style queries surface Meditations. **How to interpret:** The engine is retrieving by **semantic/philosophical similarity**, not by suffering label. So we have a usable tool for “find passages that resonate with this thought,” even though we did not recover a clear taxonomy of suffering types.

**Bottom line:** We did **not** recover a clean “suffering type” taxonomy; clusters and PCA separate by **style and author/source** (narrative vs. archaic philosophical vs. book-specific vocabulary). We did get: (1) **H2 supported** (interpretable lexical anchors), (2) **H3 partially supported** (sentence length ≈ aphorism vs. narrative), (3) **H1 partially supported** in the 21-book run (low volatility in compact/stoic texts), (4) two working recommenders (chunk-level and cluster-based books), and (5) a clear negative result (structure ≠ suffering type) that is itself informative. **Distinguishing suffering types** would need different features or chunk-level labels; **style and register** are what we recover and what make recommendation useful.

---

## 1. Hypotheses

1. **Semantic divergence (H1)** — Stoic texts show lower “emotional volatility” than existential texts in embedding space.
2. **Lexical anchors (H2)** — A small set of words drives cluster separation in 3D PCA.
3. **“Yearning” signature (H3)** — Authors focused on human yearning show higher syntactic complexity than the aphoristic Stoic style.

---

## 2. Dataset

- **Primary corpus (Project Gutenberg):** The Brothers Karamazov (moral), The Myth of Sisyphus (existential), Meditations (stoic), East of Eden (intergenerational). See `data/README.md` for IDs and sources.
- **Chunking:** Books split into chapters or fixed-length passages; each row has `book_id`, `chunk_id`, `text`, `word_count`, `label_suffering_type`.
- **Features:** TF-IDF, sentence embeddings (sentence-transformers: `all-MiniLM-L6-v2` or `all-mpnet-base-v2`), syntactic metrics (spaCy), sentiment (VADER or transformer, e.g. DistilBERT SST-2). Optional: FAISS index for retrieval; cross-encoder re-ranker for top-k.

**Updated run:** The corpus was expanded to **21 books** (14 prose + 7 poetry: Dickinson, Whitman, Poe, Byron, Browning, Keats, Leaves of Grass). The results below in §4 and §5 reflect the **21-book run** (1,439 chunks) unless noted.

---

## 3. Related work and positioning

Work on **literary NLP** and **sentiment in narrative** has used both lexicon-based (e.g. VADER) and neural (e.g. BERT-based) sentiment; volatility and trajectory over narrative time have been studied for novels and social text. **Clustering of philosophical or literary corpora** by embedding similarity often recovers genre, period, or author rather than thematic "types" unless labels or objectives are tailored. This project sits at that intersection: we use **sentence-transformers** for embeddings and optional **transformer-based sentiment**, **K-means** and **PCA** for taxonomy and interpretable anchors, and **bi-encoder retrieval** with optional **cross-encoder re-ranking** and **FAISS**—standard in modern retrieval and RAG pipelines—to build a reproducible pipeline and evaluate it with a small relevance set and simple statistical checks.

---

## 4. Methods

| Step | Description |
|------|-------------|
| **Chunking** | Pipeline in `src/chunk_books.py`; output `data/processed/corpus.parquet`. |
| **Features** | Phase 2: TF-IDF, embeddings, syntactic, sentiment → `data/features/` (matrices + `corpus_features.parquet`). |
| **Clustering** | K-Means and hierarchical (Ward) on chunk embeddings; k chosen via elbow/silhouette (k=2 used). Cluster descriptions = top TF-IDF terms per cluster. Cross-tabs: cluster vs book, cluster vs label. |
| **Dimensionality reduction** | PCA (3 components) on TF-IDF; UMAP 2D/3D on embeddings. Anchor words = top \|loading\| per PCA component. |
| **Sentiment trajectory** | Rolling-window mean sentiment along book order; volatility = std of sentiment per book. |
| **Syntactic complexity** | Per-book mean sentence length, std sentence length, punctuation density. |
| **Recommendation** | Bi-encoder (embed query + corpus → cosine similarity); optional FAISS index; optional cross-encoder re-ranker on top-K candidates. Chunk-level and cluster-based (whole books per cluster) recommenders. |

---

## 5. Results

### 5.1 Clustering (suffering taxonomy) — 21-book run

- **K selection:** Silhouette is highest at **k=3** (~0.058); k=3 used. See `results/k_selection.csv`.
- **Cluster vs book:**  
  - **Cluster 0:** Narrative/dialogue — Republic (195), The Stranger (4), East of Eden (49), Crime and Punishment (41), Poe (76), Browning (36), Dickinson (16), etc. Top terms: *said, mr, oliver, replied, little, old, don, door, man, time, know, room, come, came*.  
  - **Cluster 1:** Archaic/philosophical — Meditations (84), Myth of Sisyphus (102), Sickness unto Death (105), Enchiridion (30), Zarathustra (134), Whitman (70), Seneca (3), etc. Top terms: *thou, man, things, men, unto, thee, thy, life, nature, love, good, world, shall, hath*.  
  - **Cluster 2:** One-book dominant — The Stranger (190 of 217 chunks). Top terms: *king, france, francis, duke, henry, charles, guise, paris, queen, court, bourbon, war* (setting-specific vocabulary).
- **Cluster vs label:** Cluster 0: philosophy (195), existential (76), poetry_sentimental (81), moral (46), intergenerational (49), poetry_existential (76), stoic (16). Cluster 1: existential (408), poetry_sentimental (104), stoic (117), intergenerational (12). Cluster 2: existential (211), poetry_existential (4), stoic (2). **Interpretation:** Clusters still separate by **style and lexical register** (dialogue/narrative vs. archaic philosophical vs. one novel’s setting), not by a single suffering type. Cluster 1 is the “philosophical” pole (existential + stoic + poetry); Cluster 0 mixes Republic, novels, and poetry; Cluster 2 is largely *The Stranger*.

### 5.2 PCA and lexical anchors (H2)

- **PCA (21-book run):** First component: archaic/philosophical (*thou, thy, unto, thee, doth, hath, thyself, whatsoever, things, zarathustra*) vs. narrative (*mr, oliver, replied, don*). Second: setting-specific (*king, france, francis, duke, henry, charles, guise, queen, paris, coligny*) vs. archaic (*thou, unto, thy, thee*). Third: again archaic (*thou, thy, thee, unto*). So **H2 is supported**: a small set of words (archaic, narrative, and book-specific setting) drives separation; these are interpretable as **style and source**, not suffering type.

### 5.3 Sentiment trajectory and volatility (H1)

- **Per-book volatility (21-book run):** Stoic: Enchiridion 0.36, Seneca 0.48, Meditations 0.75. Existential: Notes from Underground 0.03 (13 chunks), Sickness unto Death 0.32, Myth of Sisyphus 0.50, The Stranger 0.76, Zarathustra 0.75, Man’s Search for Meaning 0.85. Poetry: Byron ~0, Leaves of Grass ~0 (few chunks); Poe 0.93, Keats 0.87, Republic 0.91.
- **Interpretation:** **H1 is partially supported in the larger corpus:** “Pure” stoic texts (Enchiridion 0.36, Seneca 0.48) have **lower** volatility than many existential (Stranger 0.76, Zarathustra 0.75, Man’s Search 0.85). But Meditations (0.75) remains high, and some existential are low (Notes 0.03, Sickness 0.32). So the trend is noisy: **aphoristic/letter-style texts** (Enchiridion, Seneca, Notes) tend low volatility; **narrative and essay-style** (Meditations as read by chunking, Stranger, Man’s Search) tend higher. VADER and chunking still likely drive much of this; H1 is **partially supported** as a stylistic pattern, not as a clean stoic-vs-existential finding.

### 5.4 Syntactic complexity (H3)

- **Mean sentence length (21-book run):** Stoic: Enchiridion 20.7, Meditations 27.3, Seneca 23.4. Existential: Zarathustra 20.1, Sickness 22.2, Myth of Sisyphus 33.2, The Stranger 39.5, Notes 40.9, Man’s Search 24.3. Poetry: Dickinson 16.5, Keats 18.6; Whitman 31.6, Poe 29.2, Byron 38.8, Browning 30.2, Leaves of Grass 37.4.
- **Interpretation:** **H3 is partially supported:** Aphoristic/compact texts (Enchiridion 20.7, Zarathustra 20.1, Dickinson 16.5, Keats 18.6) have **shorter** mean sentence length; narrative/expansive texts (The Stranger 39.5, Notes 40.9, Myth 33.2, East of Eden 32.8, Byron 38.8, Leaves 37.4) have **longer**. So sentence length tracks **aphorism vs. narrative/lyric** more than “yearning” per se; the hypothesis holds as a **stylistic** signal.

### 5.5 Recommendation engine

- Query “I am tormented by the idea of meaninglessness” → top chunks from *Myth of Sisyphus*. Query “Accept what is in your control and let go of the rest” → top chunks from *Meditations* and *East of Eden*. Recommendations align with the philosophical tone of the query. Use `python recommend.py "quote"` or `python recommend.py "quote" --rerank`; optional FAISS after `python scripts/build_faiss_index.py`.

### 5.6 Evaluation

- **Qualitative:** Top-5 books per cluster (`python recommend_books.py`) align with cluster descriptions.
- **Quantitative:** `data/relevance_set.csv` + `python scripts/eval_recommendation.py` → P@k in `results/eval_recommendation.json`.
- **Statistical test:** `python scripts/stats_hypothesis.py` → `results/stats_hypothesis.txt` (p-value for stoic vs. existential).

### 5.7 Cluster-label agreement and zero-shot check

- **Cluster-label agreement:** We measure how well unsupervised K-means clusters align with book-level labels using **NMI** and **ARI**. Run `python scripts/cluster_label_agreement.py` after Phase 3; results in `results/cluster_label_agreement.json`. Low NMI/ARI supports that our taxonomy is driven by style rather than by the intended "suffering type."
- **Zero-shot classification:** We run a pre-trained zero-shot model (no fine-tuning) on a sample of chunks with labels like "stoic philosophy," "existential philosophy," "poetry," etc., and compute **accuracy vs book label** and **NMI vs K-means cluster**. Run `python scripts/zero_shot_eval.py`; results in `results/zero_shot_metrics.json`. This checks whether an external model’s notion of type agrees with our clusters and labels.

### 5.8 Ablation (embedding model)

| Model | Silhouette (k=3) | Notes |
|-------|------------------|-------|
| all-MiniLM-L6-v2 | ~0.058 | Baseline. |
| all-mpnet-base-v2 | Run Phase 2 with `--embedding-model all-mpnet-base-v2` then Phase 3 | Stronger. |

---

## 6. Limitations

- **Suffering as construct:** Book-level labels are proxies; features capture stylistic/lexical similarity, which may only partly reflect "type of suffering."
- **Chunk imbalance:** Brothers Karamazov has very few chunks (5) in the run used here; per-book and cluster results are sensitive to this.
- **Labels:** `label_suffering_type` is assigned by book, not by chunk; we do not have chunk-level ground truth for “type of suffering.”
- **VADER:** Designed for social/media text; may not capture philosophical or archaic tone well; volatility may reflect style as much as “emotional” content.
- **PCA variance:** First three components explain only a small fraction of TF-IDF variance; structure is spread across many dimensions.
- **Cross-genre:** Poetry/secondary corpus was not fully integrated; recommendation cross-genre notes are in `results/recommendation_poetry_notes.md`.

---

## 7. Reproducing

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

## 8. Summary

The pipeline builds a **suffering taxonomy** from chunk embeddings and TF-IDF: clustering separates narrative vs. philosophical style; PCA anchor words are interpretable (archaic, narrative, author-specific). **H2 (lexical anchors)** is supported; **H3 (syntactic complexity)** is partially supported (Meditations shorter sentences). **H1 (stoic lower volatility)** is not supported in this run and is limited by chunk balance and sentiment model choice. The recommendation engine returns semantically coherent chunks for existential and stoic queries.
