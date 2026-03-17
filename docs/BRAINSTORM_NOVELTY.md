# Brainstorm: Novelty for LinkedIn & Final Run

## Goal

- Gear the project toward **one clear novelty** we can explain and show.
- Keep implementation **easy** so you can do a final run and then share on LinkedIn (bridging your passions: philosophy/literature + NLP).

---

## Zero-shot: no fine-tuning, no LoRA

**Important:** Zero-shot classification does **not** mean “we fine-tune a model.” It means we use an **already trained** model that can assign labels from a list **without any training on our data**.

- We pick a **pre-trained zero-shot model** (e.g. Hugging Face `zero-shot-classification` with `MoritzLaurer/deberta-v3-base-zeroshot-v2` or `facebook/bart-large-mnli`).
- We pass each chunk (or a sample) and candidate labels: e.g. `["stoic philosophy", "existential philosophy", "moral fiction", "poetry"]`.
- The model returns scores for each label; we take the argmax as “predicted” label.
- We then compare: (1) **agreement with book-level label** (accuracy), (2) **agreement with K-means cluster** (NMI or simple table). No training loop, no LoRA, no new training data — just one script and a pre-trained model.

So zero-shot is actually **easier** than fine-tuning: one script, one pipeline call, then compute metrics. The “novelty” is: we ask an external model “what type is this passage?” and see if it aligns with our taxonomy and clusters.

---

## Contrastive / constrained clustering

- **Full contrastive** (learn embeddings so same-label chunks are closer) = fine-tuning, more work.
- **Constrained clustering** (e.g. “must-link” same book) = needs a solver; doable but more code.
- **Easiest “contrastive-like” move:** **Measure how well our current clusters align with book labels.** We already have cluster assignments and book-level labels. We compute **NMI** (normalized mutual information) and **ARI** (adjusted Rand index) between cluster_id and book label. One small script, no new model. Story: “Our unsupervised clusters have NMI X with the intended suffering-type labels” — shows we evaluated the taxonomy.

---

## Recommendation: two small additions (both easy)

| Addition | What | Effort | LinkedIn / novelty angle |
|----------|------|--------|---------------------------|
| **1. Cluster–label agreement (NMI/ARI)** | One script: load clusters + book labels, compute NMI and ARI, write to `results/`. | ~15 min | “We measured how well our unsupervised taxonomy aligns with the intended labels.” |
| **2. Zero-shot “suffering type”** | One script: run HF zero-shot on a sample of chunks with labels like stoic/existential/moral/poetry; compute accuracy vs book label and NMI vs cluster; write metrics. | ~30 min | “We asked a general-purpose NLP model to classify passages and compared it to our clusters — no fine-tuning.” |

Both are **implement-and-run**; no training, no LoRA. They give you:

- A clear **number** (NMI, accuracy) to put in the report and in a LinkedIn post.
- A story: “I built a taxonomy of philosophical literature, then checked it with zero-shot classification and cluster–label agreement.”

---

## LinkedIn angle (draft)

You could say something like:

- “I combined my love of philosophy and literature with NLP: built a taxonomy of 21 works (from Meditations to East of Eden to Dickinson) using embeddings and clustering. Does ‘type of suffering’ (stoic, existential, moral) show up? Turns out **style** dominates — but we can still recommend passages and books by philosophical texture. We then ran a zero-shot model (no fine-tuning) to see if AI agrees with our clusters, and reported how well clusters align with labels (NMI). Full pipeline: retrieval with re-ranking, FAISS, evaluation.”

That highlights: passion, taxonomy, negative result, zero-shot check, and pipeline quality — all without claiming heavy ML novelty.

---

## What we’ll implement

1. **`scripts/cluster_label_agreement.py`**  
   - Inputs: `results/labels_kmeans.csv`, `data/features/corpus_features.parquet` (or corpus) for book_id → label.  
   - Compute NMI and ARI (cluster_id vs book label).  
   - Write `results/cluster_label_agreement.json` (and maybe .txt).  
   - Optional: add to `run_all.py`.

2. **`scripts/zero_shot_eval.py`**  
   - Inputs: corpus (text + book label), optional `labels_kmeans.csv`.  
   - Run Hugging Face zero-shot classification on a sample of chunks (e.g. 300) with candidate labels aligned to our schema (e.g. stoic, existential, moral, intergenerational, philosophy, poetry).  
   - Compute: accuracy vs book label; NMI( zero_shot_pred, cluster_id ) if clusters exist.  
   - Write `results/zero_shot_predictions.csv` (optional) and `results/zero_shot_metrics.json`.  
   - Optional: add to `run_all.py`.

3. **Report + TECHNICAL_BREAKDOWN**  
   - One sentence or short subsection: “Cluster–label agreement (NMI=X, ARI=Y). Zero-shot classification accuracy vs book label = Z; NMI with K-means = W.”  
   - In TECHNICAL_BREAKDOWN: “We added zero-shot evaluation (no fine-tuning) and cluster–label agreement as a simple novelty.”

4. **`run_all.py`**  
   - After stats, run cluster_label_agreement (if labels_kmeans and corpus_features exist), then zero_shot_eval (if corpus and optionally clusters exist).

No LoRA, no fine-tuning — just two scripts and a clear story for your final run and LinkedIn post.

---

## Implemented

- **`scripts/cluster_label_agreement.py`** — NMI and ARI between K-means cluster and book label. Run after Phase 3.
- **`scripts/zero_shot_eval.py`** — Zero-shot classification on a sample of chunks; accuracy vs book label, NMI vs cluster. Run after Phase 2 (optionally Phase 3 for NMI). Uses `MoritzLaurer/deberta-v3-base-zeroshot-v2` by default.
- **`run_all.py`** — Now runs cluster_label_agreement and zero_shot_eval after stats (steps 11 and 12).
- **Report** — §5.7 documents both; TECHNICAL_BREAKDOWN lists them under “What is novel.”

---

## LinkedIn post (draft)

Short version:

"I combined my love of philosophy and literature with NLP: built a taxonomy of 21 works (from Meditations to East of Eden to Dickinson) using embeddings and clustering. Does 'type of suffering' (stoic, existential, moral) show up in the structure? Turns out **style** dominates—but we can still recommend passages and books by philosophical texture. I then ran a zero-shot model (no fine-tuning) to see if AI agrees with our clusters, and measured how well clusters align with labels (NMI/ARI). Full pipeline: retrieval with re-ranking, FAISS, and evaluation. [Link to repo or write-up]"

Longer version (if you want to add one line on tech):

"Bridged two passions—philosophy/literature and NLP—in one project: a taxonomy of 21 philosophical and literary works (Gutenberg) with embeddings, clustering, and hypothesis tests. We ask: can we recover a 'taxonomy of suffering' (stoic, existential, moral) from text? Answer: style and register dominate; the intended 'type' doesn’t show up as a clean taxonomy—a negative result that’s still informative. We added zero-shot classification (no fine-tuning) to check if an external model agrees with our clusters, and we report cluster–label agreement (NMI/ARI). Pipeline: sentence-transformers, K-means, PCA/UMAP, bi-encoder retrieval with cross-encoder re-ranking and FAISS, plus evaluation and stats. [Link]"
