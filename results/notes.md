
## Phase 3: Clustering

K-Means and hierarchical (Ward) with k=2 on chunk embeddings. Cluster descriptions use top TF-IDF terms per cluster. Check cluster_vs_book.csv and cluster_vs_label.csv to see whether clusters align more with suffering type (label) or with book/author.


## Phase 4: Dimensionality reduction & anchors

PCA on TF-IDF (3 components) and UMAP 2D/3D on embeddings. Anchor words = top |loading| per PCA component (anchor_words.csv). Hypothesis 2: if anchors align with intuitive suffering types (e.g. moral, existential), lexical separation supports the taxonomy; compare with cluster_descriptions.csv and cross-tabs.


## Phase 5: Sentiment trajectory & Hypothesis 1

Rolling-window mean sentiment and per-book volatility (std of sentiment). Hypothesis 1: Stoic texts (e.g. Meditations) show lower emotional volatility than existential texts (e.g. Myth of Sisyphus). Compare volatility_by_book.csv across label_suffering_type (stoic vs existential).


## Phase 6: Syntactic complexity & Hypothesis 3

Per-book mean sentence length and punctuation density. Hypothesis 3: yearning-focused texts (e.g. Dostoevsky, Steinbeck) show higher syntactic complexity than aphoristic Stoic style (e.g. Meditations). Compare syntactic_by_book.csv and syntactic_by_book.png across books.

