# Text Mining — Course Project Portfolio

A collection of four applied NLP projects built for the Text Mining course. Each project explores a distinct problem domain while sharing a common foundation in text preprocessing, word embeddings, and transformer-based models.

---

## Projects

### 1. HierABSA — Hierarchical Aspect-Based Sentiment Analysis
**Directory:** [sentiment_analysis/](sentiment_analysis/)  
**Contributor:** Raunak Pal

A multi-paradigm ABSA system that discovers product aspects automatically using BERTopic + HDBSCAN clustering, then scores sentiment through three complementary engines — a fine-tuned DeBERTa-v3 transformer, a Gemini Flash zero-shot LLM, and a VADER+spaCy lexicon pipeline. Opinions are aggregated across Reddit posts using upvote-weighted scoring and organized into a four-level Product → Category → Aspect → Topic hierarchy via a NetworkX directed graph.

**Corpus:** Samsung Galaxy S25 — 2,858 sentences from 1,247 Reddit posts  
**Embeddings:** all-MiniLM-L6-v2 (sentence-transformers)

---

### 2. GDELT Real-Time Narrative Clustering
**Directory:** [clustering/](clustering/)  
**Contributor:** Sanjay M

An online clustering system that ingests live news data from the GDELT Global Knowledge Graph on a 15-minute heartbeat, encodes articles with all-MiniLM-L6-v2, and groups emerging narratives using an incremental centroid-based Leader Algorithm with EWMA centroid updates (α = 0.95, τ = 0.65–0.70). A real-time Streamlit dashboard visualizes cluster evolution using file-based IPC via pickle.

**Evaluation:** 24-hour monitoring — mean intra-cluster cosine similarity 0.74

---

### 3. Multimodal Meme Sarcasm Detection
**Directory:** [computer_vision/](computer_vision/)  
**Contributor:** Aryan Gupta

Compares three multimodal architectures for detecting sarcasm in memes by fusing image and text signals: a CNN+BiLSTM late-fusion model (~12.1M params), a CLIP-based three-way fusion model using concatenation, Hadamard product, and absolute difference (~87M params), and a VisualBERT Lite model with early cross-modal attention (~22.4M params). Evaluated on the MMSD 2.0 benchmark and a 200-sample CPU-compatible synthetic dataset.

**Best result:** VisualBERT Lite — F1: 77.3%

---

### 4. Emotion Classification with Crisis Detection
**Directory:** [classification/](classification/)  
**Contributor:** Jaisanjay

A fine-grained emotion classifier trained on the GoEmotions benchmark (58K Reddit comments, 28 labels) that maps predictions through a three-tier label hierarchy (fine-grained → coarse → valence). Five model families are benchmarked — Logistic Regression, SVM, BiLSTM (GloVe 300-d), BERT (bert-base-uncased), and GPT-3.5 — with a joint auxiliary head for crisis signal detection.

**Best result:** BERT — macro F1: 72.6%, crisis detection F1: 70.4%

---

## Topics Covered Across All Projects

| Topic | HierABSA | Clustering | Meme Sarcasm | Classification |
|---|:---:|:---:|:---:|:---:|
| Text Preprocessing | ✓ | ✓ | ✓ | ✓ |
| Word Embeddings | ✓ | ✓ | ✓ | ✓ |
| Transformer-based Models | ✓ | ✓ | ✓ | ✓ |
| Sentiment Analysis | ✓ | | | |
| Topic Modeling | ✓ | | | |
| Text Clustering | ✓ | ✓ | | |
| Bag of Words / TF-IDF | | | | ✓ |
| Text Classification | | | ✓ | ✓ |
| Information Retrieval | | ✓ | | |

---

## Repository Structure

```
text_mining/
├── sentiment_analysis/   # HierABSA (Raunak)
├── clustering/           # GDELT Narrative Clustering (Sanjay)
├── computer_vision/      # Meme Sarcasm Detection (Aryan)
├── classification/       # Emotion Classification (Jaisanjay)
└── papers/               # Research papers for each project
```

Each project directory contains its own `README.md` with setup instructions, dependencies, and how to run.
