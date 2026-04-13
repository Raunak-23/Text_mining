# ABSA + Topic Modeling System: Detailed Project Plan

## Project Title
**Hierarchical Aspect-Based Sentiment Analysis with Neural Topic Modeling for Product Reviews from Reddit**

---

## 1. Problem Statement

Given a product name (e.g., "iPhone 15 Pro"), the system:
1. Fetches relevant discussions from Reddit via PRAW
2. Discovers aspect hierarchy through neural topic modeling (e.g., Smartphone → {Camera, Battery, Display, Performance, Thermals, ...})
3. Performs Aspect-Based Sentiment Analysis (ABSA) at the sentence level
4. Aggregates and returns both overall product sentiment and per-aspect sentiment scores
5. Compares multiple ABSA approaches: pretrained transformer models vs. LLM-based extraction

The system runs entirely from the terminal via a CLI, requires no GPU, and produces publication-quality outputs.

---

## 2. System Architecture

```
                          ┌─────────────────────────┐
                          │     CLI Interface        │
                          │  (Rich + Typer)          │
                          └────────┬────────────────┘
                                   │
                          ┌────────▼────────────────┐
                          │   Pipeline Orchestrator  │
                          └────────┬────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
┌───────▼───────┐   ┌─────────────▼──────────┐   ┌──────────▼──────────┐
│  Data Layer   │   │  Analysis Layer         │   │  Output Layer       │
│               │   │                         │   │                     │
│ - Collection  │──▶│ - Preprocessing         │──▶│ - Aggregation       │
│ - Caching     │   │ - Topic Modeling        │   │ - Visualization     │
│ - Validation  │   │ - Aspect Hierarchy      │   │ - Report Generation │
│               │   │ - ABSA (multi-approach) │   │ - Export            │
└───────────────┘   │ - Evaluation            │   └─────────────────────┘
                    └─────────────────────────┘
```

---

## 3. Module Breakdown

### Module 1: Data Collection (`src/data/collector.py`)

**Purpose:** Fetch product-related Reddit data via PRAW with smart query construction.

**Technical Details:**
- **Reddit Search Strategy:**
  - Search across product-relevant subreddits: auto-detect category or accept user override
  - Default subreddit lists per category:
    - Smartphones: `r/Android`, `r/iphone`, `r/smartphones`, `r/gadgets`
    - Laptops: `r/laptops`, `r/SuggestALaptop`, `r/hardware`
    - Audio: `r/headphones`, `r/audiophile`
    - General: `r/gadgets`, `r/technology`, `r/BuyItForLife`
  - Query construction: product name + common review-related terms
  - Search via `subreddit.search(query, sort="relevance", time_filter="year", limit=N)`

- **PRAW Fields to Collect:**
  - **Submissions:** `id`, `title`, `selftext`, `score`, `upvote_ratio`, `num_comments`, `created_utc`, `subreddit`, `url`, `link_flair_text`, `author`
  - **Comments:** `id`, `body`, `score`, `created_utc`, `parent_id`, `depth`, `author`, `is_submitter`
  - **Comment Forest:** Use `submission.comments.replace_more(limit=0)` then `.list()` for flat traversal, preserving `parent_id` for thread reconstruction
  - Handle `MoreComments` objects properly with configurable depth

- **Data Filtering & Quality:**
  - Skip `[deleted]` and `[removed]` comments
  - Minimum comment length: 10 characters (filter bot/spam)
  - Minimum submission score: 2 (filter low-quality)
  - Deduplication by comment ID
  - Rate limiting: respect PRAW's built-in rate limiter + exponential backoff

- **Caching:**
  - Cache fetched data as JSON in `data/raw/{product_slug}/`
  - Invalidation: re-fetch if cache older than configurable TTL (default 7 days)
  - CLI flag `--no-cache` to force refresh

- **Output Schema:**
  ```python
  @dataclass
  class RedditComment:
      id: str
      body: str
      score: int
      created_utc: datetime
      parent_id: str
      depth: int
      is_submitter: bool
      submission_id: str

  @dataclass
  class RedditSubmission:
      id: str
      title: str
      selftext: str
      score: int
      upvote_ratio: float
      num_comments: int
      created_utc: datetime
      subreddit: str
      comments: list[RedditComment]
  ```

---

### Module 2: Text Preprocessing (`src/data/preprocessor.py`)

**Purpose:** Clean and segment Reddit text into analysis-ready sentences.

**Technical Details:**
- **Cleaning Pipeline:**
  1. URL removal (regex: `https?://\S+`)
  2. Reddit-specific cleanup: remove `/u/username`, `/r/subreddit` mentions, quote blocks (`>`)
  3. Markdown stripping: bold, italic, links, code blocks
  4. Emoji/emoticon handling: convert to text descriptions via `demoji` or strip
  5. Contraction expansion ("don't" → "do not") for better tokenization
  6. Unicode normalization (NFKD)
  7. Whitespace normalization

- **Sentence Segmentation:**
  - Use `spaCy` sentencizer (`en_core_web_sm`) for sentence boundary detection
  - Minimum sentence length: 5 tokens (filter fragments)
  - Maximum sentence length: 128 tokens (split run-on sentences)
  - Preserve sentence-to-comment mapping for traceability

- **Output:**
  ```python
  @dataclass
  class ProcessedSentence:
      sentence_id: str
      text: str
      comment_id: str
      submission_id: str
      comment_score: int  # inherited for weighted aggregation
  ```

---

### Module 3: Neural Topic Modeling (`src/models/topic_model.py`)

**Purpose:** Discover product aspects via BERTopic with hierarchical topic reduction — this is the "hierarchical graph structure" for aspects.

**Technical Details:**

- **Why BERTopic over LDA:**
  - Neural embeddings capture semantic similarity (LDA is bag-of-words)
  - Built-in hierarchical topic reduction (produces the tree structure)
  - Produces coherent, interpretable topics out-of-the-box
  - State-of-the-art for short-text topic modeling (Reddit comments are short)

- **BERTopic Configuration:**
  ```python
  from bertopic import BERTopic
  from sentence_transformers import SentenceTransformer
  from sklearn.cluster import HDBSCAN  # replaced with hdbscan
  from sklearn.feature_extraction.text import CountVectorizer
  from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

  # Embedding model (CPU-friendly, ~80MB)
  embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

  # UMAP for dimensionality reduction (CPU-friendly)
  umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

  # HDBSCAN for clustering
  hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples=5, prediction_data=True)

  # Vectorizer with domain stopwords
  vectorizer = CountVectorizer(
      stop_words="english",
      ngram_range=(1, 3),
      min_df=3
  )

  # Representation model for better topic labels
  representation_model = [
      KeyBERTInspired(),
      MaximalMarginalRelevance(diversity=0.3)
  ]

  topic_model = BERTopic(
      embedding_model=embedding_model,
      umap_model=umap_model,
      hdbscan_model=hdbscan_model,
      vectorizer_model=vectorizer,
      representation_model=representation_model,
      nr_topics="auto",  # auto-reduce based on coherence
      top_n_words=10,
      verbose=True
  )
  ```

- **Hierarchical Topic Reduction (The Graph):**
  - After initial topic discovery, use `topic_model.hierarchical_topics(docs)` to build a **hierarchical tree**
  - This produces a `pandas.DataFrame` with columns: `Parent_ID`, `Parent_Name`, `Topics`, `Child_Left_ID`, `Child_Right_ID`, `Distance`
  - The hierarchy naturally creates: **Product → High-level Category → Specific Aspect**
    - E.g., `Root → Hardware → Camera → {Low-light, Zoom, Video}`
    - E.g., `Root → Software → UI → {Animations, Customization}`
  - Visualize with `topic_model.visualize_hierarchy()` (saves as HTML)
  - Convert to NetworkX graph for programmatic traversal

- **Topic-to-Aspect Mapping:**
  ```python
  # Build hierarchical aspect graph
  import networkx as nx

  def build_aspect_hierarchy(topic_model, docs) -> nx.DiGraph:
      """Convert BERTopic hierarchy into a directed aspect graph."""
      hierarchy = topic_model.hierarchical_topics(docs)
      G = nx.DiGraph()

      # Add product as root
      G.add_node("product", level=0, label=product_name)

      # Add topics as leaves with their keywords
      for topic_id in topic_model.get_topic_info()["Topic"]:
          if topic_id == -1:  # skip outlier topic
              continue
          keywords = [w for w, _ in topic_model.get_topic(topic_id)]
          G.add_node(
              f"topic_{topic_id}",
              level=2,
              label=topic_model.get_topic_info().loc[topic_id, "Name"],
              keywords=keywords,
              count=topic_model.get_topic_info().loc[topic_id, "Count"]
          )

      # Add intermediate hierarchy from dendrogram
      for _, row in hierarchy.iterrows():
          parent = f"cluster_{row['Parent_ID']}"
          G.add_node(parent, level=1, label=row["Parent_Name"])
          # Connect children
          for child_topic in row["Topics"]:
              G.add_edge(parent, f"topic_{child_topic}")
          G.add_edge("product", parent)

      return G
  ```

- **Topic Coherence Evaluation:**
  - Compute `C_v` coherence score using `gensim.models.coherencemodel`
  - Report coherence per topic and mean coherence
  - Compare against LDA baseline (same data) for the evaluation section

---

### Module 4: Aspect-Based Sentiment Analysis (`src/models/absa.py`)

**Purpose:** Determine sentiment polarity for each aspect mentioned in a sentence. Three approaches implemented for comparison.

#### Approach 1: Transformer-based ABSA (Primary) — HuggingFace

- **Model:** `yangheng/deberta-v3-base-absa-v1.1` via `transformers.AutoModelForSequenceClassification`
- **Implementation:** Direct HuggingFace (no PyABSA wrapper — simpler, more stable)
- **Input format:** `"{sentence} [SEP] {aspect}"` — one call per (sentence, aspect) pair
- **Output:** logits for 3 classes → softmax → {0: negative, 1: neutral, 2: positive}
- **Why DeBERTa:** SemEval-2014 fine-tuned for ABSC; provides RQ4 domain-transfer benchmark

  ```python
  from transformers import AutoTokenizer, AutoModelForSequenceClassification
  tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
  model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
  # Input: "{sentence} [SEP] {aspect}"
  # Label map: {0: "negative", 1: "neutral", 2: "positive"}
  ```

#### Approach 2: LLM-based ABSA — Gemini Flash

- **Model:** `gemini-3.1-flash-preview` via `google-genai` SDK (new SDK, replaces deprecated `google-generativeai`)
- **Prompt Design (zero-shot structured extraction, batched 10 sentences per call):**
  ```
  Return a JSON array of arrays — one inner array per sentence.
  Each inner array: [{"aspect": str, "sentiment": "positive|neutral|negative",
                      "confidence": float, "opinion_words": [str]}]
  ```
- **Batching:** 10 sentences per API call
- **Cost control:** `llm_sample=300` cap — only 300 sentences sent to LLM by default

#### Approach 3: Lexicon + Dependency Parsing Baseline

- **Purpose:** Simple baseline for comparison (no ML, fully interpretable)
- **Method:**
  1. Parse sentence with spaCy dependency parser
  2. Extract noun phrases as candidate aspects
  3. Find sentiment-bearing words (adjectives, adverbs) linked via dependency arcs
  4. Score using VADER or SentiWordNet
- **Implementation:**
  ```python
  import spacy
  from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

  nlp = spacy.load("en_core_web_sm")
  vader = SentimentIntensityAnalyzer()

  def lexicon_absa(sentence: str) -> list[dict]:
      doc = nlp(sentence)
      results = []
      for chunk in doc.noun_chunks:
          # Find opinion words linked to this noun chunk
          opinion_words = [
              token for token in chunk.root.children
              if token.dep_ in ("amod", "advmod") or token.pos_ == "ADJ"
          ]
          if opinion_words:
              opinion_text = " ".join([t.text for t in opinion_words])
              score = vader.polarity_scores(opinion_text)["compound"]
              sentiment = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
              results.append({"aspect": chunk.text, "sentiment": sentiment, "confidence": abs(score)})
      return results
  ```

---

### Module 5: Aspect Mapping & Normalization (`src/models/aspect_mapper.py`)

**Purpose:** Map raw extracted aspects to canonical aspect nodes in the hierarchy graph.

**Technical Details:**
- Raw ABSA extracts noisy aspect terms: "cam", "cameras", "camera quality", "photo", "pics" → all map to **Camera**
- **Approach:** Semantic similarity matching using the same `all-MiniLM-L6-v2` embeddings

  ```python
  from sentence_transformers import SentenceTransformer, util

  def map_aspect_to_hierarchy(
      raw_aspect: str,
      aspect_graph: nx.DiGraph,
      embedding_model: SentenceTransformer,
      threshold: float = 0.5
  ) -> str | None:
      """Map a raw aspect term to the nearest node in the aspect hierarchy."""
      # Get canonical aspect labels from graph leaf nodes
      canonical_aspects = {
          node: data["keywords"]
          for node, data in aspect_graph.nodes(data=True)
          if data.get("level") == 2  # leaf topics
      }

      # Compute similarity
      raw_emb = embedding_model.encode(raw_aspect, convert_to_tensor=True)
      best_match, best_score = None, 0.0
      for node, keywords in canonical_aspects.items():
          kw_emb = embedding_model.encode(keywords, convert_to_tensor=True)
          score = util.cos_sim(raw_emb, kw_emb).max().item()
          if score > best_score:
              best_score = score
              best_match = node

      return best_match if best_score >= threshold else None
  ```

- **Manual Override:** Allow user-provided aspect taxonomy via YAML config for specific product categories:
  ```yaml
  # config/aspects/smartphone.yaml
  aspects:
    camera:
      aliases: [camera, cam, photos, photography, video, lens, zoom, low-light]
    battery:
      aliases: [battery, charging, battery life, endurance, power]
    display:
      aliases: [display, screen, oled, brightness, refresh rate, resolution]
    performance:
      aliases: [performance, speed, lag, ram, processor, chip, benchmark]
    thermals:
      aliases: [thermals, heating, temperature, hot, overheating, throttling]
    build:
      aliases: [build, design, weight, premium, material, glass, metal]
    software:
      aliases: [software, ui, os, updates, bloatware, features, android, ios]
    audio:
      aliases: [speaker, audio, sound, earpiece, microphone, call quality]
    value:
      aliases: [price, value, worth, expensive, cheap, cost, money]
  ```

---

### Module 6: Aggregation & Scoring (`src/analysis/aggregator.py`)

**Purpose:** Compute final sentiment scores at multiple granularities.

**Technical Details:**

- **Score-Weighted Aggregation:**
  - Reddit comment scores (upvotes) serve as implicit quality signals
  - Weight each sentence's ABSA result by `log1p(comment_score)` to amplify high-quality opinions
  - Formula:
    ```
    aspect_sentiment(a) = Σ(sentiment_i × weight_i) / Σ(weight_i)
    where weight_i = log1p(comment_score_i)
    and sentiment_i ∈ {-1, 0, +1} (negative, neutral, positive)
    ```

- **Confidence-Weighted:**
  - Also weight by model confidence: `final_weight = log1p(score) × confidence`

- **Aggregation Levels:**
  1. **Sentence-level:** Raw ABSA output (aspect, sentiment, confidence)
  2. **Comment-level:** Aggregate sentences within a comment
  3. **Aspect-level:** Aggregate across all comments for each canonical aspect
  4. **Category-level:** Aggregate aspect scores up the hierarchy (e.g., Hardware = mean(Camera, Battery, Display))
  5. **Product-level:** Overall sentiment = weighted mean of all aspect scores

- **Output Data Structure:**
  ```python
  @dataclass
  class AspectSentimentResult:
      aspect: str
      positive_pct: float  # 0-100
      negative_pct: float
      neutral_pct: float
      avg_score: float     # -1 to +1
      num_mentions: int
      sample_positive: list[str]  # top 3 representative positive sentences
      sample_negative: list[str]  # top 3 representative negative sentences

  @dataclass
  class ProductSentimentReport:
      product_name: str
      overall_score: float
      overall_label: str  # "Highly Positive", "Mixed", etc.
      total_comments_analyzed: int
      total_sentences_analyzed: int
      aspect_results: dict[str, AspectSentimentResult]
      hierarchy: nx.DiGraph
      topic_coherence: float
      data_sources: list[str]  # subreddits used
  ```

---

### Module 7: Evaluation & Model Comparison (`src/evaluation/comparator.py`)

**Purpose:** Quantitatively compare the three ABSA approaches and report metrics.

**Technical Details:**

- **Inter-Annotator Agreement (Model Comparison):**
  - Since we lack gold-standard labels, treat LLM output as silver-standard
  - Compute Cohen's Kappa between each pair of approaches
  - Report agreement matrix:
    ```
                     Transformer   LLM     Lexicon
    Transformer         —         κ=0.72   κ=0.45
    LLM               κ=0.72       —       κ=0.51
    Lexicon            κ=0.45     κ=0.51     —
    ```

- **Aspect Extraction Evaluation:**
  - Use LLM-extracted aspects as reference (most comprehensive)
  - Compute precision, recall, F1 for Transformer and Lexicon approaches
  - Report per-aspect and macro-averaged

- **Sentiment Classification Evaluation:**
  - On the intersection of aspects found by all three methods
  - Accuracy, macro-F1, per-class precision/recall

- **Topic Coherence Metrics:**
  - BERTopic `C_v` coherence vs. LDA `C_v` coherence (same data)
  - Topic diversity: proportion of unique words across topics
  - Silhouette score on the UMAP embeddings

- **Runtime & Cost Comparison:**
  - Wall-clock time per approach
  - API cost for LLM approach (token count × price)
  - Memory footprint

---

### Module 8: CLI Interface (`src/cli.py`)

**Purpose:** Rich terminal interface using Typer + Rich.

**Commands (implemented):**
```bash
# Full pipeline: fetch -> preprocess -> topics -> ABSA -> evaluate
uv run absa analyze "Samsung Galaxy S25"

# Step-by-step:
uv run absa fetch "Samsung Galaxy S25" --limit 20 --time-filter year
uv run absa preprocess "Samsung Galaxy S25"
uv run absa topics "Samsung Galaxy S25"
uv run absa absa "Samsung Galaxy S25" --paradigms transformer,llm,lexicon --llm-sample 300
uv run absa compare "Samsung Galaxy S25"   # show cached evaluation + tables

# Utilities:
uv run absa info    # show config + credential check
```

**Terminal Output (Rich):**
```
╭──────────────────────────────────────────────────────────────────╮
│              iPhone 15 Pro — Sentiment Report                    │
├──────────────────────────────────────────────────────────────────┤
│  Overall Sentiment:  ████████████████████░░░░  78% Positive     │
│  Based on: 1,247 comments from 89 threads (3 subreddits)       │
╰──────────────────────────────────────────────────────────────────╯

  Aspect Breakdown:
  ┌─────────────┬───────────┬───────────┬───────────┬──────────┐
  │ Aspect      │ Positive  │ Negative  │ Neutral   │ Mentions │
  ├─────────────┼───────────┼───────────┼───────────┼──────────┤
  │ Camera      │ ██████ 85%│ ██ 10%    │ █ 5%      │ 342      │
  │ Battery     │ ███ 35%   │ █████ 52% │ █ 13%     │ 289      │
  │ Display     │ ███████92%│ █ 4%      │ █ 4%      │ 201      │
  │ Performance │ ██████ 80%│ ██ 12%    │ █ 8%      │ 178      │
  │ Thermals    │ ██ 22%    │ ██████ 65%│ █ 13%     │ 134      │
  │ Build       │ ███████88%│ █ 7%      │ █ 5%      │ 156      │
  │ Software    │ █████ 60% │ ███ 28%   │ █ 12%     │ 198      │
  │ Price/Value │ ██ 25%    │ █████ 55% │ ██ 20%    │ 167      │
  └─────────────┴───────────┴───────────┴───────────┴──────────┘

  Topic Hierarchy:
  iPhone 15 Pro
  ├── Hardware
  │   ├── Camera (low-light, zoom, video, ProRes)
  │   ├── Display (brightness, ProMotion, always-on)
  │   ├── Battery (drain, charging, MagSafe)
  │   └── Thermals (overheating, gaming, throttling)
  ├── Software
  │   ├── iOS 17 (widgets, StandBy, updates)
  │   └── Apps (compatibility, optimization)
  └── Value
      ├── Pricing (expensive, upgrade-worthiness)
      └── Comparison (vs Samsung, vs Pixel)
```

---

## 4. Directory Structure

```
sentiment_analysis/absa/
├── pyproject.toml            # uv project config with all dependencies
├── README.md                 # GitHub README
├── PROJECT_PLAN.md           # This document
├── config/
│   ├── default.yaml          # Default configuration
│   └── aspects/              # Predefined aspect taxonomies
│       ├── smartphone.yaml
│       ├── laptop.yaml
│       └── audio.yaml
├── src/
│   ├── __init__.py
│   ├── cli.py                # Typer CLI entry point
│   ├── pipeline.py           # Pipeline orchestrator
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py      # PRAW Reddit data collection
│   │   └── preprocessor.py   # Text cleaning + sentence segmentation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── topic_model.py    # BERTopic + hierarchical reduction
│   │   ├── absa.py           # Multi-approach ABSA (transformer, LLM, lexicon)
│   │   └── aspect_mapper.py  # Raw aspect → canonical aspect mapping
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── aggregator.py     # Score aggregation at all levels
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── comparator.py     # Model comparison + metrics
│   └── utils/
│       ├── __init__.py
│       ├── config.py         # YAML config loader
│       └── display.py        # Rich terminal display helpers
├── data/
│   ├── raw/                  # Cached Reddit JSON data
│   ├── processed/            # Cleaned + segmented sentences
│   └── results/              # ABSA results + reports
├── outputs/
│   ├── reports/              # Generated terminal/HTML reports
│   └── visualizations/       # Topic hierarchy HTML, charts
└── tests/
    ├── test_collector.py
    ├── test_preprocessor.py
    ├── test_topic_model.py
    ├── test_absa.py
    └── test_aggregator.py
```

---

## 5. Dependencies

```toml
[project]
dependencies = [
    # CLI & Display
    "typer>=0.12",
    "rich>=13.0",

    # Data Collection
    "praw>=7.7",

    # NLP Core
    "spacy>=3.7",
    "demoji>=1.1",

    # Topic Modeling
    "bertopic>=0.16",
    "sentence-transformers>=3.0",
    "umap-learn>=0.5",
    "hdbscan>=0.8",

    # ABSA - Transformer approach
    "transformers>=4.40",
    "torch>=2.2",          # CPU-only
    "pyabsa>=2.4",

    # ABSA - Lexicon approach
    "vaderSentiment>=3.3",

    # ABSA - LLM approach (optional)
    "anthropic>=0.25",     # Claude API
    "openai>=1.30",        # OpenAI API (optional alternative)

    # Evaluation & Analysis
    "scikit-learn>=1.4",
    "pandas>=2.2",
    "numpy>=1.26",
    "gensim>=4.3",         # Topic coherence metrics

    # Visualization
    "plotly>=5.22",        # Interactive HTML charts
    "networkx>=3.3",       # Aspect hierarchy graph

    # Configuration
    "pyyaml>=6.0",
    "pydantic>=2.7",       # Config validation

    # Utilities
    "tqdm>=4.66",
    "tenacity>=8.3",       # Retry logic for API calls
]
```

---

## 6. Configuration System

```yaml
# config/default.yaml
reddit:
  client_id: ${REDDIT_CLIENT_ID}      # from env
  client_secret: ${REDDIT_CLIENT_SECRET}
  user_agent: "ABSA-TopicModel/1.0"
  default_limit: 100
  min_comments: 5
  min_score: 2
  time_filter: "year"

preprocessing:
  min_sentence_length: 5       # tokens
  max_sentence_length: 128     # tokens
  remove_urls: true
  remove_reddit_formatting: true
  expand_contractions: true

topic_model:
  embedding_model: "all-MiniLM-L6-v2"
  min_cluster_size: 15
  min_samples: 5
  nr_topics: "auto"
  top_n_words: 10
  ngram_range: [1, 3]
  diversity: 0.3

absa:
  approaches: ["transformer", "llm", "lexicon"]  # which to run
  transformer:
    model: "yangheng/deberta-v3-base-absa-v1.1"
    batch_size: 16
    confidence_threshold: 0.6
  llm:
    provider: "gemini"
    model: "gemini-3.1-flash-preview"
    batch_size: 10                  # sentences per API call
    llm_sample: 300                 # max sentences sent to LLM (cost cap)
    temperature: 0.0
  lexicon:
    use_vader: true
    spacy_model: "en_core_web_sm"

aggregation:
  weighting: "score_confidence"     # "uniform", "score", "confidence", "score_confidence"
  min_mentions: 3                   # minimum mentions to report an aspect

output:
  format: "terminal"                # "terminal", "html", "json"
  save_intermediate: true
  visualization: true
```

---

## 7. Implementation Order

### Phase 1: Foundation (Days 1-2)
1. Set up project structure, dependencies, config system
2. Implement data collection module with PRAW
3. Implement text preprocessing pipeline
4. Unit tests for data layer

### Phase 2: Topic Modeling (Days 3-4)
5. Implement BERTopic pipeline
6. Implement hierarchical topic reduction → aspect graph
7. Implement topic coherence evaluation
8. Add LDA baseline for comparison

### Phase 3: ABSA (Days 5-7)
9. Implement transformer-based ABSA (PyABSA / DeBERTa)
10. Implement LLM-based ABSA (Claude API)
11. Implement lexicon baseline (VADER + spaCy)
12. Implement aspect normalization + mapping to hierarchy

### Phase 4: Aggregation & Evaluation (Days 8-9)
13. Implement multi-level aggregation
14. Implement model comparison framework
15. Compute inter-annotator agreement, F1, coherence metrics

### Phase 5: CLI & Output (Days 10-11)
16. Build CLI with Typer
17. Build Rich terminal display (tables, progress bars, tree views)
18. Add HTML report generation (Plotly visualizations)
19. Export capabilities (JSON, CSV)

### Phase 6: Polish & Evaluation (Days 12-14)
20. End-to-end testing on 3+ products
21. Write evaluation section / comparison tables
22. Performance optimization (caching, batching)
23. Documentation and README finalization

---

## 8. Key Research Contributions / Novelty Points

For A*-conference positioning, the system contributes:

1. **Hierarchical Aspect Discovery:** Unsupervised aspect taxonomy construction via BERTopic hierarchical reduction — no predefined aspect lists needed, aspects emerge from data and organize into a hierarchy automatically.

2. **Multi-Signal Aggregation:** Leveraging Reddit's community voting (upvotes) as implicit quality signals for weighted sentiment aggregation — high-upvote comments carry more weight, reflecting community consensus.

3. **Cross-Paradigm ABSA Comparison:** Systematic comparison of three ABSA paradigms (fine-tuned transformer, LLM zero-shot, lexicon+dependency) on social media product reviews, with inter-annotator agreement as evaluation methodology when no gold labels exist.

4. **Domain-Adaptive Pipeline:** End-to-end system that adapts to any product category without manual aspect engineering — topic modeling discovers what users actually discuss, not what we assume they discuss.

---

## 9. Potential Extensions (Future Work)

- **Temporal Analysis:** Track aspect sentiment over time (e.g., battery sentiment before vs. after software update)
- **Comparative Analysis:** Side-by-side product comparison (iPhone 15 Pro vs. Galaxy S24 Ultra)
- **Fine-tuning:** Use LLM-labeled data as silver labels to fine-tune a small model (knowledge distillation)
- **Multimodal:** Incorporate image analysis from Reddit posts (product photos)
- **Interactive Dashboard:** Streamlit/Gradio web UI
- **Aspect-Opinion Pair Extraction:** Extract explicit (aspect, opinion_word) pairs for more granular analysis

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Reddit API rate limits | Built-in caching + exponential backoff + configurable limits |
| No GPU available | All models selected for CPU inference (MiniLM, DeBERTa-base) |
| LLM API costs | Configurable max calls + sampling for large datasets + Haiku is cheap |
| Poor topic coherence | Fallback to predefined aspect taxonomy from YAML config |
| PyABSA compatibility issues | Fallback to raw HuggingFace pipeline with same model |
| Insufficient Reddit data for niche products | Expand subreddit search + lower thresholds + inform user |
