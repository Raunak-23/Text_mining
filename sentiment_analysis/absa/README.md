# Hierarchical ABSA: Aspect-Based Sentiment Analysis with Neural Topic Modeling

A research-grade system that takes a **product name**, fetches real user discussions from **Reddit**, automatically discovers **aspect hierarchies** via neural topic modeling, and performs **multi-paradigm Aspect-Based Sentiment Analysis** to deliver both overall and aspect-level sentiment reports.

```
$ absa analyze "iPhone 15 Pro"

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
  │ Price/Value │ ██ 25%    │ █████ 55% │ ██ 20%    │ 167      │
  └─────────────┴───────────┴───────────┴───────────┴──────────┘

  Topic Hierarchy:
  iPhone 15 Pro
  ├── Hardware
  │   ├── Camera (low-light, zoom, video, ProRes)
  │   ├── Display (brightness, ProMotion, always-on)
  │   └── Battery (drain, charging, MagSafe)
  ├── Software
  │   └── iOS 17 (widgets, StandBy, updates)
  └── Value
      └── Pricing (expensive, upgrade-worthiness)
```

## Overview

Traditional ABSA systems require predefined aspect lists. This system **discovers aspects automatically** from Reddit discussions using BERTopic's hierarchical topic reduction, then maps extracted aspects to the discovered hierarchy for structured sentiment aggregation.

### Key Features

- **Automatic Aspect Discovery** — BERTopic with hierarchical reduction discovers what users actually discuss, organizing aspects into a tree structure (Product -> Category -> Aspect)
- **Multi-Paradigm ABSA** — Three approaches compared side-by-side:
  - Fine-tuned Transformer (DeBERTa / PyABSA)
  - LLM zero-shot extraction (Claude API)
  - Lexicon + Dependency Parsing baseline (VADER + spaCy)
- **Reddit-Native** — Leverages Reddit's community voting as quality signals for weighted sentiment aggregation
- **No GPU Required** — All models selected for CPU inference
- **Rich Terminal UI** — Publication-quality tables, charts, and tree visualizations in the terminal

## Architecture

```
Product Name
    |
    v
+----------------+     +------------------+     +------------------+
| Data Collection|---->|  Preprocessing   |---->|  Topic Modeling  |
| (PRAW)         |     | (spaCy)          |     | (BERTopic)       |
+----------------+     +------------------+     +--------+---------+
                                                         |
                           +-----------------------------+
                           | Aspect Hierarchy Graph       |
                           | (NetworkX DiGraph)           |
                           +-------------+---------------+
                                         |
                       +-----------------v-----------------+
                       |     ABSA (3 approaches)           |
                       |  Transformer | LLM | Lexicon      |
                       +-----------------+-----------------+
                                         |
                       +-----------------v-----------------+
                       |  Aspect Mapping + Aggregation     |
                       |  (Semantic similarity -> hierarchy)|
                       +-----------------+-----------------+
                                         |
                       +-----------------v-----------------+
                       |  Report + Evaluation              |
                       |  (Rich terminal / HTML / JSON)    |
                       +-----------------------------------+
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Raunak-23/Text_mining.git
cd Text_mining/sentiment_analysis/absa

# Install with uv
uv sync

# Download spaCy model
uv run python -m spacy download en_core_web_sm
```

## Configuration

```bash
# Set Reddit API credentials (required)
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"

# Set LLM API key (optional — only needed for LLM-based ABSA)
export ANTHROPIC_API_KEY="your_key"
```

Or configure via CLI:
```bash
uv run absa config set reddit.client_id <your_id>
uv run absa config set reddit.client_secret <your_secret>
```

## Usage

### Full Pipeline (Recommended)

```bash
# Analyze a product end-to-end
uv run absa analyze "iPhone 15 Pro"

# With options
uv run absa analyze "Galaxy S24 Ultra" \
    --subreddits r/Android,r/samsung,r/smartphones \
    --limit 200 \
    --approaches transformer,llm \
    --time-filter year
```

### Step-by-Step

```bash
# 1. Fetch Reddit data
uv run absa fetch "iPhone 15 Pro" --limit 200

# 2. Preprocess text
uv run absa preprocess --input data/raw/iphone-15-pro/

# 3. Run topic modeling (discover aspects)
uv run absa topics --input data/processed/iphone-15-pro/ --visualize

# 4. Run ABSA (all three approaches)
uv run absa sentiment --input data/processed/iphone-15-pro/ --approach all

# 5. Generate report
uv run absa report --input data/results/iphone-15-pro/

# 6. Compare approaches
uv run absa compare --input data/results/iphone-15-pro/
```

### Output Formats

```bash
# Terminal (default) — Rich tables and trees
uv run absa report --format terminal

# HTML — Interactive Plotly visualizations
uv run absa report --format html --output report.html

# JSON — Machine-readable
uv run absa report --format json --output report.json
```

## Project Structure

```
absa/
├── pyproject.toml
├── config/
│   ├── default.yaml              # Default configuration
│   └── aspects/                  # Predefined aspect taxonomies
│       ├── smartphone.yaml
│       ├── laptop.yaml
│       └── audio.yaml
├── src/
│   ├── cli.py                    # Typer CLI entry point
│   ├── pipeline.py               # Pipeline orchestrator
│   ├── data/
│   │   ├── collector.py          # Reddit data collection (PRAW)
│   │   └── preprocessor.py       # Text cleaning + sentence segmentation
│   ├── models/
│   │   ├── topic_model.py        # BERTopic + hierarchical reduction
│   │   ├── absa.py               # Multi-approach ABSA
│   │   └── aspect_mapper.py      # Aspect normalization -> hierarchy
│   ├── analysis/
│   │   └── aggregator.py         # Multi-level sentiment aggregation
│   ├── evaluation/
│   │   └── comparator.py         # Model comparison + metrics
│   └── utils/
│       ├── config.py             # Configuration management
│       └── display.py            # Rich terminal display
├── data/
│   ├── raw/                      # Cached Reddit data
│   ├── processed/                # Cleaned sentences
│   └── results/                  # ABSA outputs
├── outputs/
│   ├── reports/                  # Generated reports
│   └── visualizations/           # Topic hierarchy HTML
└── tests/
```

## Methodology

### Topic Modeling: BERTopic with Hierarchical Reduction

Unlike traditional LDA, BERTopic uses sentence-transformer embeddings to capture semantic similarity between short Reddit comments. The hierarchical topic reduction produces a **dendrogram** that naturally organizes discovered topics into a tree:

```
Product
├── High-level Category A
│   ├── Specific Aspect A1
│   └── Specific Aspect A2
└── High-level Category B
    ├── Specific Aspect B1
    └── Specific Aspect B2
```

This hierarchy is converted to a NetworkX directed graph for programmatic traversal and aggregation.

**Embedding Model:** `all-MiniLM-L6-v2` (384-dim, ~80MB, fast on CPU)

### ABSA: Three Paradigms Compared

| Approach | Model | Strengths | Limitations |
|----------|-------|-----------|-------------|
| **Transformer** | DeBERTa-v3-base (PyABSA) | Purpose-built for ABSA, high accuracy | Fixed aspect vocabulary |
| **LLM** | Claude Haiku (zero-shot) | Flexible, understands context, discovers novel aspects | API cost, latency |
| **Lexicon** | VADER + spaCy dependency parsing | Fast, interpretable, no model loading | Low recall, no context understanding |

### Evaluation Without Gold Labels

Since Reddit data lacks gold-standard ABSA annotations, we use:
1. **Inter-Annotator Agreement** — Cohen's Kappa between each pair of approaches
2. **LLM-as-Reference** — Treat LLM output as silver standard, compute P/R/F1
3. **Topic Coherence** — C_v coherence score (BERTopic vs. LDA baseline)
4. **Runtime + Cost Analysis** — Practical deployment considerations

### Weighted Aggregation

Reddit's upvote system provides implicit quality signals. Sentiment scores are weighted by:

```
weight = log1p(comment_score) * model_confidence
```

This amplifies well-received opinions and down-weights low-confidence predictions.

## Models Used

| Component | Model | Size | CPU Time |
|-----------|-------|------|----------|
| Embeddings | `all-MiniLM-L6-v2` | 80 MB | ~50ms/sentence |
| Topic Modeling | BERTopic (HDBSCAN + UMAP) | -- | ~30s for 5K docs |
| ABSA (Transformer) | `deberta-v3-base-absa` | 350 MB | ~100ms/sentence |
| ABSA (LLM) | Claude Haiku | API | ~500ms/batch |
| ABSA (Lexicon) | VADER + spaCy | 50 MB | ~5ms/sentence |
| Sentence Splitting | spaCy `en_core_web_sm` | 12 MB | ~10ms/doc |

## License

MIT

## Acknowledgments

- [BERTopic](https://maartengr.github.io/BERTopic/) — Neural topic modeling
- [PyABSA](https://github.com/yangheng95/PyABSA) — Aspect-based sentiment analysis
- [PRAW](https://praw.readthedocs.io/) — Reddit API wrapper
- [Rich](https://rich.readthedocs.io/) — Terminal formatting
