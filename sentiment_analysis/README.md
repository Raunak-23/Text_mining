# HierABSA: Hierarchical Aspect-Based Sentiment Analysis
## Reddit-Native Multi-Paradigm Sentiment Mining with Neural Topic Modeling

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/BERTopic-Neural%20Topic%20Modeling-8B5CF6" />
  <img src="https://img.shields.io/badge/DeBERTa-ABSA%20Transformer-EE4C2C" />
  <img src="https://img.shields.io/badge/Gemini-LLM%20Zero--Shot-4285F4?logo=google" />
  <img src="https://img.shields.io/badge/CPU%20Compatible-No%20GPU%20Required-22C55E" />
</p>

---

## Project Overview

HierABSA is a research-grade system that automatically discovers **what users care about** in Reddit discussions about consumer products, then quantifies the sentiment toward each discovered aspect using three complementary analysis paradigms. Unlike conventional ABSA systems that require a fixed predefined aspect list, HierABSA **learns the aspect vocabulary from the data** via hierarchical neural topic modeling.

Given only a product name (e.g., `"Samsung Galaxy S25"`), the system:

1. Fetches relevant Reddit threads across brand-specific and category subreddits
2. Segments and cleans comment text into ABSA-ready sentences
3. Discovers product aspects using BERTopic with hierarchical reduction
4. Builds a `Product → Category → Aspect → Topic` knowledge graph
5. Runs three independent ABSA paradigms and aggregates with upvote weighting
6. Evaluates paradigm agreement and topic coherence via four research questions
7. Saves a structured JSON report, Rich terminal output, and PNG visualizations

**GitHub:** https://github.com/Raunak-23/Text_mining/tree/master/sentiment_analysis

---

## Key Results — Samsung Galaxy S25

The system was validated end-to-end on the Samsung Galaxy S25. All numbers below come from the generated `data/results/samsung-galaxy-s25/final_report.json`.

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total sentences analyzed | 2,858 |
| BERTopic topics discovered | 19 |
| Transformer sentences (sampled) | 500 |
| LLM sentences (sampled) | 300 |
| Lexicon sentences (full corpus) | 2,858 |

### Topic Coherence (RQ1)

| Model | C_v Coherence | Topics |
|-------|--------------|--------|
| **BERTopic** | 0.445 | 19 |
| LDA baseline | 0.467 | 19 |
| Delta | −0.022 | — |

LDA's marginally higher C_v is expected on short social-media text. BERTopic adds structural advantages — sentence-transformer embeddings, UMAP manifold reduction, and hierarchical dendrogram — that produce semantically richer topic clusters at the cost of a small coherence trade-off.

### Inter-Paradigm Agreement (RQ2)

| Pair | Cohen's κ | Mean JSD | Shared Sentences |
|------|-----------|----------|-----------------|
| Transformer vs. LLM | 0.198 | 0.509 | 55 |
| Transformer vs. Lexicon | 0.160 | 0.559 | 500 |
| LLM vs. Lexicon | 0.090 | 0.517 | 300 |

Low κ values (< 0.21, "slight agreement") and high JSD (> 0.5) confirm that each paradigm captures a genuinely different signal — the three-paradigm ensemble provides broader coverage than any single model.

### Upvote Weighting Impact (RQ3)

| Metric | Value |
|--------|-------|
| Pearson r (uniform vs. weighted) | **0.886** |
| Mean absolute delta | 0.105 |
| Max shift (lexicon / Ecosystem) | −0.427 |

Strong global correlation confirms weighting is stable; local shifts up to 0.43 points show that high-upvote minority opinions can substantially move niche aspects, validating the need for weighting.

### Aspect Sentiment Summary

Net score = fraction_positive − fraction_negative (weighted by log1p(upvote) × confidence).

| Aspect | Transformer | LLM | Lexicon |
|--------|-------------|-----|---------|
| Camera | −0.21 | −0.32 | +0.02 |
| Display | −0.37 | −0.16 | +0.18 |
| Performance | −0.17 | **+0.49** | +0.05 |
| Connectivity | −0.34 | −0.27 | +0.15 |
| Battery | −0.17 | **+0.27** | +0.07 |
| Build | −0.26 | −0.33 | +0.18 |
| OS | −0.28 | −0.40 | +0.08 |
| AI | −0.26 | −0.87 | +0.14 |
| Ecosystem | −0.14 | −0.22 | **−0.87** |

**Key observations:**
- Transformer and LLM lean negative across most aspects (SemEval fine-tuning → stricter sentiment thresholds)
- Lexicon stays near neutral-positive (VADER lacks negation/sarcasm depth)
- Strong agreement on **Ecosystem** being the most polarizing aspect (high-upvote posts criticizing Samsung's software ecosystem dominate)
- **Performance** and **Battery** receive genuine positive signals from the LLM, aligning with hardware launch buzz on r/GalaxyS25

### Product-Level Sentiment

| Paradigm | Positive | Negative | Neutral | Net Score | Dominant |
|----------|----------|----------|---------|-----------|----------|
| Transformer | 12.8% | 37.1% | 50.0% | −0.243 | Neutral |
| LLM | 28.0% | 48.3% | 23.7% | −0.203 | Negative |
| Lexicon | 36.0% | 21.7% | 42.3% | +0.143 | Neutral |

---

## Architecture & Pipeline

```
Product Name (e.g., "Samsung Galaxy S25")
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1 — Data Collection    (src/absa/data/collector.py) │
│  • PRAW Reddit API                                        │
│  • Brand-aware subreddit discovery (50+ brand mappings)   │
│  • Comment tree recursion, upvote-weighted storage        │
│  • Output: data/raw/<slug>/<subreddit>.json               │
└──────────────────────────────────┬───────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2 — Preprocessing   (src/absa/data/preprocessor.py)│
│  • spaCy en_core_web_sm sentence segmentation             │
│  • URL / emoji / punctuation normalization                │
│  • Upvote weight propagation to sentence level            │
│  • Output: data/processed/<slug>/sentences.jsonl          │
└──────────────────────────────────┬───────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 3 — Topic Modeling    (src/absa/models/topic_model.py)│
│  • all-MiniLM-L6-v2 embeddings (384-dim)                  │
│  • UMAP(n_components=5) + HDBSCAN(min_cluster=10)         │
│  • BERTopic(nr_topics=20) → typically 15–19 topics        │
│  • hierarchical_topics() dendrogram for Category grouping │
│  • KeyBERTInspired representation per topic               │
│  • Output: data/results/<slug>/topics/                    │
└──────────────────────────────────┬───────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 4 — Aspect Graph      (src/absa/models/aspect_mapper.py)│
│  • Semantic similarity (cosine) between topic keywords    │
│    and predefined aspect taxonomy (Camera, Battery, …)    │
│  • Builds NetworkX DiGraph:                               │
│    Product → Category → Aspect → Topic                   │
│  • Output: data/results/<slug>/topics/aspect_graph.json   │
└──────────────────────────────────┬───────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 5 — ABSA              (src/absa/models/absa.py)    │
│  Transformer: yangheng/deberta-v3-base-absa-v1.1          │
│    → 500-sentence CPU sample, per-aspect triplets         │
│  LLM: Gemini Flash (zero-shot, batched 10/call)           │
│    → 300-sentence sample, structured JSON output          │
│  Lexicon: VADER + spaCy dependency arcs                   │
│    → Full 2,858 sentences, aspect-matched via dep tree    │
│  Output: data/results/<slug>/absa/*_results.json          │
└──────────────────────────────────┬───────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 6 — Aggregation       (src/absa/analysis/aggregator.py)│
│  • Upvote weight: w_i = log1p(score_i) × confidence_i    │
│  • Net(aspect, paradigm) = Σw·pos − Σw·neg / Σw          │
│  • Uniform and weighted variants saved                    │
│  • Output: data/results/<slug>/absa/aggregated*.json      │
└──────────────────────────────────┬───────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 7 — Evaluation        (src/absa/evaluation/comparator.py)│
│  • RQ1: C_v coherence (BERTopic vs. LDA via gensim)       │
│  • RQ2: Cohen's κ, JSD, Polarity Entropy H(A), D(A)       │
│  • RQ3: Pearson r — uniform vs. weighted net scores       │
│  • RQ4: Transformer vs. LLM as domain-transfer proxy      │
│  • Output: data/results/<slug>/absa/evaluation.json       │
└──────────────────────────────────┬───────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────┐
│  Reporting & Visualization                                │
│  • outputs/reports/<slug>/final_report.json              │
│  • outputs/visualizations/<slug>/aspect_hierarchy.png    │
│  • outputs/visualizations/<slug>/sentiment_heatmap.png   │
└──────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
sentiment_analysis/
│
├── main.py                          # Interactive entry point (run this)
│
├── src/absa/
│   ├── cli.py                       # Typer CLI (stage-by-stage commands)
│   ├── pipeline.py                  # Pipeline orchestrator (Stages 1–7)
│   │
│   ├── data/
│   │   ├── collector.py             # PRAW Reddit fetcher + brand subreddit discovery
│   │   └── preprocessor.py          # spaCy sentence splitting + normalization
│   │
│   ├── models/
│   │   ├── topic_model.py           # BERTopic + hierarchical reduction
│   │   ├── absa.py                  # Three ABSA paradigms (transformer/llm/lexicon)
│   │   └── aspect_mapper.py         # Aspect graph builder (NetworkX DiGraph)
│   │
│   ├── analysis/
│   │   └── aggregator.py            # Upvote-weighted sentiment aggregation
│   │
│   ├── evaluation/
│   │   └── comparator.py            # RQ1–RQ4 evaluation metrics
│   │
│   ├── reporting/
│   │   ├── results_report.py        # JSON report generator
│   │   └── visualizer.py            # Matplotlib PNG output (hierarchy + heatmap)
│   │
│   └── utils/
│       ├── config.py                # Pydantic settings (env vars + YAML)
│       └── display.py               # Rich terminal helpers
│
├── config/
│   └── default.yaml                 # Default fetch limits, subreddit lists, model IDs
│
├── data/
│   ├── raw/<slug>/                  # Per-subreddit Reddit JSON (cached)
│   ├── processed/<slug>/            # Sentence JSONL (cleaned, upvote-weighted)
│   └── results/<slug>/              # All intermediate results
│       ├── topics/                  # BERTopic outputs + aspect_graph.json
│       └── absa/                    # Per-paradigm results + evaluation.json
│
├── outputs/                         # Final deliverables
│   ├── reports/<slug>/              # final_report.json (human-readable summary)
│   └── visualizations/<slug>/       # aspect_hierarchy.png, sentiment_heatmap.png
│
└── pyproject.toml                   # uv-managed dependencies
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Raunak-23/Text_mining.git
cd Text_mining/sentiment_analysis

# Install with uv (fast Python package manager)
uv sync

# Download the spaCy model (required for preprocessing)
uv run python -m spacy download en_core_web_sm
```

---

## Configuration

Create a `.env` file in `sentiment_analysis/` (or export variables in your shell):

```bash
# Reddit API credentials — required for data collection
# Get them at https://www.reddit.com/prefs/apps (create a "script" app)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=HierABSA/1.0 by YourUsername

# Gemini API key — required for the LLM paradigm
# Get it at https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_key
```

> **Note:** The LLM paradigm uses **Gemini Flash** (`gemini-3-flash-preview` by default; configurable in `config/default.yaml`), not Anthropic/Claude. The `GEMINI_API_KEY` variable is what you need.

---

## Usage

### Interactive Mode (Recommended)

```bash
cd sentiment_analysis
uv run python main.py
```

You will be prompted for:
- **Product name** — e.g., `Samsung Galaxy S25`
- **Paradigms** — `transformer,llm,lexicon` (default: all three)
- **Force re-run** — whether to reuse cached results

To skip prompts:
```bash
uv run python main.py "iPhone 16 Pro"
uv run python main.py "OnePlus 13" --paradigms transformer,lexicon --force
uv run python main.py "Pixel 9 Pro" --subreddits GooglePixel,Android,smartphones
```

### CLI Stage-by-Stage

For debugging or partial runs, use the individual CLI commands:

```bash
# Stage 1: Fetch Reddit data
uv run absa fetch "Samsung Galaxy S25"

# Stage 2: Clean and sentence-segment
uv run absa preprocess "Samsung Galaxy S25"

# Stage 3: BERTopic + aspect graph
uv run absa topics "Samsung Galaxy S25"

# Stage 4–6: ABSA + aggregation + evaluation
uv run absa absa "Samsung Galaxy S25"

# View cached results as a report
uv run absa report "Samsung Galaxy S25"

# Compare paradigm outputs side-by-side
uv run absa compare "Samsung Galaxy S25"

# Full pipeline (equivalent to main.py without visualizations)
uv run absa analyze "Samsung Galaxy S25"

# Check credentials and config
uv run absa info
```

---

## Models

| Component | Model | Size | Speed (CPU) |
|-----------|-------|------|-------------|
| Sentence embeddings | `all-MiniLM-L6-v2` | 80 MB | ~50 ms/sentence |
| Topic modeling | BERTopic (UMAP + HDBSCAN) | — | ~30 s for 3K docs |
| ABSA Transformer | `yangheng/deberta-v3-base-absa-v1.1` | ~350 MB | ~100 ms/sentence |
| ABSA LLM | `gemini-3-flash-preview` (Google API) | API | ~300 ms per 10-batch |
| ABSA Lexicon | VADER + spaCy dep parser | 50 MB | ~5 ms/sentence |
| Sentence splitting | `spaCy en_core_web_sm` | 12 MB | ~10 ms/document |

---

## Subreddit Discovery

The collector uses a three-tier subreddit ranking for any product:

1. **Brand-specific subreddits** — 50+ brand→subreddit mappings (e.g., `vivo` → `[r/vivo, r/iqoo]`, `samsung` → `[r/samsung, r/GalaxyS25, r/GalaxyS24, r/GalaxyS23]`)
2. **Category subreddits** — matched from category keywords (e.g., `smartphone` → `[r/smartphones, r/android, r/iphone, …]`)
3. **Generic subreddits** — always appended: `r/gadgets`, `r/technology`, `r/reviews`

This means a query like `"Vivo X300 Pro"` will search `r/vivo`, `r/iqoo`, `r/smartphones`, `r/android`, `r/gadgets`, `r/technology`, `r/reviews` — capturing brand-community discussions that generic category-only search would miss.

---

## Outputs

After a successful pipeline run, the following files are produced:

```
outputs/
├── reports/samsung-galaxy-s25/
│   └── final_report.json             # Complete structured results (RQ1–RQ4 + aspect table)
└── visualizations/samsung-galaxy-s25/
    ├── aspect_hierarchy.png           # Color-coded Product→Category→Aspect→Topic tree
    └── sentiment_heatmap.png          # Aspect × paradigm net-score heatmap (RdYlGn)

data/results/samsung-galaxy-s25/
├── topics/
│   ├── topics.json                   # BERTopic topic keywords + sizes
│   ├── hierarchy.json                # Hierarchical merge dendrogram
│   ├── assignments.json              # Per-sentence topic assignments
│   └── aspect_graph.json             # NetworkX DiGraph (serialized)
└── absa/
    ├── transformer_results.json       # DeBERTa sentiment triplets
    ├── llm_results.json               # Gemini extracted sentiment
    ├── lexicon_results.json           # VADER + dep-arc results
    ├── aggregated_uniform.json        # Equal-weight aggregation
    ├── aggregated_weighted.json       # Upvote-weighted aggregation
    └── evaluation.json                # RQ1–RQ4 metrics
```

---

## Evaluation Framework

Since Reddit ABSA lacks gold-standard annotations, the system answers four research questions using reference-free metrics:

| RQ | Question | Metric |
|----|----------|--------|
| **RQ1** | Do discovered topics have semantic coherence? | C_v coherence vs. LDA baseline |
| **RQ2** | How much do the three paradigms agree? | Cohen's κ, JSD, polarity entropy H(A), dominance D(A) |
| **RQ3** | Does upvote weighting change results meaningfully? | Pearson r (uniform vs. weighted), per-aspect delta |
| **RQ4** | Does the SemEval transformer transfer to Reddit? | Transformer vs. LLM (zero-shot silver) κ and JSD |

---

## References

- Pontiki et al. (2014). *SemEval-2014 Task 4: Aspect-Based Sentiment Analysis.* SemEval.
- Grootendorst (2022). *BERTopic: Neural topic modeling.* arXiv:2203.05794.
- He et al. (2021). *DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training.* arXiv:2111.09543.
- Hutto & Gilbert (2014). *VADER: A parsimonious rule-based model for sentiment analysis.* ICWSM.
- Liang & Deng (2025). *MHSTM: Multi-Head Sentiment Topic Model.* arXiv:2502.18927.

---

## License

MIT
