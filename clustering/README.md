# Incremental Centroid-Based Clustering for Real-Time Global Narrative Tracking  
## A Streaming Text Mining Pipeline for the GDELT 2.0 Global Knowledge Graph

## Introduction
This repository contains an advanced text mining system designed to track global news narratives in real-time. By leveraging the GDELT 2.0 Global Knowledge Graph (GKG), the system monitors a "firehose" of global activity, filtering for specific high-interest topics and organizing them into evolving clusters using Natural Language Processing (NLP) and Unsupervised Machine Learning.

The project solves the challenge of "Information Overload" by condensing thousands of individual news mentions into a few distinct, readable narratives.

## System Architecture
The pipeline is built on a Producer-Consumer architecture utilizing local file-based state persistence:

- **Data Ingestion Layer**: Polls GDELT every 15 minutes to synchronize with their update cycle.  
- **Processing Layer**: Filters raw tab-separated data, cleans URLs into human-readable titles, and handles high-dimensional vectorization.  
- **Analytics Layer**: Executes an incremental clustering algorithm that maintains a "living memory" of global news.  
- **Presentation Layer**: A dynamic web dashboard for real-time monitoring and historical volume analysis.  

## How to Run
Since the system is a live streaming pipeline, you must run the Backend Engine and the Frontend Dashboard concurrently in two separate terminal windows.

### 1. Environment Setup
Ensure you have all necessary libraries installed:

```bash
cd Text_mining/clustering
pip install -r requirements.txt
```

### 2. Start the Backend Engine (Terminal 1)
This script manages the "Heartbeat" cycle. It downloads GDELT data every 15 minutes, processes it, and updates the clustering model state.

```bash
# From the clustering directory
python main.py
```

You should see logs indicating:
- "Downloading latest GKG"
- "Processed X articles."

### 3. Start the Visualization Dashboard (Terminal 2)
Keep the first terminal running and open a second terminal. Launch the Streamlit app to view the live narratives.

```bash
# From the clustering directory
streamlit run dashboard/app.py
```

Access: Open the URL provided in the terminal (usually http://localhost:8501) to view the real-time clusters.

## Mathematical Methodology

### 1. Text Representation (Sentence Embeddings)
The system uses the Sentence-BERT (SBERT) architecture, specifically the `all-MiniLM-L6-v2` model. This model maps sentences and paragraphs into a 384-dimensional dense vector space.

- **Contextual Understanding**: Unlike traditional Word2Vec, this model understands that "The bank was closed" (financial) is different from "The river bank" (geographic).  
- **Efficiency**: Chosen for its high performance-to-latency ratio, making it ideal for streaming applications.  

### 2. Incremental Clustering Logic
Instead of static K-Means, we use a custom Centroid-Based Incremental Clusterer:

- **Similarity Metric**: We calculate the Cosine Similarity between the new article vector ($A$) and existing cluster centroids ($C$):

$$
\text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{C}}{\|\mathbf{A}\| \|\mathbf{C}\|}
$$

- **Centroid Evolution**: If an article matches a cluster (Similarity > Threshold), the centroid $C$ is updated using a weighted moving average:

$$
C_{new} = (C_{old} \times 0.95) + (A \times 0.05)
$$

This $0.95$ decay factor ensures that the "narrative" can shift slightly as the story develops over time.

## Data Pipeline Details

### GDELT GKG Schema Utilization
The system extracts the following specific fields from the GDELT stream:

| Field | Purpose |
|------|--------|
| DocumentIdentifier | Used to extract the source URL and generate the Narrative Title |
| V2Themes | Used for keyword-based topic filtering (Blockchain, Tech, IPL) |
| V2Tone | (Future Scope) For calculating the sentiment of specific narratives |
| GKGRECORDID | Unique identifier for deduplication within the 15-minute window |

## Dashboard Features

- **Narrative Volume Distribution**: A bar chart showing which news stories are currently dominating the global conversation.  
- **Top Trending Table**: A sorted view of current narratives with their respective article counts.  
- **Auto-Refresh**: The UI utilizes `st_autorefresh` to stay in sync with the backend every 15 minutes.  
