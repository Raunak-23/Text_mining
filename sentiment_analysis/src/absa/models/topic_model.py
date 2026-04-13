"""
Neural topic modeling with BERTopic + hierarchical reduction.

Pipeline:
  sentences.json
      -> embed with all-MiniLM-L6-v2
      -> UMAP (dim reduction, CPU-friendly params)
      -> HDBSCAN (density clustering)
      -> BERTopic (topic extraction + c-TF-IDF)
      -> hierarchical_topics() (agglomerative dendrogram)
      -> TopicModelResult (topics, hierarchy, per-sentence assignments)

Outputs saved under data/results/<slug>/topics/:
  topics.json          — topic id, label, top words, doc count
  hierarchy.json       — parent/child edges from dendrogram
  assignments.json     — per-sentence {sentence, topic_id, topic_label}
  topic_model/         — serialised BERTopic model (reusable)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from absa.utils.config import settings
from absa.utils.display import print_info, print_success, print_warning


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Topic:
    id: int
    label: str                     # human-readable: "camera_photo_zoom"
    top_words: list[str]
    top_word_scores: list[float]
    doc_count: int
    representative_docs: list[str]


@dataclass
class HierarchyEdge:
    parent_id: int
    child_id: int
    parent_label: str
    child_label: str
    distance: float                # agglomerative merge distance


@dataclass
class TopicModelResult:
    topics: list[Topic]
    hierarchy: list[HierarchyEdge]
    # per-sentence assignments: list of (sentence_idx, topic_id)
    assignments: list[dict[str, Any]] = field(default_factory=list)
    coherence_score: float | None = None
    outlier_ratio: float = 0.0


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _build_components(
    embedding_model_name: str,
    min_topic_size: int,
    nr_topics: str | int,
) -> tuple[SentenceTransformer, UMAP, HDBSCAN, CountVectorizer, BERTopic]:
    """Construct BERTopic with CPU-optimised hyperparameters."""

    embedder = SentenceTransformer(embedding_model_name)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,      # BERTopic default; keep low for CPU
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=2,
        ngram_range=(1, 2),
    )

    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=representation_model,
        nr_topics=nr_topics,
        top_n_words=settings._yaml.get("topic_model", {}).get("top_n_words", 10),
        calculate_probabilities=False,
        verbose=False,
    )
    return embedder, umap_model, hdbscan_model, vectorizer, topic_model


def _label_from_words(words: list[str]) -> str:
    """Create a short readable label from top topic words."""
    return "_".join(words[:3])


def _extract_topics(
    model: BERTopic,
    docs: list[str],
    topic_ids: list[int],
) -> list[Topic]:
    """Convert BERTopic internals into our Topic dataclass list."""
    topic_info = model.get_topic_info()
    topics: list[Topic] = []

    for _, row in topic_info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue  # skip outlier cluster

        word_scores = model.get_topic(tid) or []
        words  = [w for w, _ in word_scores]
        scores = [float(s) for _, s in word_scores]
        count  = int(row["Count"])
        rep_docs = row.get("Representative_Docs", []) or []

        topics.append(
            Topic(
                id=tid,
                label=_label_from_words(words),
                top_words=words,
                top_word_scores=scores,
                doc_count=count,
                representative_docs=list(rep_docs)[:3],
            )
        )

    return sorted(topics, key=lambda t: t.doc_count, reverse=True)


def _extract_hierarchy(
    model: BERTopic,
    topic_list: list[Topic],
    docs: list[str],
) -> list[HierarchyEdge]:
    """
    Build hierarchy edges from BERTopic's hierarchical_topics() dendrogram.

    BERTopic's hierarchical_topics() returns a DataFrame with columns:
      Parent_ID, Parent_Name, Child_Left_ID, Child_Left_Name,
      Child_Right_ID, Child_Right_Name, Distance, Topics
    Each row is an agglomerative merge between two child clusters.
    We emit two edges per row: Parent->Left and Parent->Right.

    IMPORTANT: `docs` must be the exact same list passed to fit_transform().
    """
    try:
        hier_df = model.hierarchical_topics(docs)
    except Exception as exc:
        print_warning(f"Hierarchy extraction failed: {exc}")
        return []

    id_to_label: dict[int, str] = {t.id: t.label for t in topic_list}

    edges: list[HierarchyEdge] = []
    seen: set[tuple[int, int]] = set()

    for _, row in hier_df.iterrows():
        parent_id = int(row["Parent_ID"])
        parent_label = str(row.get("Parent_Name", f"topic_{parent_id}"))
        distance = float(row.get("Distance", 0.0))

        for side in ("Child_Left", "Child_Right"):
            child_id = int(row[f"{side}_ID"])
            child_label = id_to_label.get(child_id, str(row.get(f"{side}_Name", f"topic_{child_id}")))
            key = (parent_id, child_id)
            if key not in seen:
                seen.add(key)
                edges.append(HierarchyEdge(
                    parent_id=parent_id,
                    child_id=child_id,
                    parent_label=parent_label,
                    child_label=child_label,
                    distance=distance,
                ))

    return edges


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_topic_model(
    sentences: list[dict[str, Any]],
    out_dir: Path,
    force: bool = False,
) -> TopicModelResult:
    """
    Fit BERTopic on *sentences*, extract hierarchy, save all outputs.

    Parameters
    ----------
    sentences :
        Records from preprocessor — must contain a 'sentence' key.
    out_dir :
        Directory to write topics.json / hierarchy.json / assignments.json.
    force :
        Re-fit even if cached results exist.
    """
    topics_file      = out_dir / "topics.json"
    hierarchy_file   = out_dir / "hierarchy.json"
    assignments_file = out_dir / "assignments.json"
    model_dir        = out_dir / "topic_model"

    # ---- Cache hit ----
    if (
        not force
        and topics_file.exists()
        and hierarchy_file.exists()
        and assignments_file.exists()
    ):
        print_info("Topic model results already cached — loading from disk.")
        topics    = [Topic(**t) for t in json.loads(topics_file.read_text(encoding="utf-8"))]
        hierarchy = [HierarchyEdge(**e) for e in json.loads(hierarchy_file.read_text(encoding="utf-8"))]
        assignments = json.loads(assignments_file.read_text(encoding="utf-8"))
        result = TopicModelResult(topics=topics, hierarchy=hierarchy, assignments=assignments)
        print_info(f"Loaded {len(topics)} topics, {len(hierarchy)} hierarchy edges.")
        return result

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Extract docs ----
    docs = [s["sentence"] for s in sentences]
    print_info(f"Fitting BERTopic on {len(docs)} sentences…")

    cfg = settings._yaml.get("topic_model", {})
    embedding_model_name = cfg.get("embedding_model", "all-MiniLM-L6-v2")
    min_topic_size       = cfg.get("min_topic_size", 5)
    nr_topics = cfg.get("nr_topics", "auto")
    if isinstance(nr_topics, str) and nr_topics.isdigit():
        nr_topics = int(nr_topics)

    _, _, _, _, topic_model = _build_components(
        embedding_model_name, min_topic_size, nr_topics
    )

    # ---- Fit ----
    print_info("Embedding sentences with all-MiniLM-L6-v2…")
    embedder = SentenceTransformer(embedding_model_name)
    embeddings = embedder.encode(
        docs,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print_info("Running UMAP + HDBSCAN + c-TF-IDF…")
    topic_ids, _ = topic_model.fit_transform(docs, embeddings=embeddings)

    # ---- Extract ----
    topics    = _extract_topics(topic_model, docs, topic_ids)
    hierarchy = _extract_hierarchy(topic_model, topics, docs)

    n_topics   = len(topics)
    n_outliers = sum(1 for t in topic_ids if t == -1)
    outlier_ratio = n_outliers / max(len(docs), 1)

    print_success(f"Discovered {n_topics} topics  |  outlier ratio: {outlier_ratio:.1%}")
    print_info(f"Hierarchy: {len(hierarchy)} merge edges")

    # ---- Per-sentence assignments ----
    id_to_label = {t.id: t.label for t in topics}
    assignments = [
        {
            "sentence":    docs[i],
            "post_id":     sentences[i].get("post_id"),
            "comment_id":  sentences[i].get("comment_id"),
            "subreddit":   sentences[i].get("subreddit"),
            "post_score":  sentences[i].get("post_score"),
            "comment_score": sentences[i].get("comment_score"),
            "topic_id":    int(tid),
            "topic_label": id_to_label.get(int(tid), "outlier"),
        }
        for i, tid in enumerate(topic_ids)
    ]

    result = TopicModelResult(
        topics=topics,
        hierarchy=hierarchy,
        assignments=assignments,
        outlier_ratio=outlier_ratio,
    )

    # ---- Persist ----
    topics_file.write_text(
        json.dumps([asdict(t) for t in topics], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    hierarchy_file.write_text(
        json.dumps([asdict(e) for e in hierarchy], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    assignments_file.write_text(
        json.dumps(assignments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Save model for reuse in later stages
    try:
        topic_model.save(str(model_dir), serialization="safetensors", save_ctfidf=True)
        print_success(f"Model saved -> {model_dir.relative_to(out_dir.parent.parent)}")
    except Exception as exc:
        print_warning(f"Model serialization failed (non-fatal): {exc}")

    print_success(f"All topic outputs -> {out_dir.relative_to(out_dir.parent.parent)}")
    return result
