"""
Aspect Mapper — converts BERTopic hierarchy into a NetworkX DiGraph.

Two-step process:
  1. Build a raw BERTopic tree from hierarchy edges (merge dendrogram).
  2. Map each leaf topic to the nearest node in the domain taxonomy
     (config/aspects/<category>.yaml) using cosine similarity of
     MiniLM sentence embeddings, then graft onto a canonical
     Product -> Category -> Aspect skeleton.

Output graph schema (NetworkX DiGraph):
  Nodes carry attributes:
    type        : "product" | "category" | "aspect" | "topic"
    label       : human-readable name
    topic_ids   : list of BERTopic topic IDs mapped here (leaf nodes)
    doc_count   : total sentences assigned to this node
    top_words   : aggregated top words (aspects/topics)

  Edges carry:
    weight      : sum of child doc_counts (for traversal scoring)

A JSON snapshot is saved alongside the topics outputs for later
aggregation and visualisation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from absa.models.topic_model import HierarchyEdge, Topic, TopicModelResult
from absa.utils.config import settings
from absa.utils.display import print_info, print_success, print_warning


# ---------------------------------------------------------------------------
# Taxonomy loader
# ---------------------------------------------------------------------------

def _load_taxonomy(category_hint: str = "smartphone") -> dict[str, Any]:
    """
    Load config/aspects/<hint>.yaml.
    Falls back gracefully if the file doesn't exist.
    """
    tax_dir = settings.root / "config" / "aspects"
    path = tax_dir / f"{category_hint}.yaml"
    if not path.exists():
        print_warning(f"Taxonomy file not found: {path}. Using empty taxonomy.")
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _guess_taxonomy_key(product: str) -> str:
    product_lower = product.lower()
    for key in ("smartphone", "laptop", "headphone", "tv", "camera"):
        if key in product_lower or key.rstrip("e") in product_lower:
            return key
    return "smartphone"   # default


# ---------------------------------------------------------------------------
# Semantic similarity mapper
# ---------------------------------------------------------------------------

class _SimilarityMapper:
    """
    Maps a topic (represented by its top words) to the nearest taxonomy
    aspect node using cosine similarity of MiniLM embeddings.
    """

    def __init__(self, embedder: SentenceTransformer) -> None:
        self._embedder = embedder
        self._aspect_labels: list[str] = []
        self._aspect_embeddings: np.ndarray | None = None
        # (category, aspect) pairs parallel to _aspect_labels
        self._aspect_index: list[tuple[str, str]] = []

    def build_index(self, taxonomy: dict[str, Any]) -> None:
        """Pre-embed all aspect keyword phrases from the taxonomy."""
        labels: list[str] = []
        index: list[tuple[str, str]] = []

        categories = taxonomy.get("categories", {})
        for cat_name, cat_data in categories.items():
            aspects = cat_data.get("aspects", {})
            for asp_name, asp_data in aspects.items():
                keywords: list[str] = asp_data.get("keywords", [asp_name])
                # One representative phrase per aspect = joined keywords
                phrase = " ".join(keywords[:8])
                labels.append(phrase)
                index.append((cat_name, asp_name))

        if not labels:
            return

        self._aspect_labels  = labels
        self._aspect_index   = index
        self._aspect_embeddings = self._embedder.encode(
            labels, normalize_embeddings=True, show_progress_bar=False
        )

    def map_topic(
        self,
        topic: Topic,
        threshold: float = 0.18,
        debug: bool = False,
    ) -> tuple[str, str] | None:
        """
        Return (category, aspect) for the nearest taxonomy node, or None
        if the best similarity is below *threshold* (topic is off-taxonomy).

        Query = representative sentences (richer signal) + top keyword phrase.
        """
        if self._aspect_embeddings is None or len(self._aspect_labels) == 0:
            return None

        # Build a rich query: use representative docs first, fall back to keywords
        query_parts: list[str] = []
        if topic.representative_docs:
            query_parts.extend(topic.representative_docs[:2])
        query_parts.append(" ".join(topic.top_words[:8]))
        query = " ".join(query_parts)

        query_emb = self._embedder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        sims = (query_emb @ self._aspect_embeddings.T).flatten()
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if debug:
            top3_idx = np.argsort(sims)[::-1][:3]
            print_info(
                f"Topic '{topic.label[:40]}' top matches: "
                + " | ".join(
                    f"{self._aspect_index[i]} {sims[i]:.3f}" for i in top3_idx
                )
            )

        if best_sim < threshold:
            return None
        return self._aspect_index[best_idx]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_aspect_graph(
    result: TopicModelResult,
    product: str,
    taxonomy_key: str | None = None,
    sim_threshold: float = 0.18,
    debug: bool = False,
) -> nx.DiGraph:
    """
    Build a Product -> Category -> Aspect -> Topic DiGraph.

    Parameters
    ----------
    result        : output from run_topic_model()
    product       : product name string (root node label)
    taxonomy_key  : override taxonomy YAML filename stem
    sim_threshold : minimum cosine similarity to map a topic to an aspect

    Returns
    -------
    nx.DiGraph with full hierarchy
    """
    taxonomy_key = taxonomy_key or _guess_taxonomy_key(product)
    taxonomy     = _load_taxonomy(taxonomy_key)

    embedder = SentenceTransformer(
        settings._yaml.get("topic_model", {}).get("embedding_model", "all-MiniLM-L6-v2")
    )
    mapper = _SimilarityMapper(embedder)
    mapper.build_index(taxonomy)

    G = nx.DiGraph()

    # ---- Root node ----
    G.add_node(product, type="product", label=product, topic_ids=[], doc_count=0, top_words=[])

    # ---- Category nodes from taxonomy ----
    categories_meta = taxonomy.get("categories", {})
    for cat_name, cat_data in categories_meta.items():
        G.add_node(cat_name, type="category", label=cat_name, topic_ids=[], doc_count=0, top_words=[])
        G.add_edge(product, cat_name, weight=0)

        for asp_name in cat_data.get("aspects", {}):
            G.add_node(asp_name, type="aspect", label=asp_name, topic_ids=[], doc_count=0, top_words=[])
            G.add_edge(cat_name, asp_name, weight=0)

    # ---- Map each topic to the nearest aspect ----
    unmapped_topics: list[Topic] = []

    for topic in result.topics:
        mapping = mapper.map_topic(topic, threshold=sim_threshold, debug=debug)

        if mapping is None:
            unmapped_topics.append(topic)
            continue

        cat_name, asp_name = mapping

        # Ensure nodes exist (in case taxonomy was empty)
        if cat_name not in G:
            G.add_node(cat_name, type="category", label=cat_name, topic_ids=[], doc_count=0, top_words=[])
            G.add_edge(product, cat_name, weight=0)
        if asp_name not in G:
            G.add_node(asp_name, type="aspect", label=asp_name, topic_ids=[], doc_count=0, top_words=[])
            G.add_edge(cat_name, asp_name, weight=0)

        # Add leaf topic node
        topic_node = f"topic_{topic.id}"
        G.add_node(
            topic_node,
            type="topic",
            label=topic.label,
            topic_ids=[topic.id],
            doc_count=topic.doc_count,
            top_words=topic.top_words,
        )
        G.add_edge(asp_name, topic_node, weight=topic.doc_count)

        # Accumulate doc counts up the hierarchy
        G.nodes[asp_name]["topic_ids"].append(topic.id)
        G.nodes[asp_name]["doc_count"] += topic.doc_count
        G.nodes[asp_name]["top_words"] = list(dict.fromkeys(
            G.nodes[asp_name]["top_words"] + topic.top_words
        ))[:10]
        G.nodes[cat_name]["doc_count"] += topic.doc_count
        G.nodes[product]["doc_count"] += topic.doc_count
        G.edges[cat_name, asp_name]["weight"] += topic.doc_count
        G.edges[product, cat_name]["weight"]  += topic.doc_count

    # ---- Unmapped topics -> "Other" category ----
    if unmapped_topics:
        other_cat = "Other"
        if other_cat not in G:
            G.add_node(other_cat, type="category", label=other_cat, topic_ids=[], doc_count=0, top_words=[])
            G.add_edge(product, other_cat, weight=0)

        for topic in unmapped_topics:
            topic_node = f"topic_{topic.id}"
            G.add_node(
                topic_node,
                type="topic",
                label=topic.label,
                topic_ids=[topic.id],
                doc_count=topic.doc_count,
                top_words=topic.top_words,
            )
            G.add_edge(other_cat, topic_node, weight=topic.doc_count)
            G.nodes[other_cat]["doc_count"] += topic.doc_count
            G.nodes[product]["doc_count"]   += topic.doc_count
            G.edges[product, other_cat]["weight"] += topic.doc_count

        print_info(f"{len(unmapped_topics)} topics placed under 'Other' (below similarity threshold).")

    # ---- Prune empty taxonomy leaves ----
    empty_leaves = [
        n for n in list(G.nodes)
        if G.nodes[n]["type"] == "aspect" and G.nodes[n]["doc_count"] == 0
    ]
    G.remove_nodes_from(empty_leaves)

    mapped_count = len(result.topics) - len(unmapped_topics)
    print_success(
        f"Aspect graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges  "
        f"| {mapped_count}/{len(result.topics)} topics mapped to taxonomy"
    )
    return G


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def graph_to_dict(G: nx.DiGraph) -> dict[str, Any]:
    """JSON-serialisable representation of the DiGraph."""
    return {
        "nodes": [
            {"id": n, **G.nodes[n]}
            for n in G.nodes
        ],
        "edges": [
            {"source": u, "target": v, **G.edges[u, v]}
            for u, v in G.edges
        ],
    }


def save_graph(G: nx.DiGraph, out_dir: Path, filename: str = "aspect_graph.json") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(
        json.dumps(graph_to_dict(G), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_success(f"Aspect graph saved -> {path.relative_to(path.parent.parent.parent)}")
    return path


def load_graph(path: Path) -> nx.DiGraph:
    data = json.loads(path.read_text(encoding="utf-8"))
    G = nx.DiGraph()
    for node in data["nodes"]:
        nid = node.pop("id")
        G.add_node(nid, **node)
    for edge in data["edges"]:
        src = edge.pop("source")
        tgt = edge.pop("target")
        G.add_edge(src, tgt, **edge)
    return G


# ---------------------------------------------------------------------------
# Pretty-print the tree
# ---------------------------------------------------------------------------

def print_tree(G: nx.DiGraph, product: str) -> None:
    """Render the aspect hierarchy as an indented tree using Rich."""
    from rich.tree import Tree
    from absa.utils.display import console

    def _add_children(tree_node: Tree, graph_node: str, depth: int = 0) -> None:
        if depth > 5:
            return
        for child in sorted(G.successors(graph_node),
                             key=lambda c: G.nodes[c]["doc_count"], reverse=True):
            attrs = G.nodes[child]
            ntype = attrs["type"]
            count = attrs["doc_count"]
            label = attrs["label"]

            if ntype == "category":
                branch = tree_node.add(f"[bold cyan]{label}[/bold cyan]  [dim]({count} mentions)[/dim]")
            elif ntype == "aspect":
                words = ", ".join(attrs.get("top_words", [])[:4])
                branch = tree_node.add(
                    f"[green]{label}[/green]  [dim]({count} mentions)[/dim]"
                    + (f"  [dim italic]{words}[/dim italic]" if words else "")
                )
            else:  # topic leaf
                words = ", ".join(attrs.get("top_words", [])[:3])
                branch = tree_node.add(f"[yellow]{label}[/yellow]  [dim]({count})[/dim]  [dim italic]{words}[/dim italic]")

            _add_children(branch, child, depth + 1)

    root_label = (
        f"[bold magenta]{product}[/bold magenta]  "
        f"[dim]({G.nodes[product]['doc_count']} total mentions)[/dim]"
    )
    tree = Tree(root_label)
    _add_children(tree, product)
    console.print(tree)
