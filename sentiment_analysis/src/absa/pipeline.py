"""
Pipeline orchestrator.

Wires together the individual stages:
  fetch -> preprocess -> topic_model -> (absa -> aggregate -> report)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from absa.data.collector import fetch as _fetch
from absa.data.preprocessor import preprocess as _preprocess
from absa.models.topic_model import TopicModelResult, run_topic_model
from absa.models.aspect_mapper import build_aspect_graph, save_graph
from absa.utils.config import settings
from absa.utils.display import print_header, print_info


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    product: str
    slug: str
    raw_paths: dict[str, Path] = field(default_factory=dict)
    processed_path: Path | None = None
    sentences: list[dict[str, Any]] = field(default_factory=list)
    topic_result: TopicModelResult | None = None
    aspect_graph: nx.DiGraph | None = None

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Stateless pipeline runner.  Each run() returns a PipelineResult.
    """

    def __init__(
        self,
        subreddits: list[str] | None = None,
        post_limit: int | None = None,
        comment_limit: int | None = None,
        time_filter: str | None = None,
        force: bool = False,
    ) -> None:
        self.subreddits    = subreddits
        self.post_limit    = post_limit
        self.comment_limit = comment_limit
        self.time_filter   = time_filter
        self.force         = force

    # ---- helpers ----

    @staticmethod
    def _slugify(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        return re.sub(r"[\s_-]+", "-", text)

    # ---- Stage 1: Fetch ----

    def fetch(self, product: str) -> dict[str, Path]:
        slug = self._slugify(product)
        print_header(
            f"Stage 1 — Data Collection",
            f"Product: {product}  |  slug: {slug}",
        )
        raw_paths = _fetch(
            product=product,
            subreddits=self.subreddits,
            post_limit=self.post_limit,
            comment_limit=self.comment_limit,
            time_filter=self.time_filter,
            out_dir=settings.raw_dir / slug,
            force=self.force,
        )
        return raw_paths

    # ---- Stage 2: Preprocess ----

    def preprocess(self, product: str, raw_paths: dict[str, Path]) -> Path:
        slug = self._slugify(product)
        print_header(
            "Stage 2 — Preprocessing",
            f"Cleaning & sentence-splitting {len(raw_paths)} subreddit file(s)…",
        )
        processed_path = _preprocess(
            raw_paths=list(raw_paths.values()),
            out_dir=settings.processed_dir / slug,
            slug=slug,
            force=self.force,
        )
        return processed_path

    # ---- Stage 3: Topic modeling + aspect graph ----

    def topic_model(
        self, product: str, sentences: list[dict[str, Any]]
    ) -> tuple[TopicModelResult, nx.DiGraph]:
        slug = self._slugify(product)
        print_header(
            "Stage 3 — Topic Modeling",
            f"BERTopic + hierarchical aspect graph  |  {len(sentences)} sentences",
        )
        out_dir = settings.results_dir / slug / "topics"
        topic_result = run_topic_model(sentences, out_dir=out_dir, force=self.force)

        graph = build_aspect_graph(topic_result, product=product)
        save_graph(graph, out_dir=out_dir)
        return topic_result, graph

    # ---- Future stages (stubs) ----

    def absa(self, processed_path: Path, topics: Any) -> Any:
        raise NotImplementedError("ABSA stage not yet implemented.")

    def aggregate(self, absa_results: Any, topics: Any) -> Any:
        raise NotImplementedError("Aggregation stage not yet implemented.")

    def report(self, aggregated: Any) -> None:
        raise NotImplementedError("Report stage not yet implemented.")

    # ---- Full run ----

    def run(self, product: str) -> PipelineResult:
        import json

        slug = self._slugify(product)
        result = PipelineResult(product=product, slug=slug)

        # Stage 1
        result.raw_paths = self.fetch(product)
        if not result.raw_paths:
            print_info("No data fetched — aborting pipeline.")
            return result

        # Stage 2
        result.processed_path = self.preprocess(product, result.raw_paths)
        if result.processed_path and result.processed_path.exists():
            result.sentences = json.loads(
                result.processed_path.read_text(encoding="utf-8")
            )

        if not result.sentences:
            print_info("No sentences after preprocessing — aborting pipeline.")
            return result

        # Stage 3
        result.topic_result, result.aspect_graph = self.topic_model(
            product, result.sentences
        )

        print_info(
            f"Stages 1-3 complete for '{product}': "
            f"{result.sentence_count} sentences, "
            f"{len(result.topic_result.topics)} topics, "
            f"{result.aspect_graph.number_of_nodes()} graph nodes."
        )
        return result
