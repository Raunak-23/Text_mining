"""
Pipeline orchestrator.

Wires together the individual stages:
  fetch -> preprocess -> topic_model -> absa -> aggregate -> evaluate -> report
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
from absa.models.absa_model import SentenceABSAResult, run_absa
from absa.analysis.aggregator import (
    ProductScore,
    aggregate,
    compare_weighting_schemes,
    save_aggregation,
)
from absa.evaluation.comparator import EvaluationReport, run_full_evaluation
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
    absa_results: dict[str, list[SentenceABSAResult]] = field(default_factory=dict)
    aggregated_scores: dict[str, ProductScore] = field(default_factory=dict)
    evaluation: EvaluationReport | None = None

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

    # ---- Stage 4: ABSA ----

    def run_absa(
        self,
        product: str,
        sentences: list[dict[str, Any]],
        aspect_graph: nx.DiGraph,
        paradigms: list[str] | None = None,
        llm_sample: int = 300,
    ) -> dict[str, list[SentenceABSAResult]]:
        slug = self._slugify(product)
        print_header(
            "Stage 4 — ABSA",
            f"Multi-paradigm sentiment analysis  |  {len(sentences)} sentences",
        )
        out_dir = settings.results_dir / slug / "absa"
        return run_absa(
            sentences=sentences,
            aspect_graph=aspect_graph,
            out_dir=out_dir,
            force=self.force,
            paradigms=paradigms,
            llm_sample=llm_sample,
        )

    # ---- Stage 5: Aggregation ----

    def aggregate(
        self,
        product: str,
        absa_results: dict[str, list[SentenceABSAResult]],
        aspect_graph: nx.DiGraph,
        weighting: str = "weighted",
    ) -> dict[str, ProductScore]:
        slug = self._slugify(product)
        print_header(
            "Stage 5 — Aggregation",
            f"Weighting: {weighting}",
        )
        out_dir = settings.results_dir / slug / "absa"
        scores = aggregate(absa_results, aspect_graph, product, weighting=weighting)
        save_aggregation(scores, out_dir=out_dir)
        return scores

    # ---- Stage 6: Evaluation ----

    def evaluate(
        self,
        product: str,
        absa_results: dict[str, list[SentenceABSAResult]],
        aggregated_scores: dict[str, ProductScore],
        topic_result: TopicModelResult,
        sentences: list[dict[str, Any]],
        aspect_graph: nx.DiGraph,
    ) -> EvaluationReport:
        slug = self._slugify(product)
        print_header("Stage 6 — Evaluation", "Cohen's kappa, JSD, H(A), D(A), C_v")
        out_dir = settings.results_dir / slug / "absa"
        weighting_comparisons = compare_weighting_schemes(absa_results, aspect_graph, product)
        return run_full_evaluation(
            absa_results=absa_results,
            aggregated_scores=aggregated_scores,
            topic_result=topic_result,
            sentences=sentences,
            product=product,
            out_dir=out_dir,
            weighting_comparisons=weighting_comparisons,
        )

    def report(self, aggregated: Any) -> None:
        raise NotImplementedError("HTML report stage not yet implemented.")

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

        # Stage 4 — ABSA
        result.absa_results = self.run_absa(
            product, result.sentences, result.aspect_graph
        )

        # Stage 5 — Aggregation (both weighting schemes)
        result.aggregated_scores = self.aggregate(
            product, result.absa_results, result.aspect_graph, weighting="weighted"
        )

        # Stage 6 — Evaluation
        if result.absa_results and result.topic_result:
            result.evaluation = self.evaluate(
                product,
                result.absa_results,
                result.aggregated_scores,
                result.topic_result,
                result.sentences,
                result.aspect_graph,
            )

        print_info(
            f"Pipeline complete for '{product}': "
            f"{result.sentence_count} sentences, "
            f"{len(result.topic_result.topics)} topics, "
            f"{result.aspect_graph.number_of_nodes()} graph nodes, "
            f"{len(result.absa_results)} ABSA paradigms."
        )
        return result
