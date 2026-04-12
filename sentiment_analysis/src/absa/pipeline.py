"""
Pipeline orchestrator.

Wires together the individual stages:
  fetch → preprocess → (topic_model → absa → aggregate → report)

Only fetch + preprocess are implemented in this milestone.
Later stages are stubbed with NotImplementedError so the CLI can
call them and get a clear message rather than an AttributeError.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from absa.data.collector import fetch as _fetch
from absa.data.preprocessor import preprocess as _preprocess
from absa.utils.config import settings
from absa.utils.display import print_header, print_info


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    product: str
    slug: str
    raw_paths: dict[str, Path] = field(default_factory=dict)      # subreddit → file
    processed_path: Path | None = None
    sentences: list[dict[str, Any]] = field(default_factory=list)
    # Future stages will add more fields (topics, absa_results, report)

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

    # ---- Future stages (stubs) ----

    def topic_model(self, processed_path: Path) -> Any:
        raise NotImplementedError("Topic modeling stage not yet implemented.")

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

        print_info(
            f"Pipeline complete for '{product}': "
            f"{result.sentence_count} sentences ready for modeling."
        )
        return result
