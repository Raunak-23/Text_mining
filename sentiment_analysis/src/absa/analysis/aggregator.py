"""
Multi-level sentiment aggregation with upvote weighting.

Aggregates SentenceABSAResult records up the aspect hierarchy:
  sentence -> aspect -> category -> product

Two weighting schemes (RQ3):
  uniform  : every sentence counts equally (weight = confidence)
  weighted : weight = log1p(comment_score) * confidence

Outputs
-------
  AspectSentimentScore  per aspect per paradigm
  CategoryScore         per category (aggregated from aspects)
  ProductScore          overall product sentiment

The comparison between uniform and weighted (RQ3 analysis) is exposed
via compare_weighting_schemes().

Saved under data/results/<slug>/absa/aggregated.json
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from absa.models.absa_model import SentenceABSAResult
from absa.utils.display import print_info, print_success


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AspectSentimentScore:
    aspect: str
    paradigm: str
    n_mentions: int
    positive: float         # weighted fraction
    negative: float
    neutral: float
    weighted_score: float   # net score in [-1, 1]: positive - negative
    dominant: str           # majority sentiment label
    weighting: str          # "uniform" | "weighted"


@dataclass
class CategoryScore:
    category: str
    paradigm: str
    n_mentions: int
    positive: float
    negative: float
    neutral: float
    weighted_score: float
    dominant: str
    weighting: str
    aspects: list[AspectSentimentScore] = field(default_factory=list)


@dataclass
class ProductScore:
    product: str
    paradigm: str
    n_mentions: int
    positive: float
    negative: float
    neutral: float
    weighted_score: float
    dominant: str
    weighting: str
    categories: list[CategoryScore] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _safe_score(value: float) -> float:
    return round(float(value), 4)


def _dominant(pos: float, neg: float, neu: float) -> str:
    m = max(pos, neg, neu)
    if m == pos:
        return "positive"
    if m == neg:
        return "negative"
    return "neutral"


def _aggregate_opinions(
    aspect_name: str,
    paradigm: str,
    opinions: list[tuple[float, str]],  # (weight, sentiment)
    weighting: str,
) -> AspectSentimentScore:
    """
    Given a list of (weight, sentiment) tuples, compute distribution + net score.
    """
    total_w = sum(w for w, _ in opinions)
    if total_w == 0.0:
        total_w = 1.0

    pos = sum(w for w, s in opinions if s == "positive") / total_w
    neg = sum(w for w, s in opinions if s == "negative") / total_w
    neu = sum(w for w, s in opinions if s == "neutral")  / total_w

    return AspectSentimentScore(
        aspect=aspect_name,
        paradigm=paradigm,
        n_mentions=len(opinions),
        positive=_safe_score(pos),
        negative=_safe_score(neg),
        neutral=_safe_score(neu),
        weighted_score=_safe_score(pos - neg),
        dominant=_dominant(pos, neg, neu),
        weighting=weighting,
    )


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def aggregate(
    results: dict[str, list[SentenceABSAResult]],
    aspect_graph: nx.DiGraph,
    product: str,
    weighting: str = "weighted",
) -> dict[str, ProductScore]:
    """
    Aggregate per-sentence ABSA results up the aspect hierarchy.

    Parameters
    ----------
    results       : output of run_absa() — paradigm -> sentences
    aspect_graph  : NetworkX DiGraph (Product->Category->Aspect->Topic)
    product       : root node label in the graph
    weighting     : "weighted" (log1p upvote) or "uniform"

    Returns
    -------
    dict mapping paradigm -> ProductScore
    """
    # Build aspect->category lookup from graph
    asp_to_cat: dict[str, str] = {}
    for node, attrs in aspect_graph.nodes(data=True):
        if attrs["type"] == "aspect":
            for pred in aspect_graph.predecessors(node):
                if aspect_graph.nodes[pred]["type"] == "category":
                    asp_to_cat[attrs["label"]] = aspect_graph.nodes[pred]["label"]

    scores: dict[str, ProductScore] = {}

    for paradigm, sentence_results in results.items():
        if not sentence_results:
            continue

        # Collect opinions per aspect: {aspect_name: [(weight, sentiment)]}
        asp_opinions: dict[str, list[tuple[float, str]]] = {}

        for sr in sentence_results:
            # Compute sentence-level weight
            if weighting == "weighted":
                w_base = math.log1p(max(sr.comment_score, 0))
                if w_base == 0.0:
                    w_base = 0.1   # floor weight so all sentences count
            else:
                w_base = 1.0

            for op in sr.aspects:
                w = w_base * op.confidence
                asp_name = op.aspect
                asp_opinions.setdefault(asp_name, []).append((w, op.sentiment))

        if not asp_opinions:
            continue

        # Aspect-level scores
        asp_scores: dict[str, AspectSentimentScore] = {}
        for asp_name, opinions in asp_opinions.items():
            asp_scores[asp_name] = _aggregate_opinions(asp_name, paradigm, opinions, weighting)

        # Category-level scores
        cat_opinions: dict[str, list[tuple[float, str]]] = {}
        for asp_name, opinions in asp_opinions.items():
            cat = asp_to_cat.get(asp_name, "Other")
            cat_opinions.setdefault(cat, []).extend(opinions)

        cat_scores: dict[str, CategoryScore] = {}
        for cat_name, opinions in cat_opinions.items():
            aspect_list = [s for a, s in asp_scores.items() if asp_to_cat.get(a, "Other") == cat_name]
            total_w = sum(w for w, _ in opinions) or 1.0
            pos = sum(w for w, s in opinions if s == "positive") / total_w
            neg = sum(w for w, s in opinions if s == "negative") / total_w
            neu = sum(w for w, s in opinions if s == "neutral")  / total_w
            cat_scores[cat_name] = CategoryScore(
                category=cat_name,
                paradigm=paradigm,
                n_mentions=len(opinions),
                positive=_safe_score(pos),
                negative=_safe_score(neg),
                neutral=_safe_score(neu),
                weighted_score=_safe_score(pos - neg),
                dominant=_dominant(pos, neg, neu),
                weighting=weighting,
                aspects=[s for s in asp_scores.values() if asp_to_cat.get(s.aspect, "Other") == cat_name],
            )

        # Product-level score
        all_opinions: list[tuple[float, str]] = []
        for opinions in asp_opinions.values():
            all_opinions.extend(opinions)
        total_w = sum(w for w, _ in all_opinions) or 1.0
        pos = sum(w for w, s in all_opinions if s == "positive") / total_w
        neg = sum(w for w, s in all_opinions if s == "negative") / total_w
        neu = sum(w for w, s in all_opinions if s == "neutral")  / total_w

        scores[paradigm] = ProductScore(
            product=product,
            paradigm=paradigm,
            n_mentions=len(all_opinions),
            positive=_safe_score(pos),
            negative=_safe_score(neg),
            neutral=_safe_score(neu),
            weighted_score=_safe_score(pos - neg),
            dominant=_dominant(pos, neg, neu),
            weighting=weighting,
            categories=list(cat_scores.values()),
        )

    return scores


# ---------------------------------------------------------------------------
# RQ3: Compare weighting schemes
# ---------------------------------------------------------------------------

@dataclass
class WeightingComparison:
    paradigm: str
    aspect: str
    uniform_score: float
    weighted_score: float
    delta: float            # weighted - uniform


def compare_weighting_schemes(
    results: dict[str, list[SentenceABSAResult]],
    aspect_graph: nx.DiGraph,
    product: str,
) -> list[WeightingComparison]:
    """
    Compute per-aspect net sentiment under both weighting schemes and
    return the delta list (used to answer RQ3).
    """
    uniform_scores  = aggregate(results, aspect_graph, product, weighting="uniform")
    weighted_scores = aggregate(results, aspect_graph, product, weighting="weighted")

    comparisons: list[WeightingComparison] = []
    for paradigm in uniform_scores:
        if paradigm not in weighted_scores:
            continue
        u_cats = {c.category: c for c in uniform_scores[paradigm].categories}
        w_cats = {c.category: c for c in weighted_scores[paradigm].categories}

        for cat_name in u_cats:
            if cat_name not in w_cats:
                continue
            u_asps = {a.aspect: a for a in u_cats[cat_name].aspects}
            w_asps = {a.aspect: a for a in w_cats[cat_name].aspects}
            for asp_name in u_asps:
                if asp_name not in w_asps:
                    continue
                comparisons.append(WeightingComparison(
                    paradigm=paradigm,
                    aspect=asp_name,
                    uniform_score=u_asps[asp_name].weighted_score,
                    weighted_score=w_asps[asp_name].weighted_score,
                    delta=w_asps[asp_name].weighted_score - u_asps[asp_name].weighted_score,
                ))

    return sorted(comparisons, key=lambda c: abs(c.delta), reverse=True)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_aggregation(
    scores: dict[str, ProductScore],
    out_dir: Path,
    filename: str = "aggregated.json",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(
        json.dumps({p: asdict(s) for p, s in scores.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_success(f"Aggregated scores saved -> {path.name}")
    return path


def load_aggregation(path: Path) -> dict[str, ProductScore]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, ProductScore] = {}
    for paradigm, data in raw.items():
        cats = []
        for c in data.pop("categories", []):
            asps = [AspectSentimentScore(**a) for a in c.pop("aspects", [])]
            cats.append(CategoryScore(aspects=asps, **c))
        out[paradigm] = ProductScore(categories=cats, **data)
    return out


# ---------------------------------------------------------------------------
# Rich visualisations
# ---------------------------------------------------------------------------

def print_aspect_table(
    scores: dict[str, ProductScore],
    weighting: str = "weighted",
) -> None:
    """Print a Rich table comparing aspect sentiment across paradigms."""
    from rich.table import Table
    from absa.utils.display import console

    paradigms = list(scores.keys())
    if not paradigms:
        return

    # Collect all aspects
    asp_names: list[str] = []
    for ps in scores.values():
        for cat in ps.categories:
            for asp in cat.aspects:
                if asp.aspect not in asp_names:
                    asp_names.append(asp.aspect)

    # Build lookup
    lookup: dict[tuple[str, str], AspectSentimentScore] = {}
    for paradigm, ps in scores.items():
        for cat in ps.categories:
            for asp in cat.aspects:
                lookup[(paradigm, asp.aspect)] = asp

    tbl = Table(title=f"Aspect Sentiment Scores ({weighting} weighting)", show_lines=True)
    tbl.add_column("Aspect", style="green")
    for p in paradigms:
        tbl.add_column(f"{p[:12]}\n[dim]pos/neg/neu[/dim]", justify="center")

    SENT_COLOR = {"positive": "green", "negative": "red", "neutral": "yellow"}

    for asp in sorted(asp_names):
        row = [asp]
        for p in paradigms:
            sc = lookup.get((p, asp))
            if sc is None:
                row.append("[dim]-[/dim]")
            else:
                col = SENT_COLOR.get(sc.dominant, "white")
                row.append(
                    f"[{col}]{sc.positive:.0%}[/{col}] / "
                    f"[red]{sc.negative:.0%}[/red] / "
                    f"[dim]{sc.neutral:.0%}[/dim]"
                )
        tbl.add_row(*row)

    console.print(tbl)


def print_final_scorecard(scores: dict[str, ProductScore]) -> None:
    """
    User-facing final scorecard: each aspect rated 0-100 (mean across paradigms),
    with a verdict band, ASCII bar, and overall product score.

    Mapping:  net_score in [-1, 1]  ->  score_100 = round((net + 1) * 50)
              -1 -> 0     0 -> 50     +1 -> 100
    """
    from rich.table import Table
    from absa.utils.display import console

    if not scores:
        return

    # Per-aspect mean net score across paradigms (+ total mention count)
    aspect_nets: dict[str, list[float]] = {}
    aspect_mentions: dict[str, int] = {}
    for ps in scores.values():
        for cat in ps.categories:
            for asp in cat.aspects:
                aspect_nets.setdefault(asp.aspect, []).append(asp.weighted_score)
                aspect_mentions[asp.aspect] = aspect_mentions.get(asp.aspect, 0) + asp.n_mentions

    if not aspect_nets:
        return

    def _band(s100: int) -> tuple[str, str]:
        if s100 >= 70: return "Excellent", "bright_green"
        if s100 >= 60: return "Good",      "green"
        if s100 >= 50: return "Mixed",     "yellow"
        if s100 >= 40: return "Poor",      "dark_orange"
        return "Critical", "red"

    def _bar(s100: int, width: int = 22) -> str:
        filled = round(width * max(0, min(100, s100)) / 100)
        return "█" * filled + "░" * (width - filled)

    rows: list[tuple[str, int, int]] = []
    for aspect, nets in aspect_nets.items():
        mean_net = sum(nets) / len(nets)
        s100 = round((mean_net + 1) * 50)
        rows.append((aspect, s100, aspect_mentions[aspect]))
    rows.sort(key=lambda r: -r[1])

    tbl = Table(
        title="[bold]Final Aspect Scorecard[/bold]  (0-100, mean across paradigms)",
        show_lines=False,
        title_justify="left",
    )
    tbl.add_column("Aspect",   style="cyan", no_wrap=True)
    tbl.add_column("Score",    justify="right")
    tbl.add_column("Bar",      justify="left")
    tbl.add_column("Verdict",  justify="center")
    tbl.add_column("Mentions", justify="right", style="dim")

    for aspect, s100, n in rows:
        label, color = _band(s100)
        tbl.add_row(
            aspect,
            f"[{color}]{s100:>3}/100[/{color}]",
            f"[{color}]{_bar(s100)}[/{color}]",
            f"[{color}]{label}[/{color}]",
            str(n),
        )

    # Overall product-level score (mean of per-paradigm net scores)
    prod_nets = [ps.weighted_score for ps in scores.values()]
    prod_100 = round((sum(prod_nets) / len(prod_nets) + 1) * 50)
    label, color = _band(prod_100)
    try:
        tbl.add_section()
    except Exception:
        pass
    tbl.add_row(
        "[bold]OVERALL[/bold]",
        f"[bold {color}]{prod_100:>3}/100[/bold {color}]",
        f"[bold {color}]{_bar(prod_100)}[/bold {color}]",
        f"[bold {color}]{label}[/bold {color}]",
        f"[bold]{sum(aspect_mentions.values())}[/bold]",
    )

    console.print(tbl)


def print_product_summary(scores: dict[str, ProductScore]) -> None:
    """Print a one-line summary per paradigm."""
    from rich.table import Table
    from absa.utils.display import console

    tbl = Table(title="Product-Level Sentiment Summary", show_lines=False)
    tbl.add_column("Paradigm", style="cyan")
    tbl.add_column("Positive", style="green",  justify="right")
    tbl.add_column("Negative", style="red",    justify="right")
    tbl.add_column("Neutral",  style="yellow", justify="right")
    tbl.add_column("Net Score", justify="right")
    tbl.add_column("Dominant", justify="center")
    tbl.add_column("Weighting", style="dim")

    SENT_COLOR = {"positive": "green", "negative": "red", "neutral": "yellow"}

    for paradigm, ps in scores.items():
        col = SENT_COLOR.get(ps.dominant, "white")
        tbl.add_row(
            paradigm,
            f"{ps.positive:.1%}",
            f"{ps.negative:.1%}",
            f"{ps.neutral:.1%}",
            f"[{col}]{ps.weighted_score:+.3f}[/{col}]",
            f"[{col}]{ps.dominant}[/{col}]",
            ps.weighting,
        )
    console.print(tbl)
