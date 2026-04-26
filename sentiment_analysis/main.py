"""
HierABSA — interactive single-file entry point.

Usage:
    uv run python main.py
    uv run python main.py "Samsung Galaxy S25"
    uv run python main.py "iPhone 16 Pro" --paradigms transformer,lexicon --force
"""
from __future__ import annotations

import io
import sys
import argparse

# ── Force UTF-8 on Windows (cp1252 chokes on Rich box-drawing chars) ──────────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.rule import Rule

console = Console()

# ── Banner ─────────────────────────────────────────────────────────────────────
_BANNER = """[bold magenta]HierABSA[/bold magenta]  ·  Hierarchical Aspect-Based Sentiment Analysis

  Discovers a [cyan]Product → Category → Aspect[/cyan] hierarchy from Reddit,
  then runs three ABSA paradigms [dim](transformer · LLM · lexicon)[/dim],
  aggregates sentiment with upvote weighting, and saves results to [green]outputs/[/green]."""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HierABSA — full pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("product", nargs="?", default=None,
                   help="Product name (prompted if omitted)")
    p.add_argument("--paradigms", default=None,
                   help="Comma-separated: transformer,llm,lexicon (prompted if omitted)")
    p.add_argument("--force", action="store_true",
                   help="Re-run all stages even if results are cached")
    p.add_argument("--subreddits", default=None,
                   help="Comma-separated subreddit names (uses defaults if omitted)")
    return p.parse_args()


def _prompt_inputs(args: argparse.Namespace) -> tuple[str, list[str], bool]:
    """Interactively gather product name, paradigms, and force flag."""
    console.print(Panel(_BANNER, border_style="magenta", padding=(1, 2)))
    console.print()

    product = args.product or Prompt.ask(
        "[bold]Product name[/bold]",
        default="Samsung Galaxy S25",
    )

    default_paradigms = "transformer,llm,lexicon"
    if args.paradigms:
        paradigms_str = args.paradigms
    else:
        paradigms_str = Prompt.ask(
            "[bold]Paradigms[/bold]  [dim](transformer / llm / lexicon, comma-separated)[/dim]",
            default=default_paradigms,
        )

    paradigm_list = [p.strip() for p in paradigms_str.split(",") if p.strip()]

    force = args.force or Confirm.ask(
        "[bold]Force re-run?[/bold]  [dim](ignore cached results)[/dim]",
        default=False,
    )

    console.print()
    console.print(Rule(style="dim"))
    console.print(
        f"  [dim]Product :[/dim]  [cyan]{product}[/cyan]\n"
        f"  [dim]Paradigms:[/dim] [cyan]{', '.join(paradigm_list)}[/cyan]\n"
        f"  [dim]Force    :[/dim] [cyan]{force}[/cyan]"
    )
    console.print(Rule(style="dim"))
    console.print()

    return product, paradigm_list, force


def main() -> None:
    args = _parse_args()
    product, paradigm_list, force = _prompt_inputs(args)

    # ── Lazy imports (keep startup fast) ──────────────────────────────────────
    from absa.pipeline import Pipeline
    from absa.models.aspect_mapper import print_tree
    from absa.analysis.aggregator import (
        print_product_summary, print_aspect_table, print_final_scorecard,
        save_aggregation,
    )
    from absa.evaluation.comparator import print_evaluation_report
    from absa.reporting.results_report import run_report
    from absa.reporting.visualizer import save_visualizations
    from absa.utils.display import print_success, print_error, print_header
    from absa.utils.config import settings

    subs = (
        [s.strip() for s in args.subreddits.split(",")]
        if args.subreddits else None
    )

    pipeline = Pipeline(
        subreddits=subs,
        force=force,
    )

    # ── Stage 1–6: full pipeline ───────────────────────────────────────────────
    try:
        result = pipeline.run(product)
    except EnvironmentError as exc:
        print_error(str(exc))
        sys.exit(1)

    if not result.sentences:
        print_error("Pipeline produced no sentences — check your product name and API credentials.")
        sys.exit(1)

    # ── Display: hierarchy tree ────────────────────────────────────────────────
    if result.aspect_graph:
        print_header("Aspect Hierarchy", product)
        print_tree(result.aspect_graph, product)

    # ── Display: aggregated sentiment ─────────────────────────────────────────
    if result.aggregated_scores:
        print_product_summary(result.aggregated_scores)
        print_aspect_table(result.aggregated_scores, weighting="weighted")

    # ── Display: evaluation metrics ───────────────────────────────────────────
    if result.evaluation:
        print_evaluation_report(result.evaluation)

    # ── Display: final 0-100 scorecard (user-facing summary) ──────────────────
    if result.aggregated_scores:
        console.print()
        print_header("Final Scorecard", f"{product} — sentiment rating per aspect")
        print_final_scorecard(result.aggregated_scores)

    # ── Save full report to outputs/reports/<slug>/ ────────────────────────────
    import json, re
    slug = re.sub(r"[^\w\s-]", "", product.lower().strip())
    slug = re.sub(r"[\s_-]+", "-", slug)

    run_report(product, save=True)

    # ── Visualizations ────────────────────────────────────────────────────────
    if result.aspect_graph and result.aggregated_scores:
        print_header("Visualizations", "Generating PNG outputs…")

        # Convert ProductScore dataclasses → raw dict for visualizer
        from dataclasses import asdict
        agg_raw: dict = {}
        for paradigm, ps in result.aggregated_scores.items():
            agg_raw[paradigm] = asdict(ps)

        save_visualizations(product, result.aspect_graph, agg_raw)

    # ── Final summary ─────────────────────────────────────────────────────────
    console.print()
    console.print(Rule(style="green"))
    console.print(
        f"\n[bold green]Done![/bold green]  "
        f"{result.sentence_count} sentences · "
        f"{len(result.topic_result.topics if result.topic_result else [])} topics · "
        f"{len(result.absa_results)} paradigms\n"
    )
    console.print(f"  [dim]Reports  :[/dim] [green]outputs/reports/{slug}/[/green]")
    console.print(f"  [dim]Visuals  :[/dim] [green]outputs/visualizations/{slug}/[/green]")
    console.print()


if __name__ == "__main__":
    main()
