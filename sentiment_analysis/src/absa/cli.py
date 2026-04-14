"""
Typer CLI entry point.

Usage (after `uv sync`):
    uv run absa fetch "iPhone 15 Pro"
    uv run absa preprocess "iPhone 15 Pro"
    uv run absa analyze "iPhone 15 Pro"
"""
from __future__ import annotations

import io
import json
import sys
from typing import Annotated

# Force UTF-8 output on Windows (cp1252 can't encode Rich box-drawing chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import typer
from rich.table import Table

from absa.pipeline import Pipeline
from absa.utils.config import settings
from absa.utils.display import console, print_error, print_header, print_success

app = typer.Typer(
    name="absa",
    help="Hierarchical ABSA on Reddit product discourse.",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

# ---------------------------------------------------------------------------
# Shared option types
# ---------------------------------------------------------------------------

ProductArg   = Annotated[str,           typer.Argument(help="Product name, e.g. 'iPhone 15 Pro'")]
SubredditsOpt = Annotated[str | None,   typer.Option("--subreddits", "-s",
                                         help="Comma-separated subreddit names, e.g. apple,iphone")]
LimitOpt     = Annotated[int | None,    typer.Option("--limit", "-l", help="Max posts per subreddit")]
TimeOpt      = Annotated[str,           typer.Option("--time-filter", "-t",
                                         help="Reddit time filter: all|year|month|week|day")]
ForceOpt     = Annotated[bool,          typer.Option("--force", "-f",
                                         help="Re-fetch/re-process even if cached")]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def fetch(
    product: ProductArg,
    subreddits: SubredditsOpt = None,
    limit: LimitOpt = None,
    time_filter: TimeOpt = "year",
    force: ForceOpt = False,
) -> None:
    """Fetch Reddit posts and comments for a product and cache them as JSON."""
    subs = [s.strip() for s in subreddits.split(",")] if subreddits else None
    pipeline = Pipeline(
        subreddits=subs,
        post_limit=limit,
        time_filter=time_filter,
        force=force,
    )
    try:
        raw_paths = pipeline.fetch(product)
    except EnvironmentError as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    if not raw_paths:
        print_error("No data fetched. Try a different product name or subreddits.")
        raise typer.Exit(1)

    table = Table(title="Fetched files", show_lines=True)
    table.add_column("Subreddit", style="cyan")
    table.add_column("Posts", style="green", justify="right")
    table.add_column("Path", style="dim")
    for sub, path in raw_paths.items():
        posts = len(json.loads(path.read_text(encoding="utf-8"))["posts"])
        table.add_row(f"r/{sub}", str(posts), str(path.relative_to(settings.root)))
    console.print(table)


@app.command()
def preprocess(
    product: ProductArg,
    force: ForceOpt = False,
) -> None:
    """Clean and sentence-split cached Reddit data for a product."""
    from absa.data.collector import _slugify
    slug = _slugify(product)
    raw_dir = settings.raw_dir / slug

    if not raw_dir.exists():
        print_error(f"No raw data found for '{product}'. Run `absa fetch \"{product}\"` first.")
        raise typer.Exit(1)

    raw_paths = {p.stem: p for p in raw_dir.glob("*.json")}
    if not raw_paths:
        print_error(f"Raw directory exists but is empty: {raw_dir}")
        raise typer.Exit(1)

    pipeline = Pipeline(force=force)
    try:
        processed_path = pipeline.preprocess(product, raw_paths)
    except OSError as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    sentences = json.loads(processed_path.read_text(encoding="utf-8"))
    print_success(f"{len(sentences)} sentences → {processed_path.relative_to(settings.root)}")

    # Quick breakdown by source type
    from collections import Counter
    counts = Counter(s["source"] for s in sentences)
    table = Table(title="Sentence breakdown", show_lines=False)
    table.add_column("Source", style="cyan")
    table.add_column("Count",  style="green", justify="right")
    for src, n in sorted(counts.items()):
        table.add_row(src, str(n))
    console.print(table)


@app.command()
def topics(
    product: ProductArg,
    force: ForceOpt = False,
) -> None:
    """Run BERTopic on preprocessed data and print the aspect hierarchy tree."""
    from absa.data.collector import _slugify
    from absa.models.topic_model import run_topic_model
    from absa.models.aspect_mapper import build_aspect_graph, save_graph, load_graph, print_tree

    slug = _slugify(product)
    sentences_file = settings.processed_dir / slug / "sentences.json"
    graph_file     = settings.results_dir / slug / "topics" / "aspect_graph.json"

    if not sentences_file.exists():
        print_error(
            f"No preprocessed data for '{product}'. "
            f"Run `absa preprocess \"{product}\"` first."
        )
        raise typer.Exit(1)

    # Load or rebuild graph
    if graph_file.exists() and not force:
        print_success(f"Loading cached aspect graph from {graph_file.relative_to(settings.root)}")
        graph = load_graph(graph_file)
    else:
        sentences = json.loads(sentences_file.read_text(encoding="utf-8"))
        out_dir = settings.results_dir / slug / "topics"
        topic_result = run_topic_model(sentences, out_dir=out_dir, force=force)
        graph = build_aspect_graph(topic_result, product=product)
        save_graph(graph, out_dir=out_dir)

    print_header("Aspect Hierarchy", product)
    print_tree(graph, product)

    # Summary table
    from collections import defaultdict
    from rich.table import Table as RTable
    cat_counts: dict[str, int] = defaultdict(int)
    for node, attrs in graph.nodes(data=True):
        if attrs["type"] == "aspect":
            for pred in graph.predecessors(node):
                cat_counts[pred] += attrs["doc_count"]

    tbl = RTable(title="Category breakdown", show_lines=False)
    tbl.add_column("Category", style="cyan")
    tbl.add_column("Mentions", style="green", justify="right")
    tbl.add_column("Aspects", style="yellow", justify="right")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        n_aspects = sum(
            1 for n in graph.successors(cat) if graph.nodes[n]["type"] == "aspect"
        )
        tbl.add_row(cat, str(cnt), str(n_aspects))
    console.print(tbl)


@app.command()
def absa(
    product: ProductArg,
    paradigms: Annotated[str, typer.Option("--paradigms", "-p",
        help="Comma-separated: transformer,llm,lexicon")] = "transformer,llm,lexicon",
    llm_sample: Annotated[int, typer.Option("--llm-sample",
        help="Max sentences sent to Gemini")] = 300,
    force: ForceOpt = False,
) -> None:
    """Run multi-paradigm ABSA on preprocessed + topic-modeled data."""
    from absa.data.collector import _slugify
    from absa.models.topic_model import run_topic_model
    from absa.models.aspect_mapper import build_aspect_graph, load_graph
    from absa.analysis.aggregator import (
        aggregate, compare_weighting_schemes,
        print_aspect_table, print_product_summary,
    )
    from absa.evaluation.comparator import (
        run_full_evaluation,
        print_evaluation_report,
    )
    from absa.models.absa_model import run_absa

    slug              = _slugify(product)
    sentences_file    = settings.processed_dir / slug / "sentences.json"
    graph_file        = settings.results_dir  / slug / "topics" / "aspect_graph.json"

    if not sentences_file.exists():
        print_error(f"No preprocessed data. Run `absa preprocess \"{product}\"` first.")
        raise typer.Exit(1)
    if not graph_file.exists():
        print_error(f"No aspect graph. Run `absa topics \"{product}\"` first.")
        raise typer.Exit(1)

    sentences = json.loads(sentences_file.read_text(encoding="utf-8"))
    graph     = load_graph(graph_file)
    p_list    = [p.strip() for p in paradigms.split(",")]
    out_dir   = settings.results_dir / slug / "absa"

    # ---- Stage 4: ABSA ----
    print_header("Stage 4 — ABSA", f"{len(sentences)} sentences | paradigms: {p_list}")
    results = run_absa(
        sentences=sentences,
        aspect_graph=graph,
        out_dir=out_dir,
        force=force,
        paradigms=p_list,
        llm_sample=llm_sample,
    )

    # ---- Stage 5: Aggregation ----
    print_header("Stage 5 — Aggregation", "weighted + uniform")
    w_scores = aggregate(results, graph, product, weighting="weighted")
    u_scores = aggregate(results, graph, product, weighting="uniform")
    from absa.analysis.aggregator import save_aggregation
    save_aggregation(w_scores, out_dir=out_dir, filename="aggregated_weighted.json")
    save_aggregation(u_scores, out_dir=out_dir, filename="aggregated_uniform.json")

    print_product_summary(w_scores)
    print_aspect_table(w_scores, weighting="weighted")

    # ---- Stage 6: Evaluation ----
    print_header("Stage 6 — Evaluation", "Cohen's kappa, JSD, H(A), D(A)")
    # Load topic result for coherence
    topic_cache = settings.results_dir / slug / "topics" / "topics.json"
    topic_result = None
    if topic_cache.exists():
        from absa.models.topic_model import Topic, HierarchyEdge, TopicModelResult
        topics    = [Topic(**t) for t in json.loads(topic_cache.read_text(encoding="utf-8"))]
        hier_file = settings.results_dir / slug / "topics" / "hierarchy.json"
        hier      = [HierarchyEdge(**e) for e in json.loads(hier_file.read_text(encoding="utf-8"))]
        topic_result = TopicModelResult(topics=topics, hierarchy=hier)

    w_comps = compare_weighting_schemes(results, graph, product)
    eval_report = run_full_evaluation(
        absa_results=results,
        aggregated_scores=w_scores,
        topic_result=topic_result,
        sentences=sentences,
        product=product,
        out_dir=out_dir,
        weighting_comparisons=w_comps,
    )
    print_evaluation_report(eval_report)

    print_success(
        f"ABSA complete: {len(results)} paradigms analysed. "
        f"Results -> {out_dir.relative_to(settings.root)}"
    )


@app.command()
def compare(
    product: ProductArg,
) -> None:
    """Show cached evaluation metrics and aspect comparison for a product."""
    from absa.data.collector import _slugify
    from absa.analysis.aggregator import load_aggregation, print_aspect_table, print_product_summary
    from absa.evaluation.comparator import EvaluationReport, print_evaluation_report

    slug = _slugify(product)
    agg_file  = settings.results_dir / slug / "absa" / "aggregated_weighted.json"
    eval_file = settings.results_dir / slug / "absa" / "evaluation.json"

    if not agg_file.exists():
        print_error(f"No ABSA results. Run `absa absa \"{product}\"` first.")
        raise typer.Exit(1)

    scores = load_aggregation(agg_file)
    print_header("ABSA Comparison", product)
    print_product_summary(scores)
    print_aspect_table(scores, weighting="weighted")

    if eval_file.exists():
        from dataclasses import fields
        import dataclasses
        raw = json.loads(eval_file.read_text(encoding="utf-8"))

        from absa.evaluation.comparator import (
            CoherenceResult, AgreementResult, EntropyDominance,
            WeightingImpact, EvaluationReport,
        )
        # Reconstruct report
        report = EvaluationReport(product=raw.get("product", product))
        if raw.get("coherence"):
            report.coherence = CoherenceResult(**raw["coherence"])
        report.agreements = [AgreementResult(**a) for a in raw.get("agreements", [])]
        report.entropy_dominance = [EntropyDominance(**e) for e in raw.get("entropy_dominance", [])]
        if raw.get("weighting_impact"):
            report.weighting_impact = WeightingImpact(**raw["weighting_impact"])
        print_evaluation_report(report)


@app.command()
def analyze(
    product: ProductArg,
    subreddits: SubredditsOpt = None,
    limit: LimitOpt = None,
    time_filter: TimeOpt = "year",
    force: ForceOpt = False,
) -> None:
    """Full pipeline: fetch -> preprocess -> topics -> ABSA -> evaluate."""
    from absa.models.aspect_mapper import print_tree
    from absa.analysis.aggregator import print_product_summary, print_aspect_table
    from absa.evaluation.comparator import print_evaluation_report

    subs = [s.strip() for s in subreddits.split(",")] if subreddits else None
    pipeline = Pipeline(
        subreddits=subs,
        post_limit=limit,
        time_filter=time_filter,
        force=force,
    )
    print_header("HierABSA", f"Analyzing  ·  {product}")
    try:
        result = pipeline.run(product)
    except EnvironmentError as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    if result.aspect_graph:
        print_header("Aspect Hierarchy", product)
        print_tree(result.aspect_graph, product)

    if result.aggregated_scores:
        print_product_summary(result.aggregated_scores)
        print_aspect_table(result.aggregated_scores)

    if result.evaluation:
        print_evaluation_report(result.evaluation)

    print_success(
        f"Full pipeline complete: {result.sentence_count} sentences, "
        f"{len(result.topic_result.topics)} topics, "
        f"{len(result.absa_results)} paradigms."
    )


@app.command()
def report(
    product: ProductArg,
) -> None:
    """Generate final RQ-answering results report (JSON + LaTeX + terminal)."""
    from absa.reporting.results_report import run_report
    run_report(product, save=True)


@app.command()
def info() -> None:
    """Show resolved config and credential status."""
    print_header("HierABSA — Config")

    def _check(label: str, getter) -> None:
        try:
            getter()
            console.print(f"  [green]✓[/green]  {label}")
        except EnvironmentError:
            console.print(f"  [red]✗[/red]  {label}  [dim](not set)[/dim]")

    console.print("\n[bold]Credentials[/bold]")
    _check("Reddit CLIENT_ID",     lambda: settings.reddit_client_id)
    _check("Reddit CLIENT_SECRET", lambda: settings.reddit_client_secret)
    _check("Reddit USER_AGENT",    lambda: settings.reddit_user_agent)
    _check("Gemini API key",       lambda: settings.gemini_api_key)

    console.print("\n[bold]Runtime settings[/bold]")
    rows = [
        ("post_limit",       settings.fetch_post_limit),
        ("comment_limit",    settings.fetch_comment_limit),
        ("time_filter",      settings.fetch_time_filter),
        ("min_score",        settings.fetch_min_score),
        ("min_comments",     settings.fetch_min_comments),
        ("spacy_model",      settings.spacy_model),
        ("embedding_model",  settings.embedding_model),
        ("gemini_model",     settings.gemini_model),
        ("raw_dir",          settings.raw_dir),
        ("processed_dir",    settings.processed_dir),
    ]
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="cyan")
    for k, v in rows:
        table.add_row(k, str(v))
    console.print(table)
