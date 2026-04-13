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
def analyze(
    product: ProductArg,
    subreddits: SubredditsOpt = None,
    limit: LimitOpt = None,
    time_filter: TimeOpt = "year",
    force: ForceOpt = False,
) -> None:
    """Full pipeline: fetch -> preprocess -> topic model (-> ABSA coming soon)."""
    from absa.models.aspect_mapper import print_tree

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

    print_success(
        f"Stages 1-3 complete: {result.sentence_count} sentences, "
        f"{len(result.topic_result.topics)} topics, "
        f"{result.aspect_graph.number_of_nodes()} graph nodes. "
        "ABSA coming in next milestone."
    )


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
