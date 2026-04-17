"""
Visualizer for HierABSA results.

Generates two PNG figures per product:
  1. aspect_hierarchy.png  — hierarchical tree: Product→Category→Aspect→Topic
                             Aspect nodes are colour-coded by average net sentiment.
  2. sentiment_heatmap.png — aspect × paradigm net-score heatmap.

Saved to  outputs/visualizations/<slug>/
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")           # non-interactive — works headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from absa.utils.config import settings
from absa.utils.display import print_success, print_warning


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_PRODUCT_COLOR  = "#7c3aed"   # violet
_CATEGORY_COLOR = "#1d4ed8"   # blue
_TOPIC_COLOR    = "#94a3b8"   # slate-400

def _net_color(net: float | None) -> str:
    """Green for positive, red for negative, amber for neutral."""
    if net is None:
        return "#64748b"
    if net > 0.12:
        return "#16a34a"
    if net < -0.12:
        return "#dc2626"
    return "#d97706"


def _aspect_net(node: str, agg: dict[str, Any]) -> float | None:
    """Mean net score across paradigms for an aspect node."""
    vals: list[float] = []
    for p_data in agg.values():
        for cat in p_data.get("categories", []):
            for asp in cat.get("aspects", []):
                if asp["aspect"] == node:
                    vals.append(asp["weighted_score"])
    return float(np.mean(vals)) if vals else None


def _node_color(node: str, G: nx.DiGraph, agg: dict[str, Any]) -> str:
    ntype = G.nodes[node].get("type", "topic")
    if ntype == "product":
        return _PRODUCT_COLOR
    if ntype == "category":
        return _CATEGORY_COLOR
    if ntype == "aspect":
        return _net_color(_aspect_net(node, agg))
    return _TOPIC_COLOR


# ---------------------------------------------------------------------------
# Tree layout
# ---------------------------------------------------------------------------

def _tree_pos(G: nx.DiGraph, root: str) -> dict[str, tuple[float, float]]:
    """
    Compute (x, y) positions for a rooted tree.
    Root is at y=1.0; leaves at y=0.0.
    Width is distributed proportionally to subtree leaf-count.
    """
    # BFS depth map
    depth: dict[str, int] = {root: 0}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for child in G.successors(node):
            if child not in depth:
                depth[child] = depth[node] + 1
                queue.append(child)

    max_depth = max(depth.values()) if depth else 1

    def _leaves(n: str) -> int:
        ch = [c for c in G.successors(n) if c in depth]
        return sum(_leaves(c) for c in ch) if ch else 1

    pos: dict[str, tuple[float, float]] = {}

    def _place(node: str, left: float, right: float) -> None:
        d = depth.get(node, 0)
        pos[node] = ((left + right) / 2, 1.0 - d / max_depth)
        children = [c for c in G.successors(node) if c in depth]
        if not children:
            return
        total = sum(_leaves(c) for c in children)
        x = left
        for ch in children:
            w = (right - left) * _leaves(ch) / total
            _place(ch, x, x + w)
            x += w

    _place(root, 0.0, 1.0)
    return pos


# ---------------------------------------------------------------------------
# Figure 1 — Hierarchy tree
# ---------------------------------------------------------------------------

def plot_hierarchy(
    G: nx.DiGraph,
    product: str,
    agg: dict[str, Any],
    out_path: Path,
) -> None:
    """Draw aspect hierarchy as a colour-coded tree and save to *out_path*."""
    if G.number_of_nodes() == 0:
        print_warning("Empty graph — skipping hierarchy visualisation.")
        return

    pos = _tree_pos(G, product)
    visible = set(pos)
    edges = [(u, v) for u, v in G.edges() if u in visible and v in visible]

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.12, 1.08)
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    # edges
    for u, v in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="#cbd5e1", lw=0.9, zorder=1)

    # nodes + labels
    SIZE = {"product": 1800, "category": 1000, "aspect": 650, "topic": 220}
    FONT = {"product": (11, "bold"), "category": (9, "bold"),
            "aspect": (8, "bold"), "topic": (6, "normal")}
    OFFSET = {"product": 0.05, "category": 0.042, "aspect": 0.038, "topic": 0.028}

    for node in visible:
        x, y   = pos[node]
        ntype  = G.nodes[node].get("type", "topic")
        color  = _node_color(node, G, agg)
        s      = SIZE.get(ntype, 220)
        fsize, fweight = FONT.get(ntype, (6, "normal"))
        off    = OFFSET.get(ntype, 0.028)

        ax.scatter(x, y, s=s, c=color, zorder=4,
                   edgecolors="white", linewidths=1.5, alpha=0.93)

        label = G.nodes[node].get("label", node)
        if ntype == "topic":
            words = G.nodes[node].get("top_words", [])
            label = ", ".join(words[:2]) if words else label[:18]
        elif ntype == "aspect":
            net = _aspect_net(node, agg)
            suffix = f"\n({net:+.2f})" if net is not None else ""
            label = label + suffix

        ax.text(x, y - off, label, ha="center", va="top",
                fontsize=fsize, fontweight=fweight,
                color="#1e293b", zorder=5)

    # legend
    patches = [
        mpatches.Patch(color=_PRODUCT_COLOR,  label="Product"),
        mpatches.Patch(color=_CATEGORY_COLOR, label="Category"),
        mpatches.Patch(color="#16a34a",        label="Aspect — positive"),
        mpatches.Patch(color="#d97706",        label="Aspect — neutral"),
        mpatches.Patch(color="#dc2626",        label="Aspect — negative"),
        mpatches.Patch(color=_TOPIC_COLOR,     label="Topic (BERTopic cluster)"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              framealpha=0.85, ncol=2, edgecolor="#e2e8f0")

    ax.set_title(
        f"Hierarchical Aspect Graph — {product}",
        fontsize=14, fontweight="bold", color="#0f172a", pad=12,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print_success(f"Hierarchy graph  -> {out_path.relative_to(settings.root)}")


# ---------------------------------------------------------------------------
# Figure 2 — Sentiment heatmap
# ---------------------------------------------------------------------------

def plot_sentiment_heatmap(
    agg: dict[str, Any],
    product: str,
    out_path: Path,
) -> None:
    """Draw aspect × paradigm net-sentiment heatmap and save to *out_path*."""
    if not agg:
        print_warning("No aggregated scores — skipping heatmap.")
        return

    # Collect aspects
    all_aspects: list[str] = []
    for p_data in agg.values():
        for cat in p_data.get("categories", []):
            for asp in cat.get("aspects", []):
                if asp["aspect"] not in all_aspects:
                    all_aspects.append(asp["aspect"])
    all_aspects = sorted(all_aspects)
    paradigms   = list(agg.keys())

    if not all_aspects or not paradigms:
        return

    # Build value matrix
    matrix = np.full((len(all_aspects), len(paradigms)), np.nan)
    for j, paradigm in enumerate(paradigms):
        asp_map: dict[str, float] = {}
        for cat in agg[paradigm].get("categories", []):
            for asp in cat.get("aspects", []):
                asp_map[asp["aspect"]] = asp["weighted_score"]
        for i, asp in enumerate(all_aspects):
            if asp in asp_map:
                matrix[i, j] = asp_map[asp]

    # ---- figure ----
    h = max(4.5, len(all_aspects) * 0.72)
    w = max(5.0, len(paradigms) * 2.8)
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    vmax = max(0.5, float(np.nanmax(np.abs(matrix))))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    # cell annotations
    for i in range(len(all_aspects)):
        for j in range(len(paradigms)):
            val = matrix[i, j]
            if not np.isnan(val):
                tc = "white" if abs(val) > vmax * 0.55 else "#1e293b"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=tc)

    ax.set_xticks(range(len(paradigms)))
    ax.set_xticklabels([p.capitalize() for p in paradigms], fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(all_aspects)))
    ax.set_yticklabels(all_aspects, fontsize=10)
    ax.tick_params(length=0)

    # gridlines
    for x in np.arange(-0.5, len(paradigms), 1):
        ax.axvline(x, color="white", lw=1.5)
    for y in np.arange(-0.5, len(all_aspects), 1):
        ax.axhline(y, color="white", lw=1.5)

    ax.set_title(f"Aspect Sentiment — {product}  (net score = pos − neg)",
                 fontsize=13, fontweight="bold", color="#0f172a", pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Net Sentiment  [−1 … +1]", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print_success(f"Sentiment heatmap -> {out_path.relative_to(settings.root)}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def save_visualizations(
    product: str,
    graph: nx.DiGraph,
    agg_raw: dict[str, Any],
) -> dict[str, Path]:
    """
    Generate and save both visualisations for *product*.

    Parameters
    ----------
    product  : Product name string.
    graph    : NetworkX DiGraph from build_aspect_graph().
    agg_raw  : Raw aggregated dict (as loaded from aggregated_weighted.json).
               Keys are paradigm names; values match ProductScore JSON structure.

    Returns
    -------
    Dict mapping figure names to their saved paths.
    """
    slug = re.sub(r"[^\w\s-]", "", product.lower().strip())
    slug = re.sub(r"[\s_-]+", "-", slug)
    out_dir = settings.outputs_dir / "visualizations" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    tree_path = out_dir / "aspect_hierarchy.png"
    try:
        plot_hierarchy(graph, product, agg_raw, tree_path)
        saved["hierarchy"] = tree_path
    except Exception as exc:
        print_warning(f"Hierarchy plot failed: {exc}")

    heat_path = out_dir / "sentiment_heatmap.png"
    try:
        plot_sentiment_heatmap(agg_raw, product, heat_path)
        saved["heatmap"] = heat_path
    except Exception as exc:
        print_warning(f"Heatmap failed: {exc}")

    return saved
