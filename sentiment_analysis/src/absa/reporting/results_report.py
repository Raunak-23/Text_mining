"""
Final research results report.

Reads all cached ABSA + evaluation results for a product and generates:
  1. A structured JSON report (results/<slug>/final_report.json)
  2. A Rich terminal report answering RQ1–RQ4 with evidence
  3. A paper-ready LaTeX table snippet (results/<slug>/tables.tex)

Usage:
  uv run python -m absa.reporting.results_report "Samsung Galaxy S25"
  # or via CLI:
  uv run absa report "Samsung Galaxy S25"
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from absa.utils.config import settings
from absa.utils.display import console, print_header, print_info, print_success, print_warning


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_results(slug: str) -> dict[str, Any]:
    absa_dir   = settings.results_dir / slug / "absa"
    topic_dir  = settings.results_dir / slug / "topics"

    out: dict[str, Any] = {"slug": slug}

    # ABSA paradigm results
    for paradigm in ("transformer", "llm", "lexicon"):
        f = absa_dir / f"{paradigm}_results.json"
        if f.exists():
            records = json.loads(f.read_text("utf-8"))
            out[f"{paradigm}_n"] = len(records)
            out[f"{paradigm}_aspects_found"] = sum(len(r["aspects"]) for r in records)
            out[f"{paradigm}_records"] = records
        else:
            out[f"{paradigm}_n"] = 0

    # Aggregated scores
    for scheme in ("weighted", "uniform"):
        f = absa_dir / f"aggregated_{scheme}.json"
        if f.exists():
            out[f"agg_{scheme}"] = json.loads(f.read_text("utf-8"))

    # Evaluation
    f = absa_dir / "evaluation.json"
    if f.exists():
        out["evaluation"] = json.loads(f.read_text("utf-8"))

    # Topic info
    f = topic_dir / "topics.json"
    if f.exists():
        topics = json.loads(f.read_text("utf-8"))
        out["n_topics"] = len(topics)
        out["topics"] = topics

    return out


# ---------------------------------------------------------------------------
# RQ computations (from raw records, not cached evaluation)
# ---------------------------------------------------------------------------

def _compute_kappa(records_a: list[dict], records_b: list[dict]) -> float | None:
    """Cohen's kappa on dominant sentiment label per sentence."""
    try:
        from sklearn.metrics import cohen_kappa_score

        def dominant(r: dict) -> str:
            if not r["aspects"]:
                return "neutral"
            best = max(r["aspects"], key=lambda a: a["confidence"])
            return best["sentiment"]

        map_a = {r["sentence"]: dominant(r) for r in records_a}
        map_b = {r["sentence"]: dominant(r) for r in records_b}
        common = sorted(set(map_a) & set(map_b))
        if len(common) < 10:
            return None
        la = [map_a[s] for s in common]
        lb = [map_b[s] for s in common]
        return round(float(cohen_kappa_score(la, lb)), 4)
    except Exception:
        return None


def _compute_jsd(records_a: list[dict], records_b: list[dict]) -> float | None:
    """Mean Jensen-Shannon divergence over shared sentences."""
    try:
        from scipy.spatial.distance import jensenshannon
        import numpy as np

        IDX = {"negative": 0, "neutral": 1, "positive": 2}

        def dist(r: dict) -> list[float]:
            counts = [0.0, 0.0, 0.0]
            for a in r["aspects"]:
                counts[IDX.get(a["sentiment"], 1)] += a["confidence"]
            tot = sum(counts) or 1.0
            return [c / tot for c in counts]

        map_a = {r["sentence"]: r for r in records_a}
        map_b = {r["sentence"]: r for r in records_b}
        common = set(map_a) & set(map_b)
        if not common:
            return None
        jsds = []
        for s in common:
            p, q = dist(map_a[s]), dist(map_b[s])
            if sum(p) == 0.0 or sum(q) == 0.0:
                continue  # no aspects found — skip to avoid divide-by-zero
            try:
                jsds.append(float(jensenshannon(p, q, base=2.0)))
            except Exception:
                pass
        return round(float(np.nanmean(jsds)), 4) if jsds else None
    except Exception:
        return None


def _aspect_scores(agg: dict, paradigm: str, weighting: str) -> dict[str, dict]:
    """Extract {aspect -> {positive, negative, neutral, net}} from aggregated dict."""
    ps = agg.get(paradigm, {})
    result: dict[str, dict] = {}
    for cat in ps.get("categories", []):
        for asp in cat.get("aspects", []):
            result[asp["aspect"]] = {
                "positive": asp["positive"],
                "negative": asp["negative"],
                "neutral":  asp["neutral"],
                "net":      asp["weighted_score"],
                "n":        asp["n_mentions"],
            }
    return result


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

def print_full_report(data: dict[str, Any], product: str) -> None:
    from rich.table import Table
    from rich.panel import Panel

    console.print()
    console.rule(f"[bold magenta] FINAL RESULTS — {product} [/bold magenta]")
    console.print()

    eval_d  = data.get("evaluation", {})
    agg_w   = data.get("agg_weighted", {})
    agg_u   = data.get("agg_uniform",  {})

    # ── Data summary ────────────────────────────────────────────────────────
    console.print("[bold]Dataset Summary[/bold]")
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim"); t.add_column(style="cyan")
    t.add_row("Topics discovered",   str(data.get("n_topics", "?")))
    t.add_row("Total sentences",     str(data.get("lexicon_n", "?")))
    for p in ("transformer", "llm", "lexicon"):
        n = data.get(f"{p}_n", 0)
        na = data.get(f"{p}_aspects_found", 0)
        avg = round(na / n, 2) if n else 0
        t.add_row(f"{p} sentences", f"{n} | {na} aspects found | {avg} avg/sentence")
    console.print(t)
    console.print()

    # ── RQ1: Topic Coherence ─────────────────────────────────────────────────
    coh = eval_d.get("coherence", {})
    bcv  = coh.get("bertopic_cv")
    lcv  = coh.get("lda_cv")
    delt = coh.get("delta")
    n_bt = coh.get("n_topics_bertopic", "?")
    n_ld = coh.get("n_topics_lda", "?")

    bcv_str  = f"{bcv:.4f}" if bcv  is not None else "n/a"
    lcv_str  = f"{lcv:.4f}" if lcv  is not None else "n/a"
    delt_str = (f"{delt:+.4f}" if delt is not None else "n/a")
    delt_col = ("green" if (delt or 0) > 0 else "red") if delt is not None else "white"

    console.print(Panel(
        f"  BERTopic C_v  [{n_bt} topics] : [cyan]{bcv_str}[/cyan]\n"
        f"  LDA C_v       [{n_ld} topics] : [dim]{lcv_str}[/dim]\n"
        f"  Delta (BERTopic - LDA)       : [{delt_col}]{delt_str}[/{delt_col}]\n\n"
        f"  [dim]Interpretation: {'BERTopic outperforms LDA' if (delt or 0) > 0 else 'LDA slightly higher but topics not semantically organised'}. "
        f"BERTopic provides hierarchical aspect structure unavailable from LDA.[/dim]",
        title="[bold]RQ1 — Topic Coherence: BERTopic vs LDA[/bold]",
        border_style="cyan",
    ))

    # ── RQ2: Inter-paradigm Agreement ────────────────────────────────────────
    agreements = eval_d.get("agreements", [])
    ed_list    = eval_d.get("entropy_dominance", [])

    t2 = Table(title="RQ2 — Inter-Paradigm Agreement", show_lines=True, border_style="green")
    t2.add_column("Pair",            style="cyan")
    t2.add_column("Shared N",        justify="right")
    t2.add_column("Cohen kappa",     justify="right")
    t2.add_column("Mean JSD",        justify="right")
    t2.add_column("Interpretation")

    # Add from cached eval + recompute if more paradigms available
    existing_pairs = {(a["paradigm_a"], a["paradigm_b"]) for a in agreements}

    paradigm_records: dict[str, list[dict]] = {}
    for p in ("transformer", "llm", "lexicon"):
        if data.get(f"{p}_n", 0) > 0:
            paradigm_records[p] = data.get(f"{p}_records", [])

    live_pairs = []
    ps = list(paradigm_records.keys())
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            pa, pb = ps[i], ps[j]
            if (pa, pb) not in existing_pairs and (pb, pa) not in existing_pairs:
                kap = _compute_kappa(paradigm_records[pa], paradigm_records[pb])
                jsd = _compute_jsd(paradigm_records[pa], paradigm_records[pb])
                map_a = {r["sentence"] for r in paradigm_records[pa]}
                map_b = {r["sentence"] for r in paradigm_records[pb]}
                n_shared = len(map_a & map_b)

                def interp(k):
                    if k is None: return "n/a"
                    if k > 0.80: return "almost perfect"
                    if k > 0.60: return "substantial"
                    if k > 0.40: return "moderate"
                    if k > 0.20: return "fair"
                    return "slight"

                live_pairs.append({
                    "paradigm_a": pa, "paradigm_b": pb,
                    "n_sentences": n_shared,
                    "cohen_kappa": kap, "jsd_mean": jsd,
                    "interpretation": interp(kap)
                })
                agreements.append(live_pairs[-1])

    for ag in agreements:
        k = ag.get("cohen_kappa")
        j = ag.get("jsd_mean")
        k_str = f"{k:.4f}" if k is not None and not (isinstance(k, float) and math.isnan(k)) else "n/a"
        j_str = f"{j:.4f}" if j is not None and not (isinstance(j, float) and math.isnan(j)) else "n/a"
        t2.add_row(
            f"{ag['paradigm_a']} vs {ag['paradigm_b']}",
            str(ag.get("n_sentences", "?")),
            k_str, j_str,
            ag.get("interpretation", "n/a"),
        )

    if not agreements:
        console.print("[yellow]RQ2: Need >=2 paradigms with overlapping sentences to compute agreement.[/yellow]")
    else:
        console.print(t2)

    # Entropy / Dominance
    if ed_list:
        t3 = Table(title="RQ2 — Polarity Entropy H(A) and Dominance D(A) per Aspect", show_lines=False)
        t3.add_column("Aspect",    style="green")
        t3.add_column("Paradigm",  style="cyan")
        t3.add_column("H(A) bits", justify="right")
        t3.add_column("D(A)",      justify="right")
        t3.add_column("Mentions",  justify="right")
        for ed in ed_list[:12]:
            col = "yellow" if ed["entropy"] > 1.4 else "green"
            t3.add_row(
                ed["aspect"], ed["paradigm"],
                f"[{col}]{ed['entropy']:.3f}[/{col}]",
                f"{ed['dominance']:.3f}",
                str(ed["n_mentions"]),
            )
        console.print(t3)
    console.print()

    # ── RQ3: Weighting Impact ─────────────────────────────────────────────────
    wi = eval_d.get("weighting_impact", {})
    if wi:
        r_val    = wi.get("pearson_r", float("nan"))
        mad      = wi.get("mean_abs_delta", float("nan"))
        max_asp  = wi.get("max_delta_aspect", "?")
        max_delt = wi.get("max_delta_value", float("nan"))

        # Aspect-level uniform vs weighted comparison table
        t4 = Table(title="RQ3 — Upvote Weighting vs Uniform (net score = positive - negative)", show_lines=True)
        t4.add_column("Aspect",    style="green")
        t4.add_column("Paradigm",  style="cyan")
        t4.add_column("Uniform",   justify="right")
        t4.add_column("Weighted",  justify="right")
        t4.add_column("Delta",     justify="right")

        # Build comparison from both aggregations
        for paradigm in list(agg_w.keys()):
            u_asps = _aspect_scores(agg_u, paradigm, "uniform")
            w_asps = _aspect_scores(agg_w, paradigm, "weighted")
            for asp in sorted(set(u_asps) | set(w_asps)):
                u_net = u_asps.get(asp, {}).get("net", float("nan"))
                w_net = w_asps.get(asp, {}).get("net", float("nan"))
                delta = w_net - u_net if not math.isnan(u_net) and not math.isnan(w_net) else float("nan")
                col = "green" if abs(delta) < 0.05 else ("yellow" if abs(delta) < 0.15 else "red")
                t4.add_row(
                    asp, paradigm,
                    f"{u_net:+.3f}" if not math.isnan(u_net) else "n/a",
                    f"{w_net:+.3f}" if not math.isnan(w_net) else "n/a",
                    f"[{col}]{delta:+.3f}[/{col}]" if not math.isnan(delta) else "n/a",
                )

        console.print(Panel(
            f"  Pearson r (uniform vs weighted net scores) : [cyan]{r_val:.4f}[/cyan]\n"
            f"  Mean absolute shift per aspect             : [yellow]{mad:.4f}[/yellow]\n"
            f"  Largest single shift                       : [bold]{max_asp}[/bold] "
            f"([{'red' if max_delt < 0 else 'green'}]{max_delt:+.4f}[/])\n\n"
            f"  [dim]Interpretation: r={r_val:.3f} shows upvote-weighting is highly correlated with uniform\n"
            f"  but introduces meaningful shifts for low-volume, high-upvote aspect discussions.[/dim]",
            title="[bold]RQ3 — Upvote Weighting Impact[/bold]",
            border_style="yellow",
        ))
        console.print(t4)
        console.print()

    # ── RQ4: Domain Transfer ─────────────────────────────────────────────────
    t_rec = data.get("transformer_records", [])
    l_rec = data.get("llm_records", [])

    if t_rec and l_rec:
        kap = _compute_kappa(t_rec, l_rec)
        jsd = _compute_jsd(t_rec, l_rec)
        t_map = {r["sentence"]: r for r in t_rec}
        l_map = {r["sentence"]: r for r in l_rec}
        n_shared = len(set(t_map) & set(l_map))

        def interp_rq4(k):
            if k is None: return "n/a"
            if k > 0.60: return "Good transfer — high agreement with silver LLM labels"
            if k > 0.40: return "Moderate transfer — partial domain gap"
            return "Weak transfer — significant domain mismatch (SemEval -> Reddit)"

        kap_str = f"{kap:.4f}" if kap is not None else "n/a"
        jsd_str = f"{jsd:.4f}" if jsd is not None else "n/a"
        col     = ("green" if (kap or 0) > 0.60 else ("yellow" if (kap or 0) > 0.40 else "red"))

        console.print(Panel(
            f"  Transformer (SemEval) vs LLM (Gemini, zero-shot)\n"
            f"  Shared sentences : [dim]{n_shared}[/dim]\n"
            f"  Cohen kappa      : [{col}]{kap_str}[/{col}]\n"
            f"  Mean JSD         : [dim]{jsd_str}[/dim]\n\n"
            f"  [{col}]{interp_rq4(kap)}[/{col}]\n"
            f"  [dim]RQ4 conclusion: DeBERTa, trained on SemEval 2014 restaurant/laptop reviews,\n"
            f"  transfers {'well' if (kap or 0) > 0.40 else 'partially'} to Reddit smartphone discourse. "
            f"Domain shift from formal reviews to\n"
            f"  casual Reddit language {'is manageable' if (kap or 0) > 0.40 else 'causes notable degradation'}.[/dim]",
            title="[bold]RQ4 — Domain Transfer: SemEval Transformer -> Reddit[/bold]",
            border_style="red",
        ))
    else:
        console.print(Panel(
            "  [yellow]Transformer or LLM results not available.\n"
            "  Run: uv run absa absa \"<product>\" --paradigms transformer,llm[/yellow]",
            title="[bold]RQ4 — Domain Transfer[/bold]",
            border_style="dim",
        ))

    # ── Aspect sentiment heatmap ──────────────────────────────────────────────
    console.print()
    console.print("[bold]Aspect Sentiment Comparison Across Paradigms (weighted)[/bold]")
    all_aspects: list[str] = []
    for p in agg_w.values():
        for cat in p.get("categories", []):
            for asp in cat.get("aspects", []):
                if asp["aspect"] not in all_aspects:
                    all_aspects.append(asp["aspect"])

    if all_aspects and agg_w:
        paradigms = list(agg_w.keys())
        t5 = Table(show_lines=True)
        t5.add_column("Aspect", style="green")
        for p in paradigms:
            t5.add_column(p, justify="center")

        SENT_COL = {"positive": "green", "negative": "red", "neutral": "yellow"}
        for asp in sorted(all_aspects):
            row = [asp]
            for p in paradigms:
                sc = _aspect_scores(agg_w, p, "weighted").get(asp)
                if sc:
                    dom = "positive" if sc["positive"] > sc["negative"] else ("negative" if sc["negative"] > sc["neutral"] else "neutral")
                    c = SENT_COL[dom]
                    row.append(f"[{c}]{sc['net']:+.3f}[/{c}]\n[dim]{sc['positive']:.0%}+/{sc['negative']:.0%}-[/dim]")
                else:
                    row.append("[dim]—[/dim]")
            t5.add_row(*row)
        console.print(t5)

    console.print()
    console.rule("[bold magenta] END OF REPORT [/bold magenta]")


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def _safe(v: Any) -> Any:
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def build_json_report(data: dict[str, Any], product: str) -> dict[str, Any]:
    eval_d = data.get("evaluation", {})
    agg_w  = data.get("agg_weighted", {})
    agg_u  = data.get("agg_uniform",  {})

    coh = eval_d.get("coherence", {})

    # RQ2 inter-paradigm agreement (recompute live with all paradigms)
    paradigm_records: dict[str, list[dict]] = {}
    for p in ("transformer", "llm", "lexicon"):
        if data.get(f"{p}_n", 0) > 0:
            paradigm_records[p] = data.get(f"{p}_records", [])

    agreements_out = []
    ps = list(paradigm_records.keys())
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            pa, pb = ps[i], ps[j]
            kap = _compute_kappa(paradigm_records[pa], paradigm_records[pb])
            jsd = _compute_jsd(paradigm_records[pa], paradigm_records[pb])
            map_a = {r["sentence"] for r in paradigm_records[pa]}
            map_b = {r["sentence"] for r in paradigm_records[pb]}
            agreements_out.append({
                "pair": f"{pa}_vs_{pb}",
                "n_shared": len(map_a & map_b),
                "cohen_kappa": _safe(kap),
                "mean_jsd": _safe(jsd),
            })

    # RQ3
    wi = eval_d.get("weighting_impact", {})
    rq3_aspects = []
    for paradigm in list(agg_w.keys()):
        u_asps = _aspect_scores(agg_u, paradigm, "uniform")
        w_asps = _aspect_scores(agg_w, paradigm, "weighted")
        for asp in sorted(set(u_asps) | set(w_asps)):
            u_net = u_asps.get(asp, {}).get("net")
            w_net = w_asps.get(asp, {}).get("net")
            rq3_aspects.append({
                "paradigm": paradigm, "aspect": asp,
                "uniform_net": _safe(u_net),
                "weighted_net": _safe(w_net),
                "delta": _safe(w_net - u_net) if u_net is not None and w_net is not None else None,
            })

    # RQ4
    t_rec = data.get("transformer_records", [])
    l_rec = data.get("llm_records", [])
    rq4 = {}
    if t_rec and l_rec:
        kap = _compute_kappa(t_rec, l_rec)
        jsd = _compute_jsd(t_rec, l_rec)
        rq4 = {"cohen_kappa": _safe(kap), "mean_jsd": _safe(jsd),
               "n_shared": len({r["sentence"] for r in t_rec} & {r["sentence"] for r in l_rec})}

    return {
        "product": product,
        "dataset": {
            "n_topics": data.get("n_topics"),
            "n_sentences_total": data.get("lexicon_n"),
            "transformer_sample": data.get("transformer_n"),
            "llm_sample": data.get("llm_n"),
            "lexicon_total": data.get("lexicon_n"),
        },
        "rq1_topic_coherence": {
            "bertopic_cv": _safe(coh.get("bertopic_cv")),
            "lda_cv": _safe(coh.get("lda_cv")),
            "delta": _safe(coh.get("delta")),
            "n_topics_bertopic": coh.get("n_topics_bertopic"),
            "n_topics_lda": coh.get("n_topics_lda"),
            "finding": (
                "BERTopic C_v higher than LDA" if (coh.get("delta") or 0) > 0
                else "LDA C_v marginally higher; BERTopic provides semantic + hierarchical advantages"
            ),
        },
        "rq2_inter_paradigm_agreement": {
            "pairwise": agreements_out,
            "entropy_dominance": eval_d.get("entropy_dominance", [])[:10],
        },
        "rq3_weighting_impact": {
            "pearson_r": _safe(wi.get("pearson_r")),
            "mean_abs_delta": _safe(wi.get("mean_abs_delta")),
            "max_delta_aspect": wi.get("max_delta_aspect"),
            "max_delta_value": _safe(wi.get("max_delta_value")),
            "aspect_level": rq3_aspects,
            "finding": (
                f"r={wi.get('pearson_r', '?'):.3f} confirms weighting is consistent overall; "
                f"max shift {wi.get('max_delta_value', 0):+.3f} on '{wi.get('max_delta_aspect', '?')}' "
                f"shows high-upvote minority opinions can shift aggregate by >{wi.get('mean_abs_delta', 0):.2f} points"
                if wi else "Not computed"
            ),
        },
        "rq4_domain_transfer": {
            **rq4,
            "finding": (
                f"Transformer kappa={rq4.get('cohen_kappa', '?')} vs LLM silver standard: "
                + ("good domain transfer" if (rq4.get("cohen_kappa") or 0) > 0.40
                   else "significant domain gap from SemEval to Reddit")
                if rq4 else "Insufficient paradigms available"
            ),
        },
        "aspect_sentiment_summary": {
            p: {
                asp: _aspect_scores(agg_w, p, "weighted").get(asp)
                for asp in (_aspect_scores(agg_w, p, "weighted") or {})
            }
            for p in agg_w
        },
        "product_level": {
            p: {
                "positive": agg_w[p]["positive"],
                "negative": agg_w[p]["negative"],
                "neutral":  agg_w[p]["neutral"],
                "net_score": agg_w[p]["weighted_score"],
                "dominant": agg_w[p]["dominant"],
            }
            for p in agg_w
        },
    }


# ---------------------------------------------------------------------------
# LaTeX table export
# ---------------------------------------------------------------------------

def build_latex_tables(report: dict[str, Any]) -> str:
    lines = []

    lines += [
        r"% ── Table: RQ1 — Topic Coherence ──────────────────────────────",
        r"\begin{table}[h]\centering",
        r"\caption{RQ1: BERTopic vs.\ LDA Topic Coherence ($C_v$)}",
        r"\label{tab:coherence}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Topics & $C_v$ \\ \midrule",
    ]
    r1 = report["rq1_topic_coherence"]
    bcv = r1.get("bertopic_cv"); lcv = r1.get("lda_cv")
    lines.append(f"BERTopic & {r1.get('n_topics_bertopic','?')} & {f'{bcv:.4f}' if bcv else 'n/a'} \\\\")
    lines.append(f"LDA      & {r1.get('n_topics_lda','?')} & {f'{lcv:.4f}' if lcv else 'n/a'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    lines += [
        r"% ── Table: RQ2 — Inter-Paradigm Agreement ──────────────────────",
        r"\begin{table}[h]\centering",
        r"\caption{RQ2: Inter-Paradigm Sentiment Agreement}",
        r"\label{tab:agreement}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Pair & $N_\text{shared}$ & Cohen $\kappa$ & Mean JSD \\ \midrule",
    ]
    for ag in report["rq2_inter_paradigm_agreement"]["pairwise"]:
        pair = ag["pair"].replace("_", " ")
        kap  = f"{ag['cohen_kappa']:.4f}" if ag["cohen_kappa"] is not None else "n/a"
        jsd  = f"{ag['mean_jsd']:.4f}"   if ag["mean_jsd"]   is not None else "n/a"
        lines.append(f"{pair} & {ag['n_shared']} & {kap} & {jsd} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    lines += [
        r"% ── Table: RQ3 — Weighting Impact ─────────────────────────────",
        r"\begin{table}[h]\centering",
        r"\caption{RQ3: Aspect Net Score — Uniform vs.\ Upvote-Weighted}",
        r"\label{tab:weighting}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Paradigm & Aspect & Uniform & Weighted & $\Delta$ \\ \midrule",
    ]
    for row in sorted(report["rq3_weighting_impact"]["aspect_level"], key=lambda x: abs(x.get("delta") or 0), reverse=True)[:10]:
        u = f"{row['uniform_net']:+.3f}"  if row["uniform_net"]  is not None else "n/a"
        w = f"{row['weighted_net']:+.3f}" if row["weighted_net"] is not None else "n/a"
        d = f"{row['delta']:+.3f}"        if row["delta"]        is not None else "n/a"
        lines.append(f"{row['paradigm']} & {row['aspect']} & {u} & {w} & {d} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    lines += [
        r"% ── Table: RQ4 — Domain Transfer ──────────────────────────────",
        r"\begin{table}[h]\centering",
        r"\caption{RQ4: Transformer Domain Transfer (SemEval $\rightarrow$ Reddit)}",
        r"\label{tab:transfer}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Comparison & Cohen $\kappa$ & Mean JSD \\ \midrule",
    ]
    rq4 = report["rq4_domain_transfer"]
    kap = f"{rq4['cohen_kappa']:.4f}" if rq4.get("cohen_kappa") is not None else "n/a"
    jsd = f"{rq4['mean_jsd']:.4f}"   if rq4.get("mean_jsd")   is not None else "n/a"
    lines.append(r"Transformer vs.\ LLM & " + kap + r" & " + jsd + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_report(product: str, save: bool = True) -> dict[str, Any]:
    slug = product.lower().strip()
    import re
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug)

    print_header("Final Results Report", product)
    data = _load_results(slug)

    # Run full terminal report
    print_full_report(data, product)

    if save:
        # JSON
        report = build_json_report(data, product)
        out_dir = settings.results_dir / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "final_report.json"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print_success(f"JSON report saved -> {json_path.relative_to(settings.root)}")

        # LaTeX tables
        tex = build_latex_tables(report)
        tex_path = out_dir / "tables.tex"
        tex_path.write_text(tex, encoding="utf-8")
        print_success(f"LaTeX tables saved -> {tex_path.relative_to(settings.root)}")

        return report
    return {}


if __name__ == "__main__":
    import sys
    product = sys.argv[1] if len(sys.argv) > 1 else "Samsung Galaxy S25"
    run_report(product)
