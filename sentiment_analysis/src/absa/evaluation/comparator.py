"""
Cross-paradigm evaluation metrics (no gold labels required).

RQ1 — Topic Coherence
    C_v coherence score: BERTopic topics vs. LDA baseline
    (Computed via gensim; LDA trained on same corpus with same K)

RQ2 — Inter-Paradigm Agreement
    For each pair of paradigms on the overlapping sentence set:
      - Cohen's kappa          (sklearn.metrics.cohen_kappa_score)
      - Jensen-Shannon Div.    (scipy.spatial.distance.jensenshannon)
      - Polarity Entropy H(A)  = -sum_s p(s|a) log2 p(s|a)
      - Dominance D(A)         = |p(pos|a) - p(neg|a)|

RQ3 — Weighting Impact
    Pearson r between uniform and weighted net scores across aspects.
    (Comparison data comes from aggregator.compare_weighting_schemes())

RQ4 — Domain Transfer
    Agreement of transformer (SemEval-trained) with LLM (silver standard)
    as a proxy for out-of-domain generalisation.

Public API
----------
  compute_coherence(sentences, topic_result) -> CoherenceResult
  compute_agreement(results_a, results_b, label_a, label_b) -> AgreementResult
  compute_entropy_dominance(scores) -> list[EntropyDominance]
  run_full_evaluation(results, scores, topic_result, sentences, out_dir)
  -> EvaluationReport
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from absa.models.absa_model import SentenceABSAResult
from absa.analysis.aggregator import AspectSentimentScore, ProductScore
from absa.utils.display import print_info, print_success, print_warning


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CoherenceResult:
    bertopic_cv: float | None
    lda_cv: float | None
    delta: float | None       # BERTopic - LDA (positive = BERTopic wins)
    n_topics_bertopic: int
    n_topics_lda: int


@dataclass
class AgreementResult:
    paradigm_a: str
    paradigm_b: str
    n_sentences: int
    cohen_kappa: float
    jsd_mean: float           # mean JSD over shared aspects
    interpretation: str       # "substantial" / "moderate" / "fair" / "slight"


@dataclass
class EntropyDominance:
    aspect: str
    paradigm: str
    entropy: float            # H(A) in bits
    dominance: float          # D(A) = |pos - neg|
    n_mentions: int


@dataclass
class WeightingImpact:
    pearson_r: float          # correlation between uniform and weighted net scores
    mean_abs_delta: float     # average |weighted - uniform| across aspects
    max_delta_aspect: str
    max_delta_value: float


@dataclass
class EvaluationReport:
    product: str
    coherence: CoherenceResult | None = None
    agreements: list[AgreementResult] = field(default_factory=list)
    entropy_dominance: list[EntropyDominance] = field(default_factory=list)
    weighting_impact: WeightingImpact | None = None


# ---------------------------------------------------------------------------
# RQ1 — Topic Coherence
# ---------------------------------------------------------------------------

def compute_coherence(
    sentences: list[dict[str, Any]],
    topic_result: Any,          # TopicModelResult
    lda_topics: int | None = None,
) -> CoherenceResult:
    """
    Compute C_v coherence for BERTopic topics (and optionally an LDA baseline).

    gensim is used for CoherenceModel; if not installed, returns None scores.
    LDA is trained on the same tokenised corpus with the same number of topics.
    """
    try:
        import gensim
        from gensim import corpora
        from gensim.models import CoherenceModel, LdaModel
        from gensim.utils import simple_preprocess
    except ImportError:
        print_warning("gensim not installed — coherence evaluation skipped.")
        return CoherenceResult(None, None, None, len(topic_result.topics), 0)

    docs = [s["sentence"] for s in sentences]
    tokenised = [simple_preprocess(d) for d in docs]
    dictionary = corpora.Dictionary(tokenised)
    dictionary.filter_extremes(no_below=2, no_above=0.95)
    corpus = [dictionary.doc2bow(t) for t in tokenised]

    # BERTopic topic words — split n-grams into unigrams for gensim compatibility
    def _split_ngrams(words: list[str]) -> list[str]:
        """Expand bigrams (e.g. 'battery life') into individual tokens."""
        result: list[str] = []
        for w in words:
            for tok in w.split():
                tok = tok.lower()
                if tok in dictionary.token2id:
                    result.append(tok)
        return result or words[:5]  # fallback: first 5 as-is

    bertopic_words = [
        _split_ngrams(t.top_words[:10])
        for t in topic_result.topics
        if t.top_words
    ]
    n_bert = len(bertopic_words)

    bert_cv: float | None = None
    if bertopic_words:
        try:
            cm = CoherenceModel(
                topics=bertopic_words,
                texts=tokenised,
                dictionary=dictionary,
                coherence="c_v",
            )
            bert_cv = float(cm.get_coherence())
        except Exception as exc:
            print_warning(f"BERTopic C_v failed: {exc}")

    # LDA baseline
    k = lda_topics or n_bert or 10
    lda_cv: float | None = None
    try:
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k,
                       random_state=42, passes=5, alpha="auto")
        lda_words = [[w for w, _ in lda.show_topic(i, 10)] for i in range(k)]
        cm_lda = CoherenceModel(
            topics=lda_words,
            texts=tokenised,
            dictionary=dictionary,
            coherence="c_v",
        )
        lda_cv = float(cm_lda.get_coherence())
    except Exception as exc:
        print_warning(f"LDA C_v failed: {exc}")

    delta = None
    if bert_cv is not None and lda_cv is not None:
        delta = round(bert_cv - lda_cv, 4)

    return CoherenceResult(
        bertopic_cv=round(bert_cv, 4) if bert_cv is not None else None,
        lda_cv=round(lda_cv, 4) if lda_cv is not None else None,
        delta=delta,
        n_topics_bertopic=n_bert,
        n_topics_lda=k,
    )


# ---------------------------------------------------------------------------
# RQ2 — Inter-paradigm agreement
# ---------------------------------------------------------------------------

def _kappa_interpretation(kappa: float) -> str:
    if kappa > 0.80:
        return "almost perfect"
    if kappa > 0.60:
        return "substantial"
    if kappa > 0.40:
        return "moderate"
    if kappa > 0.20:
        return "fair"
    return "slight"


def _jsd_distributions(
    results_a: list[SentenceABSAResult],
    results_b: list[SentenceABSAResult],
) -> float:
    """
    Compute mean JSD over shared sentences between two paradigms.

    For each sentence, build a sentiment distribution over aspects and compute JSD.
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Build sentence -> distribution map
    SENT_IDX = {"negative": 0, "neutral": 1, "positive": 2}

    def _dist(sr: SentenceABSAResult) -> list[float]:
        counts = [0.0, 0.0, 0.0]
        for op in sr.aspects:
            counts[SENT_IDX.get(op.sentiment, 1)] += op.confidence
        total = sum(counts) or 1.0
        return [c / total for c in counts]

    # Match by sentence text
    map_a = {r.sentence: r for r in results_a}
    map_b = {r.sentence: r for r in results_b}
    common = set(map_a.keys()) & set(map_b.keys())

    if not common:
        return float("nan")

    jsds: list[float] = []
    for sent in common:
        p = _dist(map_a[sent])
        q = _dist(map_b[sent])
        try:
            jsds.append(float(jensenshannon(p, q, base=2.0)))
        except Exception:
            pass

    return round(float(np.nanmean(jsds)), 4) if jsds else float("nan")


def compute_agreement(
    results_a: list[SentenceABSAResult],
    results_b: list[SentenceABSAResult],
    label_a: str,
    label_b: str,
) -> AgreementResult:
    """
    Compute Cohen's kappa + mean JSD between two paradigm result sets.

    Kappa is computed on the dominant sentiment label per sentence
    (selecting the highest-confidence aspect opinion as the sentence label).
    """
    from sklearn.metrics import cohen_kappa_score

    map_a = {r.sentence: r for r in results_a}
    map_b = {r.sentence: r for r in results_b}
    common_sents = sorted(set(map_a.keys()) & set(map_b.keys()))

    if len(common_sents) < 5:
        print_warning(
            f"Too few overlapping sentences ({len(common_sents)}) "
            f"for {label_a} vs {label_b} — skipping kappa."
        )
        return AgreementResult(label_a, label_b, len(common_sents),
                               float("nan"), float("nan"), "insufficient data")

    def _dominant_label(sr: SentenceABSAResult) -> str:
        if not sr.aspects:
            return "neutral"
        best = max(sr.aspects, key=lambda op: op.confidence)
        return best.sentiment

    labels_a = [_dominant_label(map_a[s]) for s in common_sents]
    labels_b = [_dominant_label(map_b[s]) for s in common_sents]

    try:
        kappa = float(cohen_kappa_score(labels_a, labels_b))
    except Exception as exc:
        print_warning(f"Kappa computation failed: {exc}")
        kappa = float("nan")

    jsd = _jsd_distributions(
        [map_a[s] for s in common_sents],
        [map_b[s] for s in common_sents],
    )

    return AgreementResult(
        paradigm_a=label_a,
        paradigm_b=label_b,
        n_sentences=len(common_sents),
        cohen_kappa=round(kappa, 4),
        jsd_mean=jsd,
        interpretation=_kappa_interpretation(kappa) if not math.isnan(kappa) else "n/a",
    )


# ---------------------------------------------------------------------------
# Polarity Entropy + Dominance
# ---------------------------------------------------------------------------

def compute_entropy_dominance(
    scores: dict[str, ProductScore],
) -> list[EntropyDominance]:
    """
    For every (paradigm, aspect) pair compute:
      H(A) = -sum_s p(s|a) log2 p(s|a)   (max = log2(3) ≈ 1.585 bits)
      D(A) = |p(pos|a) - p(neg|a)|
    """
    results: list[EntropyDominance] = []

    for paradigm, ps in scores.items():
        for cat in ps.categories:
            for asp in cat.aspects:
                p_pos = asp.positive
                p_neg = asp.negative
                p_neu = asp.neutral
                probs = [p for p in (p_pos, p_neg, p_neu) if p > 0.0]
                entropy = -sum(p * math.log2(p) for p in probs)
                dominance = abs(p_pos - p_neg)
                results.append(EntropyDominance(
                    aspect=asp.aspect,
                    paradigm=paradigm,
                    entropy=round(entropy, 4),
                    dominance=round(dominance, 4),
                    n_mentions=asp.n_mentions,
                ))

    return sorted(results, key=lambda x: -x.dominance)


# ---------------------------------------------------------------------------
# RQ3 — Weighting impact
# ---------------------------------------------------------------------------

def compute_weighting_impact(
    comparisons: list,  # list[WeightingComparison] from aggregator
) -> WeightingImpact | None:
    """Summarise RQ3: how much does upvote weighting shift net scores?"""
    import numpy as np

    if not comparisons:
        return None

    uniform_scores  = [c.uniform_score  for c in comparisons]
    weighted_scores = [c.weighted_score for c in comparisons]
    deltas          = [c.delta          for c in comparisons]

    try:
        r = float(np.corrcoef(uniform_scores, weighted_scores)[0, 1])
    except Exception:
        r = float("nan")

    max_c = max(comparisons, key=lambda c: abs(c.delta))

    return WeightingImpact(
        pearson_r=round(r, 4),
        mean_abs_delta=round(float(np.mean(np.abs(deltas))), 4),
        max_delta_aspect=f"{max_c.paradigm}/{max_c.aspect}",
        max_delta_value=round(max_c.delta, 4),
    )


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

def run_full_evaluation(
    absa_results: dict[str, list[SentenceABSAResult]],
    aggregated_scores: dict[str, ProductScore],
    topic_result: Any,
    sentences: list[dict[str, Any]],
    product: str,
    out_dir: Path,
    weighting_comparisons: list | None = None,
) -> EvaluationReport:
    """
    Run all evaluation metrics and save to out_dir/evaluation.json.
    """
    report = EvaluationReport(product=product)

    # RQ1 — coherence
    print_info("Computing topic coherence (C_v) …")
    try:
        report.coherence = compute_coherence(sentences, topic_result)
    except Exception as exc:
        print_warning(f"Coherence failed: {exc}")

    # RQ2 — pairwise agreement
    paradigms = [p for p, r in absa_results.items() if r]
    for i in range(len(paradigms)):
        for j in range(i + 1, len(paradigms)):
            pa, pb = paradigms[i], paradigms[j]
            print_info(f"Agreement: {pa} vs {pb} …")
            try:
                ag = compute_agreement(absa_results[pa], absa_results[pb], pa, pb)
                report.agreements.append(ag)
            except Exception as exc:
                print_warning(f"Agreement {pa}/{pb} failed: {exc}")

    # Entropy + Dominance
    report.entropy_dominance = compute_entropy_dominance(aggregated_scores)

    # RQ3
    if weighting_comparisons:
        report.weighting_impact = compute_weighting_impact(weighting_comparisons)

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "evaluation.json"
    path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_success(f"Evaluation report saved -> {path.name}")
    return report


# ---------------------------------------------------------------------------
# Rich display
# ---------------------------------------------------------------------------

def print_evaluation_report(report: EvaluationReport) -> None:
    from rich.table import Table
    from rich.panel import Panel
    from absa.utils.display import console

    console.print(f"\n[bold magenta]Evaluation Report — {report.product}[/bold magenta]\n")

    # RQ1 — Coherence
    if report.coherence:
        c = report.coherence
        bert_str = f"{c.bertopic_cv:.4f}" if c.bertopic_cv is not None else "n/a"
        lda_str  = f"{c.lda_cv:.4f}"  if c.lda_cv  is not None else "n/a"
        delta_str = f"{c.delta:+.4f}" if c.delta is not None else "n/a"
        color     = "green" if (c.delta or 0) > 0 else "red"
        console.print(Panel(
            f"  BERTopic C_v : [cyan]{bert_str}[/cyan]  ({c.n_topics_bertopic} topics)\n"
            f"  LDA C_v      : [dim]{lda_str}[/dim]  ({c.n_topics_lda} topics)\n"
            f"  Delta        : [{color}]{delta_str}[/{color}]",
            title="RQ1 — Topic Coherence",
            expand=False,
        ))

    # RQ2 — Agreement
    if report.agreements:
        tbl = Table(title="RQ2 — Inter-Paradigm Agreement", show_lines=True)
        tbl.add_column("Pair", style="cyan")
        tbl.add_column("N sentences", justify="right")
        tbl.add_column("Cohen kappa", justify="right")
        tbl.add_column("Mean JSD",    justify="right")
        tbl.add_column("Interpretation")
        for ag in report.agreements:
            kappa_str = f"{ag.cohen_kappa:.4f}" if not math.isnan(ag.cohen_kappa) else "n/a"
            jsd_str   = f"{ag.jsd_mean:.4f}"    if not math.isnan(ag.jsd_mean)    else "n/a"
            tbl.add_row(
                f"{ag.paradigm_a} vs {ag.paradigm_b}",
                str(ag.n_sentences),
                kappa_str,
                jsd_str,
                ag.interpretation,
            )
        console.print(tbl)

    # Entropy / Dominance top 10
    if report.entropy_dominance:
        tbl2 = Table(title="RQ2 — Polarity Entropy H(A) + Dominance D(A)  [top 10]", show_lines=False)
        tbl2.add_column("Aspect",    style="green")
        tbl2.add_column("Paradigm",  style="cyan")
        tbl2.add_column("H(A) bits", justify="right")
        tbl2.add_column("D(A)",      justify="right")
        tbl2.add_column("Mentions",  justify="right")
        for ed in report.entropy_dominance[:10]:
            tbl2.add_row(
                ed.aspect, ed.paradigm,
                f"{ed.entropy:.3f}",
                f"{ed.dominance:.3f}",
                str(ed.n_mentions),
            )
        console.print(tbl2)

    # RQ3
    if report.weighting_impact:
        wi = report.weighting_impact
        console.print(Panel(
            f"  Pearson r (uniform vs weighted)  : [cyan]{wi.pearson_r:.4f}[/cyan]\n"
            f"  Mean |delta|                     : [yellow]{wi.mean_abs_delta:.4f}[/yellow]\n"
            f"  Largest shift                    : {wi.max_delta_aspect} "
            f"([bold]{wi.max_delta_value:+.4f}[/bold])",
            title="RQ3 — Upvote Weighting Impact",
            expand=False,
        ))
