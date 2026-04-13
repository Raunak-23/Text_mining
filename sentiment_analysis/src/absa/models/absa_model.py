"""
Multi-paradigm ABSA engine.

Three approaches applied to the same sentence×aspect pairs derived from
the topic/aspect graph:

  1. Transformer  — yangheng/deberta-v3-base-absa-v1.1
                    Input: "sentence [SEP] aspect"
                    Output: negative / neutral / positive
                    (SemEval-trained; domain-transfer study = RQ4)

  2. LLM          — Gemini Flash (google-genai SDK)
                    Zero-shot JSON extraction, batched 10 sentences at a time
                    (cost control: llm_sample cap, default 300 sentences)

  3. Lexicon       — VADER compound score + spaCy noun-chunk candidates
                    Opinion words extracted via amod/advmod/neg dependency arcs
                    (fast interpretable baseline)

All three write to:
  data/results/<slug>/absa/<paradigm>_results.json

Public API
----------
  run_absa(sentences, aspect_graph, out_dir, force, paradigms, llm_sample)
  -> dict[str, list[SentenceABSAResult]]
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from absa.utils.config import settings
from absa.utils.display import print_info, print_success, print_warning


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AspectOpinion:
    aspect: str
    sentiment: str          # "positive" | "neutral" | "negative"
    confidence: float       # [0.0, 1.0]
    opinion_words: list[str] = field(default_factory=list)


@dataclass
class SentenceABSAResult:
    sentence: str
    post_id: str | None
    comment_id: str | None
    subreddit: str | None
    post_score: int
    comment_score: int
    source: str
    aspects: list[AspectOpinion]
    paradigm: str           # "transformer" | "llm" | "lexicon"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _graph_aspect_candidates(graph: nx.DiGraph) -> list[str]:
    """Return all aspect node labels from the taxonomy graph."""
    return [
        graph.nodes[n]["label"]
        for n in graph.nodes
        if graph.nodes[n]["type"] == "aspect"
    ]


def _topic_aspects_for_sentence(
    sentence_record: dict[str, Any],
    graph: nx.DiGraph,
) -> list[str]:
    """
    Return aspect candidates relevant to a sentence.

    Strategy: use all graph aspect labels (relatively small list ~10-20).
    The transformer model is fast enough to score every (sentence, aspect) pair.
    """
    return _graph_aspect_candidates(graph)


# ---------------------------------------------------------------------------
# Paradigm 1: Transformer (DeBERTa ABSA)
# ---------------------------------------------------------------------------

class _TransformerABSA:
    """
    Uses yangheng/deberta-v3-base-absa-v1.1 for ABSC.

    Input format : "{sentence} [SEP] {aspect}"
    Label map    : 0 -> negative, 1 -> neutral, 2 -> positive
    """

    MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
    LABEL_MAP  = {0: "negative", 1: "neutral", 2: "positive"}

    def __init__(self) -> None:
        self._tokenizer = None
        self._model     = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        print_info(f"Loading {self.MODEL_NAME} …")
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model     = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self._model.eval()
        self._torch = torch

    def score(self, sentence: str, aspect: str) -> AspectOpinion:
        self._load()
        text = f"{sentence} [SEP] {aspect}"
        enc  = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        with self._torch.no_grad():
            logits = self._model(**enc).logits[0]
        probs = self._torch.softmax(logits, dim=-1).numpy()
        label_idx  = int(np.argmax(probs))
        confidence = float(probs[label_idx])
        return AspectOpinion(
            aspect=aspect,
            sentiment=self.LABEL_MAP[label_idx],
            confidence=confidence,
            opinion_words=[],
        )

    def run(
        self,
        sentences: list[dict[str, Any]],
        graph: nx.DiGraph,
        batch_size: int = 16,
    ) -> list[SentenceABSAResult]:
        self._load()
        aspects = _graph_aspect_candidates(graph)
        if not aspects:
            print_warning("No aspect candidates found in graph — skipping transformer ABSA.")
            return []

        results: list[SentenceABSAResult] = []
        total = len(sentences)
        for idx, rec in enumerate(sentences):
            if idx % 50 == 0:
                print_info(f"  Transformer: {idx}/{total} sentences …")
            sent = rec["sentence"]
            opinions: list[AspectOpinion] = []
            for asp in aspects:
                op = self.score(sent, asp)
                # Only keep non-neutral or high-confidence results to reduce noise
                if op.sentiment != "neutral" or op.confidence > 0.70:
                    opinions.append(op)
            results.append(SentenceABSAResult(
                sentence=sent,
                post_id=rec.get("post_id"),
                comment_id=rec.get("comment_id"),
                subreddit=rec.get("subreddit"),
                post_score=rec.get("post_score", 0) or 0,
                comment_score=rec.get("comment_score", 0) or 0,
                source=rec.get("source", ""),
                aspects=opinions,
                paradigm="transformer",
            ))
        return results


# ---------------------------------------------------------------------------
# Paradigm 2: LLM (Gemini Flash)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an Aspect-Based Sentiment Analysis (ABSA) expert.
Given a sentence and a list of aspect categories, extract the sentiment
expressed toward each aspect that is explicitly or implicitly mentioned.

Rules:
- Only include aspects that are actually discussed in the sentence.
- For each included aspect, provide: aspect name, sentiment (positive/neutral/negative),
  confidence (0.0-1.0), and the key opinion words from the text.
- Return ONLY a JSON array of objects with keys: aspect, sentiment, confidence, opinion_words.
- If no aspects are mentioned, return an empty array [].
- Do not include any text outside the JSON array.
"""

USER_TEMPLATE = """\
Aspects to consider: {aspects}

Sentences to analyse (one per line, prefixed with index):
{sentences}

Return a JSON array of arrays — one inner array per sentence, in the same order.
Each inner array contains aspect-opinion objects for that sentence.
"""


class _LLMABSA:
    """
    Gemini Flash zero-shot ABSA.

    Sends batches of 10 sentences in a single prompt, returns nested JSON.
    """

    def __init__(self) -> None:
        self._client = None

    def _load(self) -> None:
        if self._client is not None:
            return
        from google import genai
        from google.genai import types as genai_types
        self._genai       = genai
        self._genai_types = genai_types
        self._client      = genai.Client(api_key=settings.gemini_api_key)
        self._model_name  = settings.gemini_model

    def _call(self, aspect_list: list[str], batch: list[str]) -> list[list[dict]]:
        """Call Gemini for a batch of sentences. Returns list-of-lists."""
        aspects_str  = ", ".join(aspect_list)
        numbered     = "\n".join(f"{i+1}. {s}" for i, s in enumerate(batch))
        user_content = USER_TEMPLATE.format(aspects=aspects_str, sentences=numbered)

        for attempt in range(3):
            try:
                resp = self._client.models.generate_content(
                    model=self._model_name,
                    contents=user_content,
                    config=self._genai_types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.0,
                        max_output_tokens=4096,
                    ),
                )
                text = resp.text.strip()
                # Strip markdown code fences if present
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
                parsed = json.loads(text)
                if isinstance(parsed, list) and len(parsed) == len(batch):
                    return parsed
                # Gemini sometimes returns flat list for single sentence
                if len(batch) == 1 and isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    return [parsed]
                print_warning(f"LLM batch size mismatch ({len(parsed)} vs {len(batch)}); retrying…")
            except Exception as exc:
                print_warning(f"LLM attempt {attempt+1} failed: {exc}")
                time.sleep(2 ** attempt)
        # Fallback: empty results for this batch
        return [[] for _ in batch]

    def run(
        self,
        sentences: list[dict[str, Any]],
        graph: nx.DiGraph,
        sample: int = 300,
        batch_size: int = 10,
    ) -> list[SentenceABSAResult]:
        self._load()
        aspects = _graph_aspect_candidates(graph)
        if not aspects:
            print_warning("No aspect candidates — skipping LLM ABSA.")
            return []

        # Sample for cost control
        if len(sentences) > sample:
            rng = np.random.default_rng(42)
            idxs = rng.choice(len(sentences), size=sample, replace=False).tolist()
            subset = [sentences[i] for i in sorted(idxs)]
            print_info(f"  LLM ABSA: sampling {sample}/{len(sentences)} sentences for cost control.")
        else:
            subset = sentences

        results: list[SentenceABSAResult] = []
        total = len(subset)
        for start in range(0, total, batch_size):
            batch_recs = subset[start:start + batch_size]
            batch_sents = [r["sentence"] for r in batch_recs]
            if start % 50 == 0:
                print_info(f"  LLM: {start}/{total} sentences …")

            raw_batches = self._call(aspects, batch_sents)
            for rec, raw_opinions in zip(batch_recs, raw_batches):
                opinions: list[AspectOpinion] = []
                for op in raw_opinions:
                    if not isinstance(op, dict):
                        continue
                    sent_label = str(op.get("sentiment", "neutral")).lower()
                    if sent_label not in ("positive", "neutral", "negative"):
                        sent_label = "neutral"
                    opinions.append(AspectOpinion(
                        aspect=str(op.get("aspect", "")),
                        sentiment=sent_label,
                        confidence=float(op.get("confidence", 0.5)),
                        opinion_words=list(op.get("opinion_words", [])),
                    ))
                results.append(SentenceABSAResult(
                    sentence=rec["sentence"],
                    post_id=rec.get("post_id"),
                    comment_id=rec.get("comment_id"),
                    subreddit=rec.get("subreddit"),
                    post_score=rec.get("post_score", 0) or 0,
                    comment_score=rec.get("comment_score", 0) or 0,
                    source=rec.get("source", ""),
                    aspects=opinions,
                    paradigm="llm",
                ))
        return results


# ---------------------------------------------------------------------------
# Paradigm 3: Lexicon (VADER + spaCy)
# ---------------------------------------------------------------------------

class _LexiconABSA:
    """
    VADER compound sentiment + spaCy noun-chunk aspect detection.

    For each noun chunk in the sentence:
      - Check if it fuzzy-matches a taxonomy aspect (lowercase substring).
      - Score the chunk's local context window with VADER.
      - Extract opinion words via amod/advmod/neg dependency arcs.
    """

    def __init__(self) -> None:
        self._nlp     = None
        self._vader   = None

    def _load(self) -> None:
        if self._nlp is not None:
            return
        import spacy
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._vader = SentimentIntensityAnalyzer()
        self._nlp   = spacy.load(
            settings.spacy_model,
            exclude=["ner", "lemmatizer"],
        )

    @staticmethod
    def _compound_to_label(compound: float) -> tuple[str, float]:
        if compound >= 0.05:
            return "positive", min(1.0, (compound + 1.0) / 2.0)
        if compound <= -0.05:
            return "negative", min(1.0, (1.0 - compound) / 2.0)
        return "neutral", 1.0 - abs(compound) * 2

    def _opinion_words(self, token) -> list[str]:  # type: ignore[override]
        words: list[str] = []
        for child in token.children:
            if child.dep_ in ("amod", "advmod", "neg", "compound"):
                words.append(child.text.lower())
        return words

    def _score_span(self, sentence: str, chunk_text: str) -> tuple[str, float, list[str]]:
        """Score VADER on a short window around the aspect mention."""
        doc    = self._nlp(sentence)
        scores = self._vader.polarity_scores(sentence)
        compound = scores["compound"]
        label, conf = self._compound_to_label(compound)

        opinion: list[str] = []
        for token in doc:
            if token.text.lower() in chunk_text.lower():
                opinion.extend(self._opinion_words(token))
        return label, conf, opinion

    def run(
        self,
        sentences: list[dict[str, Any]],
        graph: nx.DiGraph,
    ) -> list[SentenceABSAResult]:
        self._load()
        aspects = _graph_aspect_candidates(graph)
        aspects_lower = {a.lower(): a for a in aspects}

        results: list[SentenceABSAResult] = []
        total = len(sentences)
        for idx, rec in enumerate(sentences):
            if idx % 100 == 0:
                print_info(f"  Lexicon: {idx}/{total} sentences …")
            sent = rec["sentence"]
            doc  = self._nlp(sent)

            matched: dict[str, AspectOpinion] = {}
            # Match taxonomy aspects via substring in noun chunks
            for chunk in doc.noun_chunks:
                chunk_low = chunk.text.lower()
                for asp_low, asp_orig in aspects_lower.items():
                    if asp_low in chunk_low or chunk_low in asp_low:
                        label, conf, opinion_words = self._score_span(sent, chunk.text)
                        if asp_orig not in matched:
                            matched[asp_orig] = AspectOpinion(
                                aspect=asp_orig,
                                sentiment=label,
                                confidence=conf,
                                opinion_words=opinion_words,
                            )

            # Also do a whole-sentence pass for direct keyword matches
            for asp_low, asp_orig in aspects_lower.items():
                if asp_low in sent.lower() and asp_orig not in matched:
                    scores   = self._vader.polarity_scores(sent)
                    label, conf = self._compound_to_label(scores["compound"])
                    matched[asp_orig] = AspectOpinion(
                        aspect=asp_orig,
                        sentiment=label,
                        confidence=conf,
                        opinion_words=[],
                    )

            results.append(SentenceABSAResult(
                sentence=sent,
                post_id=rec.get("post_id"),
                comment_id=rec.get("comment_id"),
                subreddit=rec.get("subreddit"),
                post_score=rec.get("post_score", 0) or 0,
                comment_score=rec.get("comment_score", 0) or 0,
                source=rec.get("source", ""),
                aspects=list(matched.values()),
                paradigm="lexicon",
            ))
        return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_absa(
    sentences: list[dict[str, Any]],
    aspect_graph: nx.DiGraph,
    out_dir: Path,
    force: bool = False,
    paradigms: list[str] | None = None,
    llm_sample: int = 300,
) -> dict[str, list[SentenceABSAResult]]:
    """
    Run multi-paradigm ABSA and cache results.

    Parameters
    ----------
    sentences     : preprocessed sentence records
    aspect_graph  : NetworkX DiGraph from build_aspect_graph()
    out_dir       : write <paradigm>_results.json here
    force         : re-run even if cached
    paradigms     : subset of ["transformer", "llm", "lexicon"]
    llm_sample    : max sentences sent to Gemini (cost control)

    Returns
    -------
    dict mapping paradigm name -> list[SentenceABSAResult]
    """
    if paradigms is None:
        paradigms = ["transformer", "llm", "lexicon"]

    out_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[SentenceABSAResult]] = {}

    runners: dict[str, Any] = {
        "transformer": _TransformerABSA(),
        "llm":         _LLMABSA(),
        "lexicon":     _LexiconABSA(),
    }

    for paradigm in paradigms:
        cache_file = out_dir / f"{paradigm}_results.json"

        if cache_file.exists() and not force:
            print_info(f"Loading cached {paradigm} ABSA results …")
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            loaded: list[SentenceABSAResult] = []
            for rec in raw:
                aspects = [AspectOpinion(**a) for a in rec.pop("aspects")]
                loaded.append(SentenceABSAResult(aspects=aspects, **rec))
            all_results[paradigm] = loaded
            print_success(f"  {paradigm}: {len(loaded)} sentences loaded from cache.")
            continue

        print_info(f"Running {paradigm} ABSA on {len(sentences)} sentences …")
        try:
            runner = runners[paradigm]
            if paradigm == "transformer":
                preds = runner.run(sentences, aspect_graph,
                                   batch_size=settings._yaml.get("absa", {}).get("batch_size", 16))
            elif paradigm == "llm":
                preds = runner.run(sentences, aspect_graph, sample=llm_sample)
            else:
                preds = runner.run(sentences, aspect_graph)
        except Exception as exc:
            print_warning(f"{paradigm} ABSA failed: {exc}. Skipping.")
            all_results[paradigm] = []
            continue

        # Persist
        cache_file.write_text(
            json.dumps([asdict(r) for r in preds], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_results[paradigm] = preds
        print_success(f"  {paradigm}: {len(preds)} sentences processed -> {cache_file.name}")

    return all_results
