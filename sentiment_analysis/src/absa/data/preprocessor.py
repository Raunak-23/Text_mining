"""
Text preprocessor.

Takes raw JSON files produced by collector.py and outputs a flat list of
cleaned, sentence-level records ready for topic modeling and ABSA.

Output schema (one JSON file per product):
[
  {
    "sentence":    "The camera is absolutely stunning in low light.",
    "post_id":     "abc123",
    "comment_id":  "xyz789",   # None for post selftext
    "subreddit":   "apple",
    "post_score":  412,
    "comment_score": 28,       # None for post selftext
    "depth":       0,
    "source":      "comment"   # "title" | "selftext" | "comment"
  },
  ...
]
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import spacy
from spacy.language import Language

from absa.utils.config import settings
from absa.utils.display import print_info, print_success, print_warning

# ---------------------------------------------------------------------------
# spaCy loader (singleton)
# ---------------------------------------------------------------------------

_nlp: Language | None = None

def _get_nlp() -> Language:
    global _nlp
    if _nlp is None:
        model = settings.spacy_model
        try:
            _nlp = spacy.load(model, disable=["ner", "lemmatizer", "attribute_ruler"])
        except OSError:
            raise OSError(
                f"spaCy model '{model}' not found.\n"
                f"Run:  uv run python -m spacy download {model}"
            )
    return _nlp


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

# Reddit markdown / URL patterns
_URL_RE       = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE   = re.compile(r"u/\w+|r/\w+")
_MARKDOWN_RE  = re.compile(r"[*_~`>#|\\]")
_MULTI_WS_RE  = re.compile(r"\s+")
_EDIT_RE      = re.compile(r"\b(edit|update|edt)\s*\d*\s*:", re.IGNORECASE)
_EMOJI_RE     = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

def _clean(text: str) -> str:
    """Apply Reddit-specific cleaning to a raw text string."""
    if not text:
        return ""
    # Normalize unicode (NFKC handles ligatures, curly quotes, etc.)
    text = unicodedata.normalize("NFKC", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _EDIT_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = _MARKDOWN_RE.sub(" ", text)
    # Replace newlines/tabs with spaces
    text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


def _is_valid_sentence(sent_text: str, min_tok: int, max_tok: int) -> bool:
    """Rough token count check — avoids loading full spaCy for filtering."""
    tokens = sent_text.split()
    return min_tok <= len(tokens) <= max_tok


# ---------------------------------------------------------------------------
# Comment tree flattener
# ---------------------------------------------------------------------------

def _flatten_comments(
    comments: list[dict[str, Any]],
    post_id: str,
    post_score: int,
    subreddit: str,
) -> list[dict[str, Any]]:
    """Depth-first traversal of the comment tree → flat list of records."""
    records: list[dict[str, Any]] = []
    stack = list(comments)
    while stack:
        node = stack.pop()
        records.append(
            {
                "_raw_text":     node["body"],
                "post_id":       post_id,
                "comment_id":    node["id"],
                "subreddit":     subreddit,
                "post_score":    post_score,
                "comment_score": node["score"],
                "depth":         node["depth"],
                "source":        "comment",
            }
        )
        # Push replies so they get processed too
        stack.extend(node.get("replies", []))
    return records


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

def _split_into_sentences(
    records: list[dict[str, Any]],
    min_tok: int,
    max_tok: int,
) -> list[dict[str, Any]]:
    """
    Replace each record's `_raw_text` with individual sentence records.
    Uses spaCy sentencizer; falls back to period-split if model missing.
    """
    nlp = _get_nlp()
    sentences: list[dict[str, Any]] = []

    texts = [r["_raw_text"] for r in records]
    # Process in batch for efficiency; disable heavy pipes we don't need
    for doc, record in zip(nlp.pipe(texts, batch_size=64), records):
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if _is_valid_sentence(sent_text, min_tok, max_tok):
                row = {k: v for k, v in record.items() if k != "_raw_text"}
                row["sentence"] = sent_text
                sentences.append(row)
    return sentences


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(
    raw_paths: list[Path],
    out_dir: Path | None = None,
    slug: str = "product",
    force: bool = False,
) -> Path:
    """
    Preprocess raw JSON files into a single flat sentence-level JSON.

    Parameters
    ----------
    raw_paths : list of Path
        Paths to <subreddit>.json files from the collector.
    out_dir : Path, optional
        Directory to write the output file. Defaults to settings.processed_dir/<slug>.
    slug : str
        Product slug used for the output filename.
    force : bool
        Re-process even if the output already exists.

    Returns
    -------
    Path
        Path to the written sentences JSON file.
    """
    out_dir = out_dir or (settings.processed_dir / slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "sentences.json"

    if out_file.exists() and not force:
        print_info(f"Preprocessed data already exists → {out_file}")
        return out_file

    min_tok = settings.min_sentence_tokens
    max_tok = settings.max_sentence_tokens

    all_records: list[dict[str, Any]] = []

    for path in raw_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        subreddit = data["subreddit"]
        posts     = data["posts"]
        print_info(f"Processing r/{subreddit} — {len(posts)} posts…")

        for post in posts:
            post_id    = post["id"]
            post_score = post["score"]

            # --- Post title ---
            title_clean = _clean(post.get("title", ""))
            if title_clean and _is_valid_sentence(title_clean, min_tok, max_tok):
                all_records.append(
                    {
                        "_raw_text":     title_clean,
                        "post_id":       post_id,
                        "comment_id":    None,
                        "subreddit":     subreddit,
                        "post_score":    post_score,
                        "comment_score": None,
                        "depth":         -1,
                        "source":        "title",
                    }
                )

            # --- Post body (selftext) ---
            selftext_clean = _clean(post.get("selftext", ""))
            if selftext_clean:
                all_records.append(
                    {
                        "_raw_text":     selftext_clean,
                        "post_id":       post_id,
                        "comment_id":    None,
                        "subreddit":     subreddit,
                        "post_score":    post_score,
                        "comment_score": None,
                        "depth":         -1,
                        "source":        "selftext",
                    }
                )

            # --- Comments tree ---
            comment_records = _flatten_comments(
                post.get("comments", []), post_id, post_score, subreddit
            )
            for rec in comment_records:
                rec["_raw_text"] = _clean(rec["_raw_text"])
                if rec["_raw_text"]:
                    all_records.append(rec)

    if not all_records:
        print_warning("No records after cleaning — check raw data.")
        return out_file

    # Sentence splitting via spaCy
    print_info(f"Splitting {len(all_records)} text chunks into sentences…")
    sentences = _split_into_sentences(all_records, min_tok, max_tok)
    print_success(f"{len(sentences)} sentences extracted from {len(all_records)} text chunks.")

    out_file.write_text(
        json.dumps(sentences, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_success(f"Saved → {out_file.relative_to(settings.processed_dir.parent)}")
    return out_file
