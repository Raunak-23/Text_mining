"""
models/llm_classifier.py
------------------------
LLM-based multi-label emotion classifier.

Two modes:
  1. SIMULATION mode – uses a hand-crafted keyword/rule system to simulate
     what an LLM would return. Fully offline, no API key required.
  2. API mode – calls the Anthropic Messages API (claude-3-haiku) for real
     classifications. Set ANTHROPIC_API_KEY in your environment to enable.

Both modes return a list of predicted emotion labels per text.
"""

import os
import re
import json
import logging
import time
import numpy as np
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── 28 GoEmotions labels ─────────────────────────────────────────────────────
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]
LABEL_SET = set(GOEMOTIONS_LABELS)

# ── Keyword rules for simulation mode ────────────────────────────────────────
KEYWORD_RULES = {
    "joy":            ["happy", "joy", "delighted", "wonderful", "amazing", "great",
                       "fantastic", "love it", "awesome", "ecstatic", "thrilled"],
    "sadness":        ["sad", "depressed", "unhappy", "miserable", "cry", "tears",
                       "heartbroken", "sorrowful", "gloomy", "grief"],
    "anger":          ["angry", "furious", "outraged", "rage", "hate", "mad",
                       "infuriated", "livid", "disgusted by"],
    "fear":           ["scared", "afraid", "terrified", "frightened", "anxious",
                       "nervous", "dread", "horror", "panic"],
    "love":           ["love", "adore", "cherish", "affection", "romantic"],
    "surprise":       ["surprised", "shocked", "unexpected", "wow", "unbelievable",
                       "astonishing", "amazed", "stunned"],
    "disgust":        ["disgusting", "gross", "revolting", "horrible", "awful",
                       "repulsed", "sick to my stomach"],
    "gratitude":      ["thank", "grateful", "thankful", "appreciate", "blessing"],
    "excitement":     ["excited", "thrilled", "pumped", "stoked", "can't wait",
                       "exhilarating"],
    "disappointment": ["disappointed", "let down", "expected better", "underwhelming",
                       "let me down"],
    "confusion":      ["confused", "don't understand", "puzzled", "unclear",
                       "perplexed", "what do you mean", "lost"],
    "curiosity":      ["curious", "wonder", "interesting", "tell me more",
                       "how does", "why does", "want to know"],
    "annoyance":      ["annoying", "irritating", "frustrating", "ugh", "bother",
                       "sick of"],
    "optimism":       ["hope", "optimistic", "better days", "bright future",
                       "looking forward", "positive"],
    "caring":         ["care", "concern", "worried about you", "hope you're okay",
                       "support you", "here for you"],
    "admiration":     ["admire", "impressive", "inspiring", "look up to",
                       "respect", "brilliant"],
    "approval":       ["approve", "good job", "well done", "agree", "right",
                       "exactly", "absolutely"],
    "disapproval":    ["disapprove", "wrong", "shouldn't", "that's bad",
                       "disagree", "not okay"],
    "grief":          ["mourning", "loss", "passed away", "died", "bereaved",
                       "devastated by loss"],
    "remorse":        ["sorry", "regret", "apologize", "should have", "my fault",
                       "guilty"],
    "embarrassment":  ["embarrassed", "ashamed", "mortified", "awkward",
                       "humiliated"],
    "pride":          ["proud", "accomplished", "achievement", "look what i did",
                       "nailed it"],
    "relief":         ["relieved", "phew", "glad it's over", "finally", "safe now"],
    "nervousness":    ["nervous", "jittery", "on edge", "butterflies", "uneasy"],
    "realization":    ["realized", "just understood", "oh wait", "it hit me",
                       "now i see"],
    "desire":         ["want", "wish", "crave", "longing", "need", "desire"],
    "amusement":      ["funny", "lol", "haha", "hilarious", "laughing",
                       "cracking up"],
    "neutral":        [],  # fallback
}


# ─────────────────────────────────────────────────────────────────────────────
# Simulation classifier
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_classify(text: str, top_k: int = 3) -> List[str]:
    """
    Rule-based simulation of LLM emotion labelling.
    Returns up to top_k emotion labels.
    """
    text_lower = text.lower()
    scores = {}
    for emotion, keywords in KEYWORD_RULES.items():
        if not keywords:
            continue
        count = sum(1 for kw in keywords if kw in text_lower)
        if count:
            scores[emotion] = count

    if not scores:
        return ["neutral"]

    sorted_emotions = sorted(scores, key=scores.get, reverse=True)
    return sorted_emotions[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# API classifier (requires ANTHROPIC_API_KEY)
# ─────────────────────────────────────────────────────────────────────────────

_API_PROMPT_TEMPLATE = """You are an expert emotion analysis system.
Analyse the following text and return ONLY a JSON array of emotion labels present.
Choose from this exact set:
{label_set}

Rules:
- Return ONLY a JSON array, e.g. ["joy", "admiration"]
- Include ALL emotions that are clearly present
- Use "neutral" if no strong emotion is detected
- Return at most 4 labels
- Do not explain your reasoning

Text: {text}
"""


def _api_classify(text: str, api_key: str, model: str = "claude-haiku-4-5-20251001",
                  max_retries: int = 3) -> List[str]:
    """
    Classify a single text via the Anthropic API.
    Falls back to simulation on failure.
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed; falling back to simulation.")
        return _simulate_classify(text)

    client = anthropic.Anthropic(api_key=api_key)
    prompt = _API_PROMPT_TEMPLATE.format(
        label_set=", ".join(GOEMOTIONS_LABELS),
        text=text,
    )

    for attempt in range(1, max_retries + 1):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=128,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            # Extract JSON array
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if match:
                labels = json.loads(match.group())
                valid = [l for l in labels if l in LABEL_SET]
                return valid if valid else ["neutral"]
        except Exception as e:
            logger.warning("API attempt %d failed: %s", attempt, e)
            time.sleep(2 ** attempt)

    logger.warning("All API attempts failed; using simulation fallback.")
    return _simulate_classify(text)


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

class LLMClassifier:
    """
    Unified interface for both simulation and API-based classification.

    Parameters
    ----------
    use_api : bool
        If True and ANTHROPIC_API_KEY is set, uses the Anthropic API.
        Otherwise falls back to simulation.
    label_names : list[str]
        Full list of emotion labels (used for binarisation).
    """

    def __init__(self, use_api: bool = False, label_names: List[str] = None):
        self.use_api = use_api
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.label_names = label_names or GOEMOTIONS_LABELS
        self.label_to_idx = {l: i for i, l in enumerate(self.label_names)}

        if use_api and self.api_key:
            logger.info("LLMClassifier: API mode (Anthropic)")
        else:
            logger.info("LLMClassifier: Simulation mode (keyword rules)")

    def predict_labels(self, texts: List[str], batch_size: int = 16,
                       delay: float = 0.1) -> List[List[str]]:
        """
        Classify a list of texts.

        Returns
        -------
        List of lists of emotion label strings.
        """
        results = []
        for i, text in enumerate(texts):
            if self.use_api and self.api_key:
                labels = _api_classify(text, self.api_key)
                if delay > 0:
                    time.sleep(delay)
            else:
                labels = _simulate_classify(text)
            results.append(labels)

            if (i + 1) % 50 == 0:
                logger.info("  Classified %d / %d", i + 1, len(texts))
        return results

    def predict_matrix(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Classify texts and return a binary matrix of shape (N, num_labels).
        """
        label_lists = self.predict_labels(texts, **kwargs)
        matrix = np.zeros((len(texts), len(self.label_names)), dtype=np.int8)
        for i, labels in enumerate(label_lists):
            for lbl in labels:
                if lbl in self.label_to_idx:
                    matrix[i, self.label_to_idx[lbl]] = 1
        return matrix


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    clf = LLMClassifier(use_api=False)
    texts = [
        "I am so grateful and happy today!",
        "I can't believe how angry and disgusted I feel.",
        "The weather is fine.",
    ]
    labels = clf.predict_labels(texts)
    matrix = clf.predict_matrix(texts)
    for t, l in zip(texts, labels):
        print(f"  [{', '.join(l)}] → {t[:60]}")
    print("Matrix shape:", matrix.shape)
