"""
utils/helpers.py
----------------
General-purpose utilities: seed setting, logging helpers,
emotion → group mapping, and inference helpers.
"""

import os
import random
import logging
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Ensure reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.debug("Random seed set to %d", seed)


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    return device


# ── Emotion hierarchy utilities ───────────────────────────────────────────────

EMOTION_HIERARCHY = {
    "positive": [
        "admiration", "amusement", "approval", "caring", "curiosity",
        "desire", "excitement", "gratitude", "joy", "love", "optimism",
        "pride", "realization", "relief",
    ],
    "negative": [
        "anger", "annoyance", "disappointment", "disapproval", "disgust",
        "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness",
    ],
    "ambiguous": ["confusion", "surprise"],
    "neutral":   ["neutral"],
}
EMOTION_TO_GROUP = {
    e: g
    for g, emotions in EMOTION_HIERARCHY.items()
    for e in emotions
}


def labels_to_groups(label_list: List[str]) -> List[str]:
    """Convert a list of emotion labels to their top-level groups (deduplicated)."""
    groups = list(dict.fromkeys(EMOTION_TO_GROUP.get(l, "unknown") for l in label_list))
    return groups


def dominant_group(label_list: List[str]) -> str:
    """Return the most frequent group, or 'neutral' if empty."""
    groups = [EMOTION_TO_GROUP.get(l, "neutral") for l in label_list]
    if not groups:
        return "neutral"
    from collections import Counter
    return Counter(groups).most_common(1)[0][0]


# ── Inference helper ──────────────────────────────────────────────────────────

def predict_emotions_bert(
    text: str,
    model,
    tokenizer,
    label_names: List[str],
    max_len: int = 128,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Run a single text through a BERT model and return a results dict.

    Parameters
    ----------
    text        : raw input string
    model       : BERTEmotionClassifier (on eval mode, loaded)
    tokenizer   : HuggingFace tokenizer
    label_names : ordered list of emotion labels
    threshold   : probability cut-off for positive prediction

    Returns
    -------
    dict with 'labels', 'probs', 'groups'
    """
    if device is None:
        device = get_device()

    model.eval()
    enc = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    predicted_labels = [
        label_names[i] for i, p in enumerate(probs) if p >= threshold
    ]
    if not predicted_labels:
        predicted_labels = ["neutral"]

    label_prob_pairs = sorted(
        [(label_names[i], float(probs[i])) for i in range(len(label_names))],
        key=lambda x: -x[1],
    )

    return {
        "text":       text,
        "labels":     predicted_labels,
        "top_probs":  label_prob_pairs[:5],
        "groups":     labels_to_groups(predicted_labels),
        "dominant":   dominant_group(predicted_labels),
    }


# ── Text truncation for display ───────────────────────────────────────────────

def truncate(text: str, max_chars: int = 80) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "…"


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    """Load a JSON config file into a dict."""
    import json
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    set_seed(42)
    print(labels_to_groups(["joy", "sadness", "surprise"]))
    print(dominant_group(["joy", "love", "sadness"]))
