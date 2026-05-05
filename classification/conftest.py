"""
tests/conftest.py
-----------------
Shared pytest fixtures for the EmotiSense test suite.
All fixtures use tiny synthetic data so tests run in seconds without
downloading datasets or requiring a GPU.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix

# Ensure project root is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# ── Label constants ────────────────────────────────────────────────────────────
LABEL_NAMES = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]
N_LABELS = len(LABEL_NAMES)


# ── Tiny text corpus ───────────────────────────────────────────────────────────
SAMPLE_TEXTS = [
    "I am so happy and grateful today!",
    "This makes me feel angry and disgusted.",
    "I'm confused about what's happening here.",
    "Wow, that's really surprising and amazing!",
    "I feel sad and disappointed after hearing the news.",
    "So excited about the upcoming trip!",
    "I don't understand this at all.",
    "Thank you so much, I really appreciate it.",
    "This is outrageous and completely wrong.",
    "I feel a bit nervous but also curious.",
    "Just another ordinary day, nothing special.",
    "I love spending time with my family.",
    "I'm terrified of what might happen next.",
    "Really proud of what we accomplished together.",
    "I regret not speaking up when I had the chance.",
]

CRISIS_TEXTS = [
    "I want to end my life, I can't go on anymore.",
    "Nobody would miss me if I was gone.",
    "I've been cutting myself to cope with the pain.",
    "I feel so hopeless, there's no point in living.",
    "I'm saying goodbye to everyone today.",
]

NON_CRISIS_TEXTS = [
    "I'm feeling a bit sad today but I'll be okay.",
    "Work is really stressful this week.",
    "I broke up with my partner and I'm upset.",
    "I failed the exam and I'm disappointed.",
    "Today was a tough day but tomorrow will be better.",
]


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def label_names():
    return LABEL_NAMES


@pytest.fixture(scope="session")
def sample_texts():
    return SAMPLE_TEXTS


@pytest.fixture(scope="session")
def tiny_goemotions_df():
    """Minimal GoEmotions-format DataFrame (15 rows)."""
    rng = np.random.default_rng(42)
    label_matrix = (rng.random((len(SAMPLE_TEXTS), N_LABELS)) > 0.85).astype(int)
    # Ensure at least one label per row (add neutral if none)
    for i in range(len(SAMPLE_TEXTS)):
        if label_matrix[i].sum() == 0:
            label_matrix[i, LABEL_NAMES.index("neutral")] = 1
    df = pd.DataFrame(label_matrix, columns=LABEL_NAMES)
    df.insert(0, "text", SAMPLE_TEXTS)
    return df


@pytest.fixture(scope="session")
def tiny_splits(tiny_goemotions_df):
    """Train/val/test split of the tiny GoEmotions DataFrame."""
    n = len(tiny_goemotions_df)
    train = tiny_goemotions_df.iloc[:9].reset_index(drop=True)
    val   = tiny_goemotions_df.iloc[9:12].reset_index(drop=True)
    test  = tiny_goemotions_df.iloc[12:].reset_index(drop=True)
    return {"train": train, "validation": val, "test": test}


@pytest.fixture(scope="session")
def tiny_label_matrix(tiny_goemotions_df):
    """Binary label matrix for the tiny dataset."""
    return tiny_goemotions_df[LABEL_NAMES].values.astype(np.float32)


@pytest.fixture(scope="session")
def tiny_tfidf_features(tiny_splits):
    """Pre-built TF-IDF sparse matrices for all splits."""
    from src.data_preprocessing import build_tfidf_features, extract_label_matrix
    X_tr, X_v, X_te, vec = build_tfidf_features(
        tiny_splits["train"]["text"],
        tiny_splits["validation"]["text"],
        tiny_splits["test"]["text"],
        max_features=500,
    )
    y_tr = extract_label_matrix(tiny_splits["train"],      LABEL_NAMES)
    y_v  = extract_label_matrix(tiny_splits["validation"], LABEL_NAMES)
    y_te = extract_label_matrix(tiny_splits["test"],       LABEL_NAMES)
    return {"X_train": X_tr, "X_val": X_v, "X_test": X_te,
            "y_train": y_tr, "y_val":  y_v,  "y_test":  y_te,
            "vectorizer": vec}


@pytest.fixture(scope="session")
def tiny_crisis_df():
    """Minimal crisis/non-crisis DataFrame."""
    rows = (
        [{"text": t, "crisis": 1} for t in CRISIS_TEXTS] +
        [{"text": t, "crisis": 0} for t in NON_CRISIS_TEXTS]
    )
    return pd.DataFrame(rows)
