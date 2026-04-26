"""
crisis_detection.py
-------------------
Binary crisis / non-crisis classifier that can run standalone or
be integrated with the emotion classifier pipeline.

Features
--------
* Rule-based crisis detector (fast, explainable, offline)
* ML-based crisis detector (Logistic Regression on TF-IDF)
* Integration layer: combine emotion scores + crisis flag
"""

import logging
import pickle
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR  = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "results" / "figures"
for p in [MODELS_DIR, FIGURES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ── Lexicon of crisis-related phrases ────────────────────────────────────────
CRISIS_PHRASES = [
    # Suicidal ideation
    "end my life", "kill myself", "take my life", "commit suicide",
    "want to die", "better off dead", "wish i was dead", "want to be dead",
    "planning to suicide", "plan to end it", "ending it all",
    "goodbye forever", "final goodbye", "written my note", "goodbye note",
    # Self-harm
    "cutting myself", "hurt myself", "harming myself", "self harm",
    "self-harm", "burning myself", "scratching myself",
    # Hopelessness (strong markers)
    "no reason to live", "don't want to live", "can't go on",
    "can't take it anymore", "nothing left to live for",
    "nobody would miss me", "no one would care if i died",
    "a burden to everyone", "burden to my family",
    # Crisis signals
    "stockpiling pills", "overdose", "swallowing pills to die",
    "hanging myself", "jump off",
]

CRISIS_WORDS = [
    "suicidal", "suicide", "self-harm", "self harm",
]


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based detector
# ─────────────────────────────────────────────────────────────────────────────

def rule_based_crisis(text: str) -> Tuple[int, float, List[str]]:
    """
    Fast keyword/phrase scan.

    Returns
    -------
    label     : 1 = crisis, 0 = non-crisis
    score     : rough severity score (0-1)
    triggers  : list of matched phrases/words
    """
    text_lower = text.lower()
    found = []

    # Phrase matching (weighted higher)
    phrase_hits = [p for p in CRISIS_PHRASES if p in text_lower]
    found.extend(phrase_hits)

    # Word matching
    word_hits = [w for w in CRISIS_WORDS if re.search(r'\b' + re.escape(w) + r'\b', text_lower)]
    found.extend(word_hits)

    # De-duplicate
    found = list(dict.fromkeys(found))

    label = 1 if found else 0
    # Score: clamp hits to [0,1]
    score = min(len(found) / 3.0, 1.0) if found else 0.0
    return label, score, found


# ─────────────────────────────────────────────────────────────────────────────
# ML-based detector
# ─────────────────────────────────────────────────────────────────────────────

class MLCrisisDetector:
    """
    Logistic Regression on TF-IDF features for binary crisis classification.
    """

    def __init__(self, C: float = 1.0, max_features: int = 10_000,
                 threshold: float = 0.4):
        self.C = C
        self.max_features = max_features
        self.threshold = threshold
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[LogisticRegression] = None

    def fit(self, texts: List[str], labels: np.ndarray) -> "MLCrisisDetector":
        """
        Train on texts (list of str) and labels (0/1 array).
        """
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(C=self.C, max_iter=1000,
                                         class_weight="balanced")
        self.model.fit(X, labels)
        logger.info("MLCrisisDetector trained on %d samples.", len(labels))
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        assert self.model is not None, "Call .fit() first."
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, texts: List[str]) -> np.ndarray:
        proba = self.predict_proba(texts)
        return (proba >= self.threshold).astype(int)

    def evaluate(self, texts: List[str], labels: np.ndarray) -> dict:
        preds = self.predict(texts)
        probas = self.predict_proba(texts)
        report = classification_report(labels, preds, target_names=["non-crisis", "crisis"],
                                        output_dict=True)
        results = {
            "f1_macro":    f1_score(labels, preds, average="macro"),
            "f1_crisis":   report["crisis"]["f1-score"],
            "roc_auc":     roc_auc_score(labels, probas),
            "report":      report,
            "preds":       preds,
            "probas":      probas,
        }
        logger.info("Crisis Detector  F1(crisis)=%.4f  ROC-AUC=%.4f",
                    results["f1_crisis"], results["roc_auc"])
        return results

    def save(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = MODELS_DIR / "crisis_detector.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved crisis detector → %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "MLCrisisDetector":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Loaded crisis detector ← %s", path)
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Integrated pipeline: emotion + crisis
# ─────────────────────────────────────────────────────────────────────────────

def integrated_analysis(
    text: str,
    emotion_labels: List[str],          # emotion labels predicted by main model
    crisis_detector: Optional[MLCrisisDetector] = None,
) -> dict:
    """
    Combine emotion output with crisis detection into a single result dict.

    Parameters
    ----------
    text            : raw input text
    emotion_labels  : list of emotion strings predicted by emotion model
    crisis_detector : optional trained MLCrisisDetector

    Returns
    -------
    dict with emotion labels, crisis flag, severity, triggers
    """
    # Rule-based (always run)
    rule_label, rule_score, triggers = rule_based_crisis(text)

    # ML-based (if detector provided)
    ml_label, ml_score = None, None
    if crisis_detector is not None:
        ml_score = float(crisis_detector.predict_proba([text])[0])
        ml_label = int(ml_score >= crisis_detector.threshold)

    # Final decision: OR of rule + ML (conservative approach)
    crisis_flag = rule_label
    if ml_label is not None:
        crisis_flag = max(crisis_flag, ml_label)

    # Escalate if negative emotions dominate
    negative_emotions = {"sadness", "grief", "fear", "anger", "despair", "remorse"}
    neg_count = sum(1 for e in emotion_labels if e in negative_emotions)
    if neg_count >= 2 and rule_score > 0:
        crisis_flag = 1

    return {
        "text":          text,
        "emotions":      emotion_labels,
        "crisis":        bool(crisis_flag),
        "rule_score":    rule_score,
        "ml_score":      ml_score,
        "triggers":      triggers,
        "recommendation": _get_recommendation(crisis_flag, rule_score),
    }


def _get_recommendation(crisis_flag: int, severity: float) -> str:
    if crisis_flag and severity >= 0.6:
        return ("⚠️ HIGH RISK: Immediate crisis intervention recommended. "
                "Refer to: National Suicide Prevention Lifeline (988) or text HOME to 741741.")
    elif crisis_flag:
        return ("⚠️ MODERATE RISK: Emotional support and follow-up recommended. "
                "Consider reaching out to a mental health professional.")
    else:
        return "✅ No immediate crisis indicators detected."


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_crisis_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                                  save: bool = True) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax,
                xticklabels=["Non-Crisis", "Crisis"],
                yticklabels=["Non-Crisis", "Crisis"])
    ax.set_title("Crisis Detection Confusion Matrix")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "crisis_confusion_matrix.png"
        fig.savefig(path, dpi=150)
        logger.info("Saved crisis confusion matrix → %s", path)
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Rule-based
    texts = [
        "I want to end my life, I can't take it anymore.",
        "I'm feeling a bit sad today but I'll be okay.",
        "I've been cutting myself to deal with the pain.",
    ]
    for t in texts:
        label, score, triggers = rule_based_crisis(t)
        print(f"Crisis={label}  Score={score:.2f}  Triggers={triggers}")
        print(f"  → {t[:70]}\n")

    # ML detector (tiny dataset for smoke-test)
    from src.dataset_loader import load_crisis_dataset, split_crisis
    df = load_crisis_dataset()
    train, test = split_crisis(df)

    detector = MLCrisisDetector()
    detector.fit(train["text"].tolist(), train["crisis"].values)
    results = detector.evaluate(test["text"].tolist(), test["crisis"].values)
    print(f"\nML Detector  F1_crisis={results['f1_crisis']:.4f}  "
          f"ROC_AUC={results['roc_auc']:.4f}")
