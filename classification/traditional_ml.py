"""
models/traditional_ml.py
------------------------
Logistic Regression and Linear SVM with multi-label classification
using scikit-learn's OneVsRestClassifier wrapper.
"""

import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Logistic Regression (multi-label)
# ─────────────────────────────────────────────────────────────────────────────

class MultiLabelLogisticRegression:
    """
    Wraps sklearn OneVsRestClassifier + LogisticRegression.
    Works with TF-IDF sparse matrices.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 threshold: float = 0.3):
        self.C = C
        self.max_iter = max_iter
        self.threshold = threshold          # probability cut-off for positive label
        self.model: Optional[OneVsRestClassifier] = None
        self.label_names: Optional[list] = None

    def build(self, label_names: list) -> "MultiLabelLogisticRegression":
        self.label_names = label_names
        base = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="saga",
            class_weight="balanced",
        )
        self.model = OneVsRestClassifier(base, n_jobs=-1)
        logger.info(
            "Built MultiLabelLogisticRegression  C=%.2f  max_iter=%d  "
            "labels=%d", self.C, self.max_iter, len(label_names)
        )
        return self

    def fit(self, X_train, y_train: np.ndarray) -> "MultiLabelLogisticRegression":
        assert self.model is not None, "Call .build() first."
        logger.info("Training Logistic Regression …")
        self.model.fit(X_train, y_train)
        logger.info("  Done.")
        return self

    def predict(self, X) -> np.ndarray:
        """Return binary predictions using the probability threshold."""
        proba = self.model.predict_proba(X)
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = MODELS_DIR / "logreg_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved LogReg model → %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "MultiLabelLogisticRegression":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Loaded LogReg model ← %s", path)
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Linear SVM (multi-label)
# ─────────────────────────────────────────────────────────────────────────────

class MultiLabelSVM:
    """
    Multi-label SVM using OneVsRest + CalibratedClassifierCV (for proba).
    LinearSVC is very fast on sparse TF-IDF features.
    """

    def __init__(self, C: float = 1.0, threshold: float = 0.3,
                 max_iter: int = 2000):
        self.C = C
        self.threshold = threshold
        self.max_iter = max_iter
        self.model: Optional[OneVsRestClassifier] = None
        self.label_names: Optional[list] = None

    def build(self, label_names: list) -> "MultiLabelSVM":
        self.label_names = label_names
        base = CalibratedClassifierCV(
            LinearSVC(C=self.C, max_iter=self.max_iter, class_weight="balanced"),
            cv=3,
        )
        self.model = OneVsRestClassifier(base, n_jobs=-1)
        logger.info(
            "Built MultiLabelSVM  C=%.2f  max_iter=%d  labels=%d",
            self.C, self.max_iter, len(label_names)
        )
        return self

    def fit(self, X_train, y_train: np.ndarray) -> "MultiLabelSVM":
        assert self.model is not None, "Call .build() first."
        logger.info("Training SVM …")
        self.model.fit(X_train, y_train)
        logger.info("  Done.")
        return self

    def predict(self, X) -> np.ndarray:
        proba = self.model.predict_proba(X)
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = MODELS_DIR / "svm_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved SVM model → %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "MultiLabelSVM":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Loaded SVM model ← %s", path)
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scipy.sparse import csr_matrix

    rng = np.random.default_rng(0)
    X = csr_matrix(rng.random((200, 500)).astype(np.float32))
    y = (rng.random((200, 5)) > 0.7).astype(int)
    labels = ["joy", "sadness", "anger", "fear", "love"]

    lr = MultiLabelLogisticRegression().build(labels)
    lr.fit(X[:150], y[:150])
    preds = lr.predict(X[150:])
    print("LR preds shape:", preds.shape)

    svm = MultiLabelSVM().build(labels)
    svm.fit(X[:150], y[:150])
    preds = svm.predict(X[150:])
    print("SVM preds shape:", preds.shape)
