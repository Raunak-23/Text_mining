"""
tests/test_evaluation.py
------------------------
Unit tests for evaluation.py — metrics computation and visualisation helpers.
"""

import numpy as np
import pytest
import pandas as pd

from src.evaluation import (
    compute_metrics,
    build_comparison_table,
    plot_label_distribution,
    plot_performance_comparison,
)


@pytest.fixture
def binary_perfect():
    """y_true == y_pred → perfect scores."""
    rng = np.random.default_rng(0)
    y = (rng.random((50, 10)) > 0.7).astype(int)
    return y, y.copy()


@pytest.fixture
def binary_random():
    """Random predictions — low but non-zero scores."""
    rng = np.random.default_rng(1)
    y_true = (rng.random((80, 10)) > 0.7).astype(int)
    y_pred = (rng.random((80, 10)) > 0.7).astype(int)
    return y_true, y_pred


class TestComputeMetrics:
    def test_perfect_scores(self, binary_perfect):
        y_true, y_pred = binary_perfect
        m = compute_metrics(y_true, y_pred)
        assert m["f1_micro"]    == pytest.approx(1.0, abs=1e-6)
        assert m["hamming_loss"] == pytest.approx(0.0, abs=1e-6)
        assert m["subset_accuracy"] == pytest.approx(1.0, abs=1e-6)

    def test_all_keys_present(self, binary_random):
        y_true, y_pred = binary_random
        m = compute_metrics(y_true, y_pred)
        expected_keys = [
            "f1_micro", "f1_macro", "f1_weighted", "f1_samples",
            "precision_micro", "precision_macro",
            "recall_micro", "recall_macro",
            "subset_accuracy", "hamming_loss",
        ]
        for k in expected_keys:
            assert k in m, f"Missing key: {k}"

    def test_scores_in_range(self, binary_random):
        y_true, y_pred = binary_random
        m = compute_metrics(y_true, y_pred)
        for k in ["f1_micro", "f1_macro", "precision_micro", "recall_micro"]:
            assert 0.0 <= m[k] <= 1.0, f"{k} out of range: {m[k]}"

    def test_hamming_loss_in_range(self, binary_random):
        y_true, y_pred = binary_random
        m = compute_metrics(y_true, y_pred)
        assert 0.0 <= m["hamming_loss"] <= 1.0

    def test_per_label_returned_when_names_given(self, binary_random):
        y_true, y_pred = binary_random
        names = [f"emotion_{i}" for i in range(y_true.shape[1])]
        m = compute_metrics(y_true, y_pred, label_names=names)
        assert "per_label" in m
        assert isinstance(m["per_label"], pd.DataFrame)
        assert len(m["per_label"]) == len(names)

    def test_per_label_not_returned_without_names(self, binary_random):
        y_true, y_pred = binary_random
        m = compute_metrics(y_true, y_pred)
        assert "per_label" not in m

    def test_per_label_columns(self, binary_random):
        y_true, y_pred = binary_random
        names = [f"e_{i}" for i in range(y_true.shape[1])]
        m = compute_metrics(y_true, y_pred, label_names=names)
        for col in ["f1", "precision", "recall", "support"]:
            assert col in m["per_label"].columns

    def test_shape_mismatch_raises(self):
        y_true = np.ones((10, 5))
        y_pred = np.ones((10, 6))   # wrong shape
        with pytest.raises(AssertionError):
            compute_metrics(y_true, y_pred)

    def test_all_zeros_prediction(self):
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_pred = np.zeros_like(y_true)
        m = compute_metrics(y_true, y_pred)
        # Precision is undefined (zero), recall is 0
        assert m["recall_micro"] == pytest.approx(0.0)

    def test_all_ones_prediction(self):
        y_true = np.array([[1, 0], [0, 1], [1, 1]])
        y_pred = np.ones_like(y_true)
        m = compute_metrics(y_true, y_pred)
        assert m["precision_micro"] < 1.0   # some FPs


class TestBuildComparisonTable:
    def test_returns_dataframe(self):
        results = {
            "LogReg": {"f1_micro": 0.4, "f1_macro": 0.3, "f1_weighted": 0.38,
                       "precision_micro": 0.45, "recall_micro": 0.36,
                       "subset_accuracy": 0.2, "hamming_loss": 0.07},
            "BERT":   {"f1_micro": 0.6, "f1_macro": 0.5, "f1_weighted": 0.58,
                       "precision_micro": 0.65, "recall_micro": 0.56,
                       "subset_accuracy": 0.3, "hamming_loss": 0.04},
        }
        table = build_comparison_table(results)
        assert isinstance(table, pd.DataFrame)
        assert "LogReg" in table.index
        assert "BERT"   in table.index

    def test_all_metric_columns_present(self):
        results = {
            "Model": {"f1_micro": 0.5, "f1_macro": 0.4, "f1_weighted": 0.48,
                      "precision_micro": 0.55, "recall_micro": 0.46,
                      "subset_accuracy": 0.25, "hamming_loss": 0.05},
        }
        table = build_comparison_table(results)
        expected_cols = ["F1 (micro)", "F1 (macro)", "F1 (weighted)",
                         "Precision (micro)", "Recall (micro)",
                         "Subset Acc.", "Hamming Loss"]
        for col in expected_cols:
            assert col in table.columns


class TestVisualisations:
    """Smoke-tests: just check they run without raising exceptions."""

    def test_plot_label_distribution(self):
        rng = np.random.default_rng(42)
        y = (rng.random((100, 10)) > 0.8).astype(int)
        names = [f"emotion_{i}" for i in range(10)]
        # Should not raise; save=False avoids filesystem dependency
        plot_label_distribution(y, names, save=False)

    def test_plot_performance_comparison(self):
        results = {
            "LogReg": {"f1_micro": 0.4, "f1_macro": 0.3, "f1_weighted": 0.38,
                       "precision_micro": 0.45, "recall_micro": 0.36,
                       "subset_accuracy": 0.2, "hamming_loss": 0.07},
            "BERT":   {"f1_micro": 0.6, "f1_macro": 0.5, "f1_weighted": 0.58,
                       "precision_micro": 0.65, "recall_micro": 0.56,
                       "subset_accuracy": 0.3, "hamming_loss": 0.04},
        }
        table = build_comparison_table(results)
        plot_performance_comparison(table, save=False)
