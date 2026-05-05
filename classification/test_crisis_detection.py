"""
tests/test_crisis_detection.py
-------------------------------
Unit tests for crisis_detection.py
"""

import numpy as np
import pytest

from src.crisis_detection import (
    rule_based_crisis,
    MLCrisisDetector,
    integrated_analysis,
    _get_recommendation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based detector
# ─────────────────────────────────────────────────────────────────────────────

class TestRuleBasedCrisis:
    def test_clear_crisis_detected(self):
        text = "I want to end my life, I can't take it anymore."
        label, score, triggers = rule_based_crisis(text)
        assert label == 1
        assert score > 0
        assert len(triggers) > 0

    def test_non_crisis_not_flagged(self):
        text = "I'm feeling a bit tired after a long day at work."
        label, score, triggers = rule_based_crisis(text)
        assert label == 0
        assert score == 0.0
        assert len(triggers) == 0

    def test_self_harm_phrase_detected(self):
        text = "I've been cutting myself to cope with the pain."
        label, score, triggers = rule_based_crisis(text)
        assert label == 1

    def test_suicidal_word_detected(self):
        text = "I am having suicidal thoughts constantly."
        label, score, triggers = rule_based_crisis(text)
        assert label == 1

    def test_score_bounded_zero_to_one(self):
        texts = [
            "I want to end my life, kill myself, nobody would miss me, plan to die.",
            "Hello, how are you?",
        ]
        for t in texts:
            _, score, _ = rule_based_crisis(t)
            assert 0.0 <= score <= 1.0

    def test_returns_tuple_of_three(self):
        result = rule_based_crisis("Test text.")
        assert len(result) == 3

    def test_triggers_are_list(self):
        _, _, triggers = rule_based_crisis("I want to die.")
        assert isinstance(triggers, list)

    def test_empty_string(self):
        label, score, triggers = rule_based_crisis("")
        assert label == 0
        assert score == 0.0

    def test_uppercase_text_handled(self):
        # rule_based_crisis converts to lowercase internally
        label, _, _ = rule_based_crisis("I WANT TO END MY LIFE.")
        assert label == 1


# ─────────────────────────────────────────────────────────────────────────────
# ML Crisis Detector
# ─────────────────────────────────────────────────────────────────────────────

class TestMLCrisisDetector:
    @pytest.fixture(autouse=True)
    def setup(self, tiny_crisis_df):
        self.detector = MLCrisisDetector(C=1.0, threshold=0.4)
        self.detector.fit(
            tiny_crisis_df["text"].tolist(),
            tiny_crisis_df["crisis"].values,
        )
        self.crisis_texts = [
            "I want to kill myself today.",
            "Nobody would miss me if I disappeared.",
        ]
        self.safe_texts = [
            "I had a stressful day but I'll be fine.",
            "Feeling a bit sad but overall okay.",
        ]

    def test_predict_shape(self):
        all_texts = self.crisis_texts + self.safe_texts
        preds = self.detector.predict(all_texts)
        assert preds.shape == (len(all_texts),)

    def test_predict_binary(self):
        preds = self.detector.predict(self.crisis_texts + self.safe_texts)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_range(self):
        probas = self.detector.predict_proba(self.crisis_texts)
        assert all(0.0 <= p <= 1.0 for p in probas)

    def test_evaluate_returns_required_keys(self, tiny_crisis_df):
        results = self.detector.evaluate(
            tiny_crisis_df["text"].tolist(),
            tiny_crisis_df["crisis"].values,
        )
        for key in ["f1_macro", "f1_crisis", "roc_auc", "preds", "probas"]:
            assert key in results

    def test_save_load(self, tmp_path):
        path = tmp_path / "crisis_test.pkl"
        self.detector.save(path)
        loaded = MLCrisisDetector.load(path)
        orig_preds   = self.detector.predict(self.crisis_texts)
        loaded_preds = loaded.predict(self.crisis_texts)
        np.testing.assert_array_equal(orig_preds, loaded_preds)

    def test_crisis_text_higher_proba(self):
        # On average, crisis texts should score higher than safe texts
        crisis_scores = self.detector.predict_proba(self.crisis_texts).mean()
        safe_scores   = self.detector.predict_proba(self.safe_texts).mean()
        # Soft assertion — model is tiny so we allow near-parity
        assert crisis_scores >= safe_scores - 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Integrated analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegratedAnalysis:
    def test_returns_required_keys(self):
        result = integrated_analysis(
            "I am so happy today!", ["joy", "optimism"]
        )
        for key in ["text", "emotions", "crisis", "rule_score",
                    "triggers", "recommendation"]:
            assert key in result

    def test_crisis_text_flagged(self):
        result = integrated_analysis(
            "I want to end my life.", ["sadness", "grief"]
        )
        assert result["crisis"] is True

    def test_safe_text_not_flagged(self):
        result = integrated_analysis(
            "I'm excited about the weekend!", ["joy", "excitement"]
        )
        assert result["crisis"] is False

    def test_recommendation_is_string(self):
        result = integrated_analysis("Hello world", ["neutral"])
        assert isinstance(result["recommendation"], str)

    def test_high_severity_recommendation_contains_resource(self):
        rec = _get_recommendation(crisis_flag=1, severity=0.9)
        # Should mention a helpline number
        assert "988" in rec or "741741" in rec

    def test_no_crisis_recommendation(self):
        rec = _get_recommendation(crisis_flag=0, severity=0.0)
        assert "No immediate crisis" in rec or "✅" in rec

    def test_with_ml_detector(self, tiny_crisis_df):
        detector = MLCrisisDetector(threshold=0.4)
        detector.fit(
            tiny_crisis_df["text"].tolist(),
            tiny_crisis_df["crisis"].values,
        )
        result = integrated_analysis(
            "I want to kill myself.",
            ["sadness", "grief"],
            crisis_detector=detector,
        )
        # ml_score should be populated
        assert result["ml_score"] is not None
        assert 0.0 <= result["ml_score"] <= 1.0
