"""
tests/test_dataset_loader.py
-----------------------------
Unit tests for dataset_loader.py (offline portions only —
the GoEmotions HF download is skipped in CI via the mock fixture).
"""

import pandas as pd
import numpy as np
import pytest

from src.dataset_loader import (
    load_crisis_dataset,
    split_crisis,
    add_hierarchical_columns,
    GOEMOTIONS_LABELS,
    EMOTION_HIERARCHY,
    EMOTION_TO_GROUP,
)


class TestCrisisDataset:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = load_crisis_dataset()

    def test_returns_dataframe(self):
        assert isinstance(self.df, pd.DataFrame)

    def test_required_columns(self):
        assert "text" in self.df.columns
        assert "crisis" in self.df.columns

    def test_binary_labels(self):
        unique = self.df["crisis"].unique()
        assert set(unique).issubset({0, 1})

    def test_balanced_classes(self):
        counts = self.df["crisis"].value_counts()
        assert counts[0] == counts[1]

    def test_no_empty_texts(self):
        assert (self.df["text"].str.strip() != "").all()


class TestSplitCrisis:
    def test_split_sizes(self):
        df = load_crisis_dataset()
        train, test = split_crisis(df, test_size=0.2)
        assert len(train) + len(test) == len(df)
        assert abs(len(test) / len(df) - 0.2) < 0.05

    def test_no_overlap(self):
        df = load_crisis_dataset()
        train, test = split_crisis(df, test_size=0.2)
        train_texts = set(train["text"].tolist())
        test_texts  = set(test["text"].tolist())
        assert len(train_texts & test_texts) == 0

    def test_class_balance_preserved(self):
        df = load_crisis_dataset()
        train, test = split_crisis(df, test_size=0.2)
        # Both splits should contain both classes
        assert set(train["crisis"].unique()) == {0, 1}
        assert set(test["crisis"].unique())  == {0, 1}


class TestHierarchicalColumns:
    @pytest.fixture(autouse=True)
    def setup(self, tiny_goemotions_df):
        self.df = add_hierarchical_columns(tiny_goemotions_df)

    def test_group_columns_added(self):
        for group in ["positive", "negative", "ambiguous", "neutral"]:
            assert f"group_{group}" in self.df.columns

    def test_group_columns_binary(self):
        for group in ["positive", "negative", "ambiguous", "neutral"]:
            col = f"group_{group}"
            unique = self.df[col].unique()
            assert set(unique).issubset({0, 1})

    def test_neutral_label_maps_to_neutral_group(self, tiny_goemotions_df):
        # Find rows where neutral label == 1
        has_neutral = tiny_goemotions_df["neutral"] == 1
        if has_neutral.any():
            rows = self.df[has_neutral]
            assert (rows["group_neutral"] == 1).all()

    def test_joy_maps_to_positive_group(self, tiny_goemotions_df):
        has_joy = tiny_goemotions_df["joy"] == 1
        if has_joy.any():
            rows = self.df[has_joy]
            assert (rows["group_positive"] == 1).all()


class TestEmotionConstants:
    def test_label_list_length(self):
        assert len(GOEMOTIONS_LABELS) == 28

    def test_no_duplicate_labels(self):
        assert len(GOEMOTIONS_LABELS) == len(set(GOEMOTIONS_LABELS))

    def test_hierarchy_covers_all_labels(self):
        all_in_hierarchy = {
            e for emotions in EMOTION_HIERARCHY.values() for e in emotions
        }
        assert all_in_hierarchy == set(GOEMOTIONS_LABELS)

    def test_emotion_to_group_covers_all(self):
        for label in GOEMOTIONS_LABELS:
            assert label in EMOTION_TO_GROUP, f"{label} missing from EMOTION_TO_GROUP"

    def test_emotion_to_group_valid_groups(self):
        valid_groups = set(EMOTION_HIERARCHY.keys())
        for label, group in EMOTION_TO_GROUP.items():
            assert group in valid_groups, f"{label} → {group} is not a valid group"
