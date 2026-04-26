"""
tests/test_preprocessing.py
----------------------------
Unit tests for data_preprocessing.py
"""

import numpy as np
import pytest
from src.data_preprocessing import (
    clean_text,
    clean_dataframe,
    build_tfidf_features,
    extract_label_matrix,
    SimpleTokenizer,
)


class TestCleanText:
    def test_lowercasing(self):
        assert clean_text("Hello WORLD") == "hello world"

    def test_url_removal(self):
        result = clean_text("Visit https://example.com for more info")
        assert "http" not in result
        assert "example" not in result

    def test_html_removal(self):
        result = clean_text("Check <b>this</b> out <br> now")
        assert "<" not in result
        assert ">" not in result

    def test_contraction_expansion(self):
        result = clean_text("I can't do it, won't try")
        assert "cannot" in result
        assert "will not" in result

    def test_special_char_removal(self):
        result = clean_text("Hello!!! How are you??? 😊 #great")
        assert "!" not in result
        assert "?" not in result
        assert "#" not in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_non_string_input(self):
        assert clean_text(None) == ""
        assert clean_text(42) == ""

    def test_stopword_removal(self):
        result = clean_text("the cat sat on the mat", remove_stopwords=True)
        assert "the" not in result.split()

    def test_negation_preserved(self):
        # "not" should NOT be removed even with stopword removal enabled
        result = clean_text("I do not like this", remove_stopwords=True)
        assert "not" in result

    def test_lemmatization(self):
        result = clean_text("running dogs are barking loudly", lemmatize=True)
        # Lemmatizer should reduce "running" → "running" or "run"
        # At minimum, verify no crash and non-empty output
        assert len(result) > 0


class TestCleanDataframe:
    def test_returns_dataframe(self, tiny_goemotions_df):
        result = clean_dataframe(tiny_goemotions_df)
        import pandas as pd
        assert isinstance(result, pd.DataFrame)

    def test_text_column_modified(self, tiny_goemotions_df):
        result = clean_dataframe(tiny_goemotions_df)
        # All cleaned texts should be lowercase
        assert all(t == t.lower() for t in result["text"])

    def test_empty_rows_dropped(self):
        import pandas as pd
        df = pd.DataFrame({"text": ["hello world", "   ", "great day"], "label": [1, 0, 1]})
        result = clean_dataframe(df)
        assert len(result) == 2  # "   " becomes "" after cleaning → dropped

    def test_label_columns_preserved(self, tiny_goemotions_df, label_names):
        result = clean_dataframe(tiny_goemotions_df)
        for col in label_names:
            assert col in result.columns


class TestTfidfFeatures:
    def test_output_shapes(self, tiny_splits, label_names):
        X_tr, X_v, X_te, vec = build_tfidf_features(
            tiny_splits["train"]["text"],
            tiny_splits["validation"]["text"],
            tiny_splits["test"]["text"],
            max_features=200,
        )
        assert X_tr.shape[0] == len(tiny_splits["train"])
        assert X_v.shape[0]  == len(tiny_splits["validation"])
        assert X_te.shape[0] == len(tiny_splits["test"])
        # All should have the same number of features
        assert X_tr.shape[1] == X_v.shape[1] == X_te.shape[1]

    def test_vocabulary_not_empty(self, tiny_splits):
        _, _, _, vec = build_tfidf_features(
            tiny_splits["train"]["text"],
            tiny_splits["validation"]["text"],
            tiny_splits["test"]["text"],
            max_features=200,
        )
        assert len(vec.vocabulary_) > 0


class TestExtractLabelMatrix:
    def test_shape(self, tiny_goemotions_df, label_names):
        matrix = extract_label_matrix(tiny_goemotions_df, label_names)
        assert matrix.shape == (len(tiny_goemotions_df), len(label_names))

    def test_binary_values(self, tiny_goemotions_df, label_names):
        matrix = extract_label_matrix(tiny_goemotions_df, label_names)
        unique = np.unique(matrix)
        assert set(unique).issubset({0.0, 1.0})

    def test_dtype(self, tiny_goemotions_df, label_names):
        matrix = extract_label_matrix(tiny_goemotions_df, label_names)
        assert matrix.dtype == np.float32


class TestSimpleTokenizer:
    def test_fit_and_encode(self, sample_texts):
        tok = SimpleTokenizer(max_vocab=500, max_len=16)
        tok.fit(sample_texts)
        ids = tok.encode(sample_texts[0])
        assert len(ids) == 16
        assert all(isinstance(i, (int, np.integer)) for i in ids)

    def test_padding(self):
        tok = SimpleTokenizer(max_vocab=100, max_len=10)
        tok.fit(["hello world"])
        ids = tok.encode("hi")   # 1 token → 9 pads
        assert len(ids) == 10
        assert ids.count(tok.PAD) == 9

    def test_truncation(self):
        tok = SimpleTokenizer(max_vocab=100, max_len=3)
        tok.fit(["one two three four five six"])
        ids = tok.encode("one two three four five six")
        assert len(ids) == 3

    def test_unk_for_unseen_word(self):
        tok = SimpleTokenizer(max_vocab=10, max_len=5)
        tok.fit(["hello world"])
        ids = tok.encode("completely_unseen_word_xyz")
        assert tok.UNK in ids

    def test_encode_batch_shape(self, sample_texts):
        tok = SimpleTokenizer(max_vocab=500, max_len=20)
        tok.fit(sample_texts)
        batch = tok.encode_batch(sample_texts[:4])
        assert batch.shape == (4, 20)

    def test_vocab_size_property(self, sample_texts):
        tok = SimpleTokenizer(max_vocab=200, max_len=10)
        tok.fit(sample_texts)
        assert tok.vocab_size <= 202   # PAD + UNK + up to 200 words

    def test_fit_required_before_encode(self):
        tok = SimpleTokenizer()
        with pytest.raises(AssertionError):
            tok.encode("hello world")
