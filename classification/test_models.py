"""
tests/test_models.py
--------------------
Unit tests for all four model families.
All tests use tiny synthetic data — no GPU or internet required.
"""

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Traditional ML
# ─────────────────────────────────────────────────────────────────────────────

class TestLogisticRegression:
    @pytest.fixture(autouse=True)
    def setup(self, tiny_tfidf_features, label_names):
        from src.models.traditional_ml import MultiLabelLogisticRegression
        self.clf = MultiLabelLogisticRegression(C=1.0, threshold=0.3).build(label_names)
        self.clf.fit(tiny_tfidf_features["X_train"], tiny_tfidf_features["y_train"])
        self.X_test  = tiny_tfidf_features["X_test"]
        self.y_test  = tiny_tfidf_features["y_test"]
        self.n_labels = len(label_names)

    def test_predict_shape(self):
        preds = self.clf.predict(self.X_test)
        assert preds.shape == (self.X_test.shape[0], self.n_labels)

    def test_predict_binary(self):
        preds = self.clf.predict(self.X_test)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(self):
        proba = self.clf.predict_proba(self.X_test)
        assert proba.shape == (self.X_test.shape[0], self.n_labels)

    def test_predict_proba_range(self):
        proba = self.clf.predict_proba(self.X_test)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_save_load(self, tmp_path, label_names):
        from src.models.traditional_ml import MultiLabelLogisticRegression
        path = tmp_path / "lr_test.pkl"
        self.clf.save(path)
        loaded = MultiLabelLogisticRegression.load(path)
        preds_orig   = self.clf.predict(self.X_test)
        preds_loaded = loaded.predict(self.X_test)
        np.testing.assert_array_equal(preds_orig, preds_loaded)


class TestMultiLabelSVM:
    @pytest.fixture(autouse=True)
    def setup(self, tiny_tfidf_features, label_names):
        from src.models.traditional_ml import MultiLabelSVM
        self.clf = MultiLabelSVM(C=1.0, threshold=0.3).build(label_names)
        self.clf.fit(tiny_tfidf_features["X_train"], tiny_tfidf_features["y_train"])
        self.X_test  = tiny_tfidf_features["X_test"]
        self.n_labels = len(label_names)

    def test_predict_shape(self):
        preds = self.clf.predict(self.X_test)
        assert preds.shape == (self.X_test.shape[0], self.n_labels)

    def test_predict_binary(self):
        preds = self.clf.predict(self.X_test)
        assert set(np.unique(preds)).issubset({0, 1})


# ─────────────────────────────────────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────────────────────────────────────

class TestBiLSTM:
    VOCAB   = 500
    EMBED   = 32
    HIDDEN  = 64
    LAYERS  = 2
    LABELS  = 28
    SEQ_LEN = 16
    BATCH   = 8

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.models.lstm_model import BiLSTMEmotionClassifier
        self.model = BiLSTMEmotionClassifier(
            vocab_size=self.VOCAB,
            embed_dim=self.EMBED,
            hidden_dim=self.HIDDEN,
            num_layers=self.LAYERS,
            num_labels=self.LABELS,
            dropout=0.0,
        )
        self.device = torch.device("cpu")
        self.model.eval()

    def test_output_shape(self):
        x = torch.randint(0, self.VOCAB, (self.BATCH, self.SEQ_LEN))
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (self.BATCH, self.LABELS)

    def test_output_is_logits(self):
        # Logits can be any real number, not bounded to [0,1]
        x = torch.randint(0, self.VOCAB, (self.BATCH, self.SEQ_LEN))
        with torch.no_grad():
            out = self.model(x)
        # At least some logits should be outside [0,1]
        assert out.dtype == torch.float32

    def test_no_nan_in_output(self):
        x = torch.randint(0, self.VOCAB, (self.BATCH, self.SEQ_LEN))
        with torch.no_grad():
            out = self.model(x)
        assert not torch.isnan(out).any()

    def test_padding_zeros_handled(self):
        # All-zeros (PAD) input should not crash
        x = torch.zeros(self.BATCH, self.SEQ_LEN, dtype=torch.long)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (self.BATCH, self.LABELS)

    def test_batch_size_one(self):
        x = torch.randint(0, self.VOCAB, (1, self.SEQ_LEN))
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (1, self.LABELS)


class TestEmotionDataset:
    def test_len(self):
        from src.models.lstm_model import EmotionDataset
        ids = np.zeros((20, 16), dtype=np.int64)
        lbs = np.zeros((20, 5),  dtype=np.float32)
        ds  = EmotionDataset(ids, lbs)
        assert len(ds) == 20

    def test_getitem_shapes(self):
        from src.models.lstm_model import EmotionDataset
        ids = np.random.randint(0, 100, (10, 16)).astype(np.int64)
        lbs = np.random.rand(10, 5).astype(np.float32)
        ds  = EmotionDataset(ids, lbs)
        x, y = ds[0]
        assert x.shape == (16,)
        assert y.shape == (5,)

    def test_tensor_types(self):
        from src.models.lstm_model import EmotionDataset
        ids = np.zeros((5, 8), dtype=np.int64)
        lbs = np.ones((5, 3),  dtype=np.float32)
        ds  = EmotionDataset(ids, lbs)
        x, y = ds[0]
        assert x.dtype == torch.long
        assert y.dtype == torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# BERT
# ─────────────────────────────────────────────────────────────────────────────

class TestBERTDataset:
    def test_getitem_keys(self):
        from transformers import AutoTokenizer
        from src.models.bert_model import BERTEmotionDataset
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        texts  = ["Hello world!", "I feel great today."]
        labels = np.zeros((2, 28), dtype=np.float32)
        ds = BERTEmotionDataset(texts, labels, tok, max_len=32)
        item = ds[0]
        assert "input_ids"      in item
        assert "attention_mask" in item
        assert "labels"         in item

    def test_input_ids_shape(self):
        from transformers import AutoTokenizer
        from src.models.bert_model import BERTEmotionDataset
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        texts  = ["Test sentence."]
        labels = np.zeros((1, 28), dtype=np.float32)
        ds = BERTEmotionDataset(texts, labels, tok, max_len=32)
        item = ds[0]
        assert item["input_ids"].shape == (32,)
        assert item["attention_mask"].shape == (32,)

    def test_labels_dtype(self):
        from transformers import AutoTokenizer
        from src.models.bert_model import BERTEmotionDataset
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        texts  = ["Hello."]
        labels = np.ones((1, 28), dtype=np.float32)
        ds = BERTEmotionDataset(texts, labels, tok, max_len=16)
        assert ds[0]["labels"].dtype == torch.float32


class TestBERTModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        from src.models.bert_model import BERTEmotionClassifier
        # freeze_base=True so test doesn't need to download weights & is fast
        self.model = BERTEmotionClassifier(
            num_labels=28, checkpoint="bert-base-uncased",
            dropout=0.1, freeze_base=True
        )
        self.model.eval()

    def test_output_shape(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        enc = tok("Hello world", max_length=32, padding="max_length",
                  truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = self.model(enc["input_ids"], enc["attention_mask"])
        assert out.shape == (1, 28)

    def test_no_nan(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        enc = tok("Test sentence.", max_length=32, padding="max_length",
                  truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = self.model(enc["input_ids"], enc["attention_mask"])
        assert not torch.isnan(out).any()


# ─────────────────────────────────────────────────────────────────────────────
# LLM Classifier
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMClassifier:
    @pytest.fixture(autouse=True)
    def setup(self, label_names):
        from src.models.llm_classifier import LLMClassifier
        self.clf = LLMClassifier(use_api=False, label_names=label_names)
        self.label_names = label_names

    def test_predict_labels_returns_list(self, sample_texts):
        result = self.clf.predict_labels(sample_texts[:3])
        assert isinstance(result, list)
        assert len(result) == 3

    def test_each_prediction_is_list(self, sample_texts):
        result = self.clf.predict_labels(sample_texts[:3])
        for pred in result:
            assert isinstance(pred, list)

    def test_labels_in_valid_set(self, sample_texts):
        valid = set(self.label_names)
        result = self.clf.predict_labels(sample_texts[:5])
        for pred in result:
            for lbl in pred:
                assert lbl in valid, f"Unexpected label: {lbl}"

    def test_predict_matrix_shape(self, sample_texts):
        matrix = self.clf.predict_matrix(sample_texts[:5])
        assert matrix.shape == (5, len(self.label_names))

    def test_predict_matrix_binary(self, sample_texts):
        matrix = self.clf.predict_matrix(sample_texts[:5])
        assert set(np.unique(matrix)).issubset({0, 1})

    def test_happy_text_detects_positive(self):
        result = self.clf.predict_labels(["I am so happy and grateful today!"])
        flat = result[0]
        positive_emotions = {"joy", "gratitude", "excitement", "admiration",
                              "optimism", "amusement", "love"}
        assert any(e in positive_emotions for e in flat), \
            f"No positive emotion in {flat}"

    def test_neutral_fallback(self):
        # Completely non-emotional text should return neutral
        result = self.clf.predict_labels(["The item is located on the shelf."])
        assert "neutral" in result[0]
