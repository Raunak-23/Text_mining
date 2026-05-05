"""
data_preprocessing.py
---------------------
Text cleaning, tokenisation, label encoding, and multi-label binarisation
for the GoEmotions + crisis datasets.
"""

import re
import string
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data silently (best-effort — works offline too)
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT_DIR / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

# ── Stopword set (keep negations — they matter for sentiment) ────────────────
_KEEP_NEGATIONS = {"no", "not", "nor", "neither", "never", "none", "nobody",
                   "nothing", "nowhere", "don't", "doesn't", "didn't",
                   "won't", "wouldn't", "can't", "couldn't", "shouldn't",
                   "haven't", "hasn't", "hadn't", "isn't", "aren't", "wasn't",
                   "weren't"}
# Graceful fallback if NLTK data unavailable (e.g. no internet in CI)
try:
    _STOP = set(stopwords.words("english")) - _KEEP_NEGATIONS
except LookupError:
    logger.warning("NLTK stopwords not available — stopword removal disabled.")
    _STOP: set = set()


# ─────────────────────────────────────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str, remove_stopwords: bool = False,
               lemmatize: bool = False) -> str:
    """
    Apply a sequence of cleaning steps to a raw comment string.

    Steps
    -----
    1. Lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Expand common contractions
    5. Remove special characters / extra whitespace
    6. (Optional) Remove stopwords
    7. (Optional) Lemmatize
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)

    # 3. Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # 4. Expand basic contractions
    contractions = {
        r"won't": "will not", r"can't": "cannot", r"n't": " not",
        r"'re": " are", r"'s": " is", r"'d": " would",
        r"'ll": " will", r"'ve": " have", r"'m": " am",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)

    # 5. Remove special characters / punctuation (keep spaces)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    # 6. Remove stopwords (optional)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOP]

    # 7. Lemmatize (optional — skipped silently if wordnet unavailable)
    if lemmatize:
        try:
            lem = WordNetLemmatizer()
            tokens = [lem.lemmatize(t) for t in tokens]
        except LookupError:
            pass  # wordnet not downloaded; skip lemmatization

    return " ".join(tokens)


def clean_dataframe(df: pd.DataFrame, text_col: str = "text",
                    remove_stopwords: bool = False,
                    lemmatize: bool = False) -> pd.DataFrame:
    """Apply clean_text to every row and return updated DataFrame."""
    df = df.copy()
    df[text_col] = df[text_col].apply(
        lambda t: clean_text(t, remove_stopwords=remove_stopwords,
                             lemmatize=lemmatize)
    )
    # Drop rows with empty text after cleaning
    df = df[df[text_col].str.strip() != ""].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Multi-label Binarisation
# ─────────────────────────────────────────────────────────────────────────────

GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]


def extract_label_matrix(df: pd.DataFrame,
                         label_cols: Optional[List[str]] = None
                         ) -> np.ndarray:
    """
    Given a DataFrame where emotion columns already exist as 0/1 integers,
    return the label matrix as a NumPy array.
    """
    if label_cols is None:
        label_cols = [c for c in GOEMOTIONS_LABELS if c in df.columns]
    return df[label_cols].values.astype(np.float32)


def get_label_names(df: pd.DataFrame) -> List[str]:
    return [c for c in GOEMOTIONS_LABELS if c in df.columns]


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF Features (for traditional ML)
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_features(train_texts: pd.Series,
                         val_texts: pd.Series,
                         test_texts: pd.Series,
                         max_features: int = 20_000,
                         ngram_range=(1, 2)):
    """
    Fit a TF-IDF vectoriser on training data and transform all splits.

    Returns
    -------
    X_train, X_val, X_test : scipy sparse matrices
    vectorizer             : fitted TfidfVectorizer
    """
    logger.info("Fitting TF-IDF (max_features=%d, ngrams=%s) …",
                max_features, ngram_range)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    logger.info("  Vocabulary size: %d", len(vectorizer.vocabulary_))
    return X_train, X_val, X_test, vectorizer


# ─────────────────────────────────────────────────────────────────────────────
# Simple word-to-index tokeniser (for LSTM)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleTokenizer:
    """
    Builds a word vocabulary from training texts and encodes sentences
    as fixed-length integer sequences (for LSTM embedding layer).
    """

    PAD = 0
    UNK = 1

    def __init__(self, max_vocab: int = 30_000, max_len: int = 128):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.word2idx: dict = {}
        self.idx2word: dict = {}
        self._fitted = False

    def fit(self, texts: List[str]) -> "SimpleTokenizer":
        from collections import Counter
        freq: Counter = Counter()
        for text in texts:
            freq.update(text.lower().split())

        self.word2idx = {"<PAD>": self.PAD, "<UNK>": self.UNK}
        for word, _ in freq.most_common(self.max_vocab - 2):
            self.word2idx[word] = len(self.word2idx)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self._fitted = True
        logger.info("SimpleTokenizer vocab size: %d", len(self.word2idx))
        return self

    def encode(self, text: str) -> List[int]:
        assert self._fitted, "Call .fit() first."
        tokens = text.lower().split()[: self.max_len]
        ids = [self.word2idx.get(t, self.UNK) for t in tokens]
        # Pad or truncate to max_len
        ids += [self.PAD] * (self.max_len - len(ids))
        return ids

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.encode(t) for t in texts], dtype=np.int64)

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing pipeline (convenience)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_splits(splits: dict,
                      text_col: str = "text",
                      remove_stopwords: bool = False,
                      lemmatize: bool = False) -> dict:
    """
    Clean text in all splits (train/validation/test).

    Parameters
    ----------
    splits : dict returned by dataset_loader.load_goemotions()

    Returns
    -------
    dict with same keys, cleaned DataFrames
    """
    cleaned = {}
    for split, df in splits.items():
        logger.info("Cleaning '%s' split (%d rows) …", split, len(df))
        cleaned[split] = clean_dataframe(df, text_col=text_col,
                                         remove_stopwords=remove_stopwords,
                                         lemmatize=lemmatize)
        # Save
        path = DATA_PROC / f"goemotions_{split}_clean.csv"
        cleaned[split].to_csv(path, index=False)
        logger.info("  Saved → %s", path)
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "I can't believe how AMAZING this is!!! Check out https://example.com 😊"
    print("Raw     :", sample)
    print("Cleaned :", clean_text(sample, remove_stopwords=True, lemmatize=True))

    texts = ["I am so happy today", "This is terrible and sad"]
    tok = SimpleTokenizer(max_vocab=100, max_len=8)
    tok.fit(texts)
    print("Encoded :", tok.encode_batch(texts))
