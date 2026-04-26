"""
training.py
-----------
End-to-end training pipeline for all four model families:
  1. Logistic Regression (TF-IDF)
  2. Linear SVM         (TF-IDF)
  3. BiLSTM             (PyTorch)
  4. BERT               (HuggingFace)
  5. LLM classifier     (simulation / API)

Run this script directly to train everything:
    python src/training.py

Or import individual train_* functions for fine-grained control.
"""

import argparse
import logging
import os
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports (avoids loading transformers unless needed)
# ─────────────────────────────────────────────────────────────────────────────

def _import_modules():
    """Import project modules (allows running without installing everything)."""
    import sys
    sys.path.insert(0, str(ROOT_DIR))

    from src.dataset_loader import load_all, GOEMOTIONS_LABELS
    from src.data_preprocessing import (
        preprocess_splits, build_tfidf_features,
        extract_label_matrix, SimpleTokenizer, clean_dataframe,
    )
    from src.models.traditional_ml import MultiLabelLogisticRegression, MultiLabelSVM
    from src.models.lstm_model import (
        BiLSTMEmotionClassifier, EmotionDataset, train_lstm, evaluate,
    )
    from src.models.bert_model import (
        BERTEmotionClassifier, build_bert_loaders, train_bert, evaluate_bert,
        DEFAULT_CHECKPOINT,
    )
    from src.models.llm_classifier import LLMClassifier
    from src.evaluation import compute_metrics, print_metrics, save_results
    from src.crisis_detection import MLCrisisDetector

    return {
        "load_all": load_all,
        "GOEMOTIONS_LABELS": GOEMOTIONS_LABELS,
        "preprocess_splits": preprocess_splits,
        "build_tfidf_features": build_tfidf_features,
        "extract_label_matrix": extract_label_matrix,
        "SimpleTokenizer": SimpleTokenizer,
        "clean_dataframe": clean_dataframe,
        "MultiLabelLogisticRegression": MultiLabelLogisticRegression,
        "MultiLabelSVM": MultiLabelSVM,
        "BiLSTMEmotionClassifier": BiLSTMEmotionClassifier,
        "EmotionDataset": EmotionDataset,
        "train_lstm": train_lstm,
        "evaluate_lstm": evaluate,
        "BERTEmotionClassifier": BERTEmotionClassifier,
        "build_bert_loaders": build_bert_loaders,
        "train_bert": train_bert,
        "evaluate_bert": evaluate_bert,
        "DEFAULT_CHECKPOINT": DEFAULT_CHECKPOINT,
        "LLMClassifier": LLMClassifier,
        "compute_metrics": compute_metrics,
        "print_metrics": print_metrics,
        "save_results": save_results,
        "MLCrisisDetector": MLCrisisDetector,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Traditional ML
# ─────────────────────────────────────────────────────────────────────────────

def train_traditional_ml(m, data: dict, label_names: list,
                          subsample: Optional[int] = None) -> dict:
    """Train LogReg and SVM on TF-IDF features."""
    logger.info("=" * 60)
    logger.info("STEP: Traditional ML (LogReg + SVM)")
    logger.info("=" * 60)

    splits = data["goemotions"]
    train_df = splits["train"]
    val_df   = splits["validation"]
    test_df  = splits["test"]

    # Optionally subsample for speed
    if subsample:
        train_df = train_df.sample(n=min(subsample, len(train_df)), random_state=42)

    X_train, X_val, X_test, vectorizer = m["build_tfidf_features"](
        train_df["text"], val_df["text"], test_df["text"]
    )

    y_train = m["extract_label_matrix"](train_df, label_names)
    y_val   = m["extract_label_matrix"](val_df,   label_names)
    y_test  = m["extract_label_matrix"](test_df,  label_names)

    results = {}

    # LogReg
    lr = m["MultiLabelLogisticRegression"]().build(label_names)
    t0 = time.time()
    lr.fit(X_train, y_train)
    logger.info("LogReg trained in %.1fs", time.time() - t0)
    preds = lr.predict(X_test)
    results["Logistic Regression"] = m["compute_metrics"](y_test, preds, label_names)
    m["print_metrics"](results["Logistic Regression"], "Logistic Regression")
    lr.save()

    # SVM
    svm = m["MultiLabelSVM"]().build(label_names)
    t0 = time.time()
    svm.fit(X_train, y_train)
    logger.info("SVM trained in %.1fs", time.time() - t0)
    preds = svm.predict(X_test)
    results["SVM"] = m["compute_metrics"](y_test, preds, label_names)
    m["print_metrics"](results["SVM"], "SVM")
    svm.save()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. LSTM
# ─────────────────────────────────────────────────────────────────────────────

def train_lstm_model(m, data: dict, label_names: list,
                     epochs: int = 5, batch_size: int = 64,
                     subsample: Optional[int] = None) -> dict:
    """Train BiLSTM with attention."""
    logger.info("=" * 60)
    logger.info("STEP: BiLSTM with Attention")
    logger.info("=" * 60)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = data["goemotions"]
    train_df = splits["train"]
    val_df   = splits["validation"]
    test_df  = splits["test"]

    if subsample:
        train_df = train_df.sample(n=min(subsample, len(train_df)), random_state=42)

    # Build tokeniser
    tokenizer = m["SimpleTokenizer"](max_vocab=30_000, max_len=128)
    tokenizer.fit(train_df["text"].tolist())

    # Encode
    X_train = tokenizer.encode_batch(train_df["text"].tolist())
    X_val   = tokenizer.encode_batch(val_df["text"].tolist())
    X_test  = tokenizer.encode_batch(test_df["text"].tolist())

    y_train = m["extract_label_matrix"](train_df, label_names)
    y_val   = m["extract_label_matrix"](val_df,   label_names)
    y_test  = m["extract_label_matrix"](test_df,  label_names)

    # DataLoaders
    train_ds = m["EmotionDataset"](X_train, y_train)
    val_ds   = m["EmotionDataset"](X_val,   y_val)
    test_ds  = m["EmotionDataset"](X_test,  y_test)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_size)

    model = m["BiLSTMEmotionClassifier"](
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        num_labels=len(label_names),
        dropout=0.4,
    )

    history = m["train_lstm"](model, train_loader, val_loader, epochs=epochs)

    # Evaluate
    criterion = torch.nn.BCEWithLogitsLoss()
    res = m["evaluate_lstm"](model, test_loader, criterion)
    metrics = m["compute_metrics"](res["labels"], res["preds"], label_names)
    m["print_metrics"](metrics, "BiLSTM")
    metrics["history"] = history

    return {"BiLSTM": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# 3. BERT
# ─────────────────────────────────────────────────────────────────────────────

def train_bert_model(m, data: dict, label_names: list,
                     epochs: int = 2, batch_size: int = 32,
                     max_len: int = 128,
                     subsample: Optional[int] = None) -> dict:
    """Fine-tune BERT."""
    logger.info("=" * 60)
    logger.info("STEP: BERT Fine-tuning")
    logger.info("=" * 60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(m["DEFAULT_CHECKPOINT"])

    splits = data["goemotions"]
    train_df = splits["train"]
    val_df   = splits["validation"]
    test_df  = splits["test"]

    if subsample:
        train_df = train_df.sample(n=min(subsample, len(train_df)), random_state=42)

    y_train = m["extract_label_matrix"](train_df, label_names)
    y_val   = m["extract_label_matrix"](val_df,   label_names)
    y_test  = m["extract_label_matrix"](test_df,  label_names)

    train_loader, val_loader, test_loader = m["build_bert_loaders"](
        train_df["text"].tolist(), y_train,
        val_df["text"].tolist(),   y_val,
        test_df["text"].tolist(),  y_test,
        tokenizer=tokenizer,
        max_len=max_len,
        batch_size=batch_size,
    )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = m["BERTEmotionClassifier"](
        num_labels=len(label_names),
        checkpoint=m["DEFAULT_CHECKPOINT"],
        dropout=0.3,
    )

    history = m["train_bert"](model, train_loader, val_loader,
                               y_train=y_train, epochs=epochs)

    criterion = torch.nn.BCEWithLogitsLoss()
    res = m["evaluate_bert"](model, test_loader, criterion)
    metrics = m["compute_metrics"](res["labels"], res["preds"], label_names)
    m["print_metrics"](metrics, "BERT")
    metrics["history"] = history

    return {"BERT": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# 4. LLM Classifier
# ─────────────────────────────────────────────────────────────────────────────

def run_llm_classifier(m, data: dict, label_names: list,
                        n_samples: int = 500) -> dict:
    """Run LLM simulation on a sample of test data."""
    logger.info("=" * 60)
    logger.info("STEP: LLM Classifier (simulation)")
    logger.info("=" * 60)

    test_df = data["goemotions"]["test"]
    sample  = test_df.sample(n=min(n_samples, len(test_df)), random_state=42)

    clf = m["LLMClassifier"](use_api=False, label_names=label_names)
    preds_matrix = clf.predict_matrix(sample["text"].tolist())
    y_true = m["extract_label_matrix"](sample, label_names)

    metrics = m["compute_metrics"](y_true, preds_matrix, label_names)
    m["print_metrics"](metrics, "LLM (Simulation)")

    return {"LLM (Simulation)": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Crisis Detector
# ─────────────────────────────────────────────────────────────────────────────

def train_crisis_detector(m, data: dict) -> None:
    logger.info("=" * 60)
    logger.info("STEP: Crisis Detector")
    logger.info("=" * 60)

    crisis_train = data["crisis_train"]
    crisis_test  = data["crisis_test"]

    detector = m["MLCrisisDetector"]()
    detector.fit(crisis_train["text"].tolist(), crisis_train["crisis"].values)
    results = detector.evaluate(crisis_test["text"].tolist(), crisis_test["crisis"].values)
    detector.save()

    logger.info("Crisis  F1=%.4f  ROC-AUC=%.4f",
                results["f1_crisis"], results["roc_auc"])


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(args: argparse.Namespace) -> None:
    m = _import_modules()

    logger.info("Loading datasets …")
    data = m["load_all"]()
    label_names = m["GOEMOTIONS_LABELS"]

    # Preprocess
    logger.info("Preprocessing text …")
    data["goemotions"] = m["preprocess_splits"](data["goemotions"])

    all_results = {}

    if args.models in ("all", "ml"):
        r = train_traditional_ml(m, data, label_names,
                                  subsample=args.subsample)
        all_results.update(r)

    if args.models in ("all", "lstm"):
        r = train_lstm_model(m, data, label_names,
                              epochs=args.lstm_epochs,
                              batch_size=args.batch_size,
                              subsample=args.subsample)
        all_results.update(r)

    if args.models in ("all", "bert"):
        r = train_bert_model(m, data, label_names,
                              epochs=args.bert_epochs,
                              batch_size=args.batch_size,
                              subsample=args.subsample)
        all_results.update(r)

    if args.models in ("all", "llm"):
        r = run_llm_classifier(m, data, label_names,
                                n_samples=args.llm_samples)
        all_results.update(r)

    if args.models in ("all", "crisis"):
        train_crisis_detector(m, data)

    # Save comparison table
    if all_results:
        m["save_results"](all_results, "model_comparison.csv")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train emotion classification models")
    p.add_argument("--models",       default="all",
                   choices=["all", "ml", "lstm", "bert", "llm", "crisis"],
                   help="Which models to train")
    p.add_argument("--subsample",    type=int,   default=None,
                   help="Subsample training data (for quick tests)")
    p.add_argument("--lstm-epochs",  type=int,   default=5)
    p.add_argument("--bert-epochs",  type=int,   default=2)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--llm-samples",  type=int,   default=500)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_full_pipeline(args)
