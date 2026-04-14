"""
utils.py
========
Shared utility functions used across the project:
  - Metric computation (accuracy, precision, recall, F1)
  - Checkpoint saving / loading
  - Early stopping
  - Logging helpers
  - Graph plotting (loss / accuracy curves, confusion matrix)
  - Results CSV writing

Author: Lab Project – Meme & Sarcasm Understanding
"""

import os
import csv
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")                # Non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MemeProject")


def get_logger(name: str = "MemeProject") -> logging.Logger:
    return logging.getLogger(name)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    average: str = "binary") -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true   : ground-truth labels
        y_pred   : predicted labels
        average  : 'binary' | 'macro' | 'weighted'

    Returns:
        dict with keys: accuracy, precision, recall, f1
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {
        "accuracy" : float(acc),
        "precision": float(prec),
        "recall"   : float(rec),
        "f1"       : float(f1),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """Return a pretty one-line string of metrics."""
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())


# ──────────────────────────────────────────────
# AverageMeter (for loss/acc tracking per epoch)
# ──────────────────────────────────────────────
class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / max(self.count, 1)

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


# ──────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────
class EarlyStopping:
    """
    Stops training if validation loss does not improve for `patience` epochs.

    Args:
        patience (int)   : how many epochs to wait
        delta    (float) : minimum improvement to count
        mode     (str)   : 'min' (loss) or 'max' (accuracy)
    """

    def __init__(self, patience: int = 5, delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.delta    = delta
        self.mode     = mode
        self.best     = float("inf") if mode == "min" else -float("inf")
        self.counter  = 0
        self.best_epoch = 0

    def __call__(self, value: float, epoch: int) -> bool:
        improved = (
            (self.mode == "min" and value < self.best - self.delta) or
            (self.mode == "max" and value > self.best + self.delta)
        )
        if improved:
            self.best       = value
            self.counter    = 0
            self.best_epoch = epoch
            return False    # Do NOT stop
        else:
            self.counter += 1
            return self.counter >= self.patience   # Stop if patience exceeded


# ──────────────────────────────────────────────
# Checkpoint Management
# ──────────────────────────────────────────────
def save_checkpoint(state: dict, path: str) -> None:
    """Save training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(path: str, model: nn.Module,
                    optimizer=None) -> Dict:
    """
    Load a training checkpoint.

    Returns:
        dict with 'epoch', 'best_val_loss', 'history'
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    logger.info(f"Loaded checkpoint from {path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt


# ──────────────────────────────────────────────
# Prediction helpers
# ──────────────────────────────────────────────
def batch_predict(model: nn.Module, loader,
                  device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model inference on a DataLoader.

    Returns:
        y_true  : (N,)  ground-truth labels
        y_pred  : (N,)  predicted labels
        y_probs : (N, C) predicted probabilities
    """
    model.eval()
    all_true, all_pred, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            images   = batch["image"].to(device)
            ids      = batch["input_ids"].to(device)
            masks    = batch["attention_mask"].to(device)
            labels   = batch["label"].numpy()

            logits   = model(images, ids, masks)
            probs    = torch.softmax(logits, dim=-1).cpu().numpy()
            preds    = np.argmax(probs, axis=1)

            all_true.append(labels)
            all_pred.append(preds)
            all_probs.append(probs)

    return (np.concatenate(all_true),
            np.concatenate(all_pred),
            np.concatenate(all_probs, axis=0))


# ──────────────────────────────────────────────
# Plotting: Loss + Accuracy Curves
# ──────────────────────────────────────────────
COLOUR_PALETTE = {
    "train": "#2196F3",
    "val"  : "#F44336",
    "grid" : "#EEEEEE",
}


def plot_training_curves(history: dict, model_name: str,
                         save_dir: str = "outputs/graphs") -> str:
    """
    Plot training and validation loss & accuracy curves side-by-side.

    Args:
        history    : dict with keys 'train_loss', 'val_loss',
                     'train_acc', 'val_acc'  (each a list of floats)
        model_name : label for the plot title and filename
        save_dir   : directory to save the PNG

    Returns:
        str: path to the saved figure
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} – Training Curves", fontsize=14, fontweight="bold")

    # ── Loss ────────────────────────────────────
    ax1.plot(epochs, history["train_loss"], color=COLOUR_PALETTE["train"],
             linewidth=2, label="Train Loss", marker="o", markersize=4)
    ax1.plot(epochs, history["val_loss"],   color=COLOUR_PALETTE["val"],
             linewidth=2, label="Val Loss",   marker="s", markersize=4,
             linestyle="--")
    ax1.set_title("Loss", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.yaxis.grid(True, color=COLOUR_PALETTE["grid"])
    ax1.set_axisbelow(True)

    # ── Accuracy ────────────────────────────────
    ax2.plot(epochs, history["train_acc"], color=COLOUR_PALETTE["train"],
             linewidth=2, label="Train Acc", marker="o", markersize=4)
    ax2.plot(epochs, history["val_acc"],   color=COLOUR_PALETTE["val"],
             linewidth=2, label="Val Acc",   marker="s", markersize=4,
             linestyle="--")
    ax2.set_title("Accuracy", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.legend()
    ax2.yaxis.grid(True, color=COLOUR_PALETTE["grid"])
    ax2.set_axisbelow(True)

    plt.tight_layout()
    fname = f"{model_name.lower().replace(' ', '_')}_training_curves.png"
    fpath = os.path.join(save_dir, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved → {fpath}")
    return fpath


# ──────────────────────────────────────────────
# Plotting: Confusion Matrix
# ──────────────────────────────────────────────
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: List[str], model_name: str,
                          save_dir: str = "outputs/confusion_matrix") -> str:
    """
    Plot and save a confusion matrix as a PNG.

    Args:
        y_true      : ground-truth labels
        y_pred      : predicted labels
        class_names : list of class name strings
        model_name  : used in title and filename
        save_dir    : output directory

    Returns:
        str: path to saved figure
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight="bold")

    ax.set_title(f"{model_name} – Confusion Matrix", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()

    fname = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    fpath = os.path.join(save_dir, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved → {fpath}")
    return fpath


# ──────────────────────────────────────────────
# Plotting: Model Comparison Bar Chart
# ──────────────────────────────────────────────
def plot_model_comparison(results: List[Dict], metric: str = "f1",
                          save_dir: str = "outputs/graphs") -> str:
    """
    Bar chart comparing multiple models on a given metric.

    Args:
        results    : list of dicts, each with 'model_name' and metric keys
        metric     : which metric to compare ('f1', 'accuracy', etc.)
        save_dir   : output directory

    Returns:
        str: path to saved figure
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    names  = [r["model_name"] for r in results]
    values = [r.get(metric, 0.0)  for r in results]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors[:len(names)], edgecolor="white",
                  linewidth=1.2, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    ax.set_ylim(0, min(1.0, max(values) + 0.12))
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f"Model Comparison – {metric.upper()}", fontsize=13,
                 fontweight="bold")
    ax.yaxis.grid(True, color=COLOUR_PALETTE["grid"], alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()

    fpath = os.path.join(save_dir, f"model_comparison_{metric}.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparison chart saved → {fpath}")
    return fpath


# ──────────────────────────────────────────────
# Results CSV
# ──────────────────────────────────────────────
def append_results_csv(result: dict,
                       csv_path: str = "outputs/results.csv") -> None:
    """
    Append one row of model results to the master CSV file.

    Args:
        result   : dict with model metrics + metadata
        csv_path : output CSV path
    """
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name", "accuracy", "precision", "recall", "f1",
        "train_time_s", "epochs", "best_val_loss", "num_params",
    ]
    write_header = not Path(csv_path).exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    logger.info(f"Results appended to {csv_path}")


def print_results_table(results: List[Dict]) -> None:
    """Pretty-print a comparison table to stdout."""
    if not results:
        return
    header = f"{'Model':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Params':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.get('model_name','?'):<25} "
              f"{r.get('accuracy',0):.4f}   "
              f"{r.get('precision',0):.4f}   "
              f"{r.get('recall',0):.4f}   "
              f"{r.get('f1',0):.4f}   "
              f"{r.get('num_params',0):>10,}")
    print("=" * len(header) + "\n")


# ──────────────────────────────────────────────
# Device helper
# ──────────────────────────────────────────────
def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return CUDA device if available and preferred, else CPU."""
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


# ──────────────────────────────────────────────
# Timer
# ──────────────────────────────────────────────
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start

    def __str__(self):
        m, s = divmod(self.elapsed, 60)
        return f"{int(m)}m {s:.1f}s"


# ──────────────────────────────────────────────
# Seed
# ──────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    logger.info(f"Random seed set to {seed}")
