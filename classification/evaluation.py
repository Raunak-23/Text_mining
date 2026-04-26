"""
evaluation.py
-------------
Multi-label classification metrics, confusion matrices, and result tables.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, hamming_loss,
    multilabel_confusion_matrix,
    classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR    = ROOT_DIR / "results"
FIGURES_DIR    = RESULTS_DIR / "figures"
METRICS_DIR    = RESULTS_DIR / "metrics"
for p in [FIGURES_DIR, METRICS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    label_names: Optional[List[str]] = None) -> dict:
    """
    Compute a comprehensive set of multi-label classification metrics.

    Returns
    -------
    dict with scalar and per-label metrics
    """
    assert y_true.shape == y_pred.shape, "Shape mismatch"

    results = {
        # Subset accuracy: fraction of samples where ALL labels match
        "subset_accuracy":    accuracy_score(y_true, y_pred),
        "hamming_loss":       hamming_loss(y_true, y_pred),
        "f1_micro":           f1_score(y_true, y_pred, average="micro",    zero_division=0),
        "f1_macro":           f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_samples":         f1_score(y_true, y_pred, average="samples",  zero_division=0),
        "precision_micro":    precision_score(y_true, y_pred, average="micro",    zero_division=0),
        "precision_macro":    precision_score(y_true, y_pred, average="macro",    zero_division=0),
        "recall_micro":       recall_score(y_true, y_pred, average="micro",       zero_division=0),
        "recall_macro":       recall_score(y_true, y_pred, average="macro",       zero_division=0),
    }

    # Per-label metrics
    if label_names is not None:
        per_label_f1  = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_label_pre = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_label_rec = recall_score(y_true, y_pred, average=None, zero_division=0)

        n = min(len(label_names), y_true.shape[1])
        results["per_label"] = pd.DataFrame({
            "label":     label_names[:n],
            "f1":        per_label_f1[:n],
            "precision": per_label_pre[:n],
            "recall":    per_label_rec[:n],
            "support":   y_true[:, :n].sum(axis=0),
        }).set_index("label")

    return results


def print_metrics(metrics: dict, model_name: str = "Model") -> None:
    """Pretty-print the scalar metrics."""
    print(f"\n{'─'*50}")
    print(f"  {model_name} Evaluation Results")
    print(f"{'─'*50}")
    scalar_keys = [k for k in metrics if k != "per_label"]
    for k in scalar_keys:
        print(f"  {k:<25s}: {metrics[k]:.4f}")
    if "per_label" in metrics:
        print(f"\n  Per-Label Metrics (top 10 by F1):")
        top = metrics["per_label"].sort_values("f1", ascending=False).head(10)
        print(top.to_string())
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison table
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(results_dict: dict) -> pd.DataFrame:
    """
    Parameters
    ----------
    results_dict : {model_name: metrics_dict}

    Returns
    -------
    DataFrame indexed by model name with key metrics as columns.
    """
    rows = []
    for model, m in results_dict.items():
        rows.append({
            "Model":            model,
            "F1 (micro)":       round(m.get("f1_micro",       0), 4),
            "F1 (macro)":       round(m.get("f1_macro",       0), 4),
            "F1 (weighted)":    round(m.get("f1_weighted",    0), 4),
            "Precision (micro)":round(m.get("precision_micro",0), 4),
            "Recall (micro)":   round(m.get("recall_micro",   0), 4),
            "Subset Acc.":      round(m.get("subset_accuracy",0), 4),
            "Hamming Loss":     round(m.get("hamming_loss",   0), 4),
        })
    df = pd.DataFrame(rows).set_index("Model")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_label_distribution(y: np.ndarray, label_names: List[str],
                             title: str = "Label Distribution",
                             save: bool = True) -> None:
    """Bar chart of label frequency."""
    counts = y.sum(axis=0)
    order  = np.argsort(counts)[::-1]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(label_names)),
           counts[order],
           color=sns.color_palette("viridis", len(label_names)))
    ax.set_xticks(range(len(label_names)))
    ax.set_xticklabels([label_names[i] for i in order], rotation=45, ha="right")
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Count")
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "label_distribution.png"
        fig.savefig(path, dpi=150)
        logger.info("Saved label distribution → %s", path)
    plt.show()
    plt.close(fig)


def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray,
                             label_names: List[str],
                             top_n: int = 9,
                             save: bool = True) -> None:
    """
    Plot per-label binary confusion matrices (top-N by support).
    """
    supports = y_true.sum(axis=0)
    top_idx  = np.argsort(supports)[::-1][:top_n]

    mlcm = multilabel_confusion_matrix(y_true, y_pred)

    rows = int(np.ceil(top_n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(14, rows * 4))
    axes = axes.flatten()

    for plot_i, label_i in enumerate(top_idx):
        cm = mlcm[label_i]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[plot_i],
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        axes[plot_i].set_title(label_names[label_i], fontsize=11)

    # Hide unused axes
    for j in range(plot_i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Per-Label Confusion Matrices (Top 9 by Support)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        path = FIGURES_DIR / "confusion_matrices.png"
        fig.savefig(path, dpi=150)
        logger.info("Saved confusion matrices → %s", path)
    plt.show()
    plt.close(fig)


def plot_performance_comparison(comparison_df: pd.DataFrame,
                                 save: bool = True) -> None:
    """Grouped bar chart comparing models across key metrics."""
    metrics_to_plot = ["F1 (micro)", "F1 (macro)", "Precision (micro)", "Recall (micro)"]
    df_plot = comparison_df[metrics_to_plot]

    ax = df_plot.plot(kind="bar", figsize=(10, 5), colormap="tab10", edgecolor="white")
    ax.set_title("Model Performance Comparison", fontsize=14)
    ax.set_ylabel("Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "model_comparison.png"
        plt.savefig(path, dpi=150)
        logger.info("Saved model comparison → %s", path)
    plt.show()
    plt.close()


def plot_training_history(history: dict, model_name: str = "Model",
                           save: bool = True) -> None:
    """Plot train/val loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="Train Loss", marker="o")
    ax.plot(history["val_loss"],   label="Val Loss",   marker="s")
    ax.set_title(f"{model_name} – Training History", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        fname = model_name.lower().replace(" ", "_") + "_training_history.png"
        path = FIGURES_DIR / fname
        fig.savefig(path, dpi=150)
        logger.info("Saved training history → %s", path)
    plt.show()
    plt.close(fig)


def plot_per_label_f1(metrics: dict, model_name: str = "Model",
                      save: bool = True) -> None:
    """Horizontal bar chart of per-label F1 scores."""
    if "per_label" not in metrics:
        return
    df = metrics["per_label"].sort_values("f1", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.35)))
    ax.barh(df.index, df["f1"], color=sns.color_palette("rocket", len(df)))
    ax.set_title(f"{model_name} – Per-Label F1 Scores", fontsize=13)
    ax.set_xlabel("F1 Score")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save:
        fname = model_name.lower().replace(" ", "_") + "_per_label_f1.png"
        path = FIGURES_DIR / fname
        fig.savefig(path, dpi=150)
        logger.info("Saved per-label F1 → %s", path)
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Save / load utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results_dict: dict, fname: str = "all_results.csv") -> None:
    table = build_comparison_table(results_dict)
    path = METRICS_DIR / fname
    table.to_csv(path)
    logger.info("Saved comparison table → %s", path)
    print(table.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_samples, n_labels = 100, 10
    label_names = [f"emotion_{i}" for i in range(n_labels)]

    y_true = (rng.random((n_samples, n_labels)) > 0.7).astype(int)
    y_pred = (rng.random((n_samples, n_labels)) > 0.6).astype(int)

    m = compute_metrics(y_true, y_pred, label_names)
    print_metrics(m, "TestModel")
    plot_label_distribution(y_true, label_names, save=False)
    plot_confusion_matrices(y_true, y_pred, label_names, top_n=6, save=False)
