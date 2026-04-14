"""
evaluate.py
===========
Standalone evaluation script for a trained checkpoint.
Computes full metrics, generates confusion matrix, classification report,
and prints comparison table if multiple model checkpoints exist.

Usage:
    # Evaluate a specific checkpoint:
    python src/evaluate.py --model cnn_lstm --ckpt outputs/checkpoints/cnn_lstm_best.pt

    # Evaluate all three models:
    python src/evaluate.py --model all

Author: Lab Project – Meme & Sarcasm Understanding
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from data_loader          import SampleDatasetLoader, MMSD2Loader
from preprocessing        import generate_sample_dataset
from model1_cnn_lstm      import build_cnn_lstm
from model2_clip          import build_clip_model
from model3_transformer   import build_visual_bert
from utils import (
    batch_predict, compute_metrics, plot_confusion_matrix,
    plot_model_comparison, print_results_table, append_results_csv,
    get_device, set_seed, logger,
)
from sklearn.metrics import classification_report

# ──────────────────────────────────────────────
# Config mirrors train.py
# ──────────────────────────────────────────────
CONFIGS = {
    "cnn_lstm": {
        "num_classes"  : 2,
        "img_embed_dim": 256,
        "txt_embed_dim": 256,
        "hidden_dim"   : 256,
        "dropout"      : 0.4,
        "vocab_size"   : 30522,
        "freeze_cnn"   : False,
    },
    "clip": {
        "num_classes"   : 2,
        "embed_dim"     : 512,
        "hidden_dim"    : 512,
        "dropout"       : 0.3,
        "use_pretrained": False,
        "freeze_clip"   : False,
    },
    "vbert": {
        "num_classes"  : 2,
        "d_model"      : 256,
        "nhead"        : 8,
        "num_layers"   : 4,
        "vocab_size"   : 30522,
        "max_text_len" : 64,
        "dropout"      : 0.1,
        "freeze_cnn"   : False,
    },
}

MODEL_BUILDERS = {
    "cnn_lstm": (build_cnn_lstm,   "CNN_LSTM"),
    "clip"    : (build_clip_model, "CLIP"),
    "vbert"   : (build_visual_bert,"VisualBERT_Lite"),
}

CKPT_DIR    = Path("outputs/checkpoints")
CLASS_NAMES = ["Not Sarcastic", "Sarcastic"]


# ──────────────────────────────────────────────
# Evaluate one model
# ──────────────────────────────────────────────
def evaluate_model(model_key: str, ckpt_path: str,
                   test_loader, device: torch.device) -> dict:
    """
    Load a checkpoint and compute full test-set metrics.

    Args:
        model_key   : 'cnn_lstm' | 'clip' | 'vbert'
        ckpt_path   : path to .pt checkpoint file
        test_loader : DataLoader for the test split
        device      : torch.device

    Returns:
        dict of metrics
    """
    builder, model_name = MODEL_BUILDERS[model_key]
    model = builder(CONFIGS[model_key].copy()).to(device)

    # Load checkpoint
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint: {ckpt_path}")
    else:
        logger.warning(
            f"No checkpoint at {ckpt_path}. Evaluating with random weights."
        )

    # Inference
    y_true, y_pred, y_probs = batch_predict(model, test_loader, device)

    # Metrics
    metrics = compute_metrics(y_true, y_pred, average="binary")
    metrics["model_name"] = model_name
    metrics["num_params"] = model.count_parameters()

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )

    # Per-class metrics
    per_class = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
    )

    logger.info(f"\n{'='*55}")
    logger.info(f"  Model     : {model_name}")
    logger.info(f"  Accuracy  : {metrics['accuracy']:.4f}")
    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  Recall    : {metrics['recall']:.4f}")
    logger.info(f"  F1        : {metrics['f1']:.4f}")
    logger.info(f"{'='*55}")
    logger.info("\n" + report)

    # Confusion matrix plot
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, model_name)

    # Save detailed metrics JSON
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"{model_key}_eval_metrics.json", "w") as f:
        json.dump({
            "metrics"      : metrics,
            "per_class"    : per_class,
            "class_names"  : CLASS_NAMES,
            "n_test"       : len(y_true),
            "checkpoint"   : str(ckpt_path),
        }, f, indent=2)

    # Append to master CSV
    append_results_csv(metrics)

    return metrics


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate meme/sarcasm detection models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",      type=str, default="all",
                        choices=["cnn_lstm", "clip", "vbert", "all"])
    parser.add_argument("--ckpt",       type=str, default=None,
                        help="Path to checkpoint (single model only)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_mmsd2",  action="store_true")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--no_gpu",     action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(prefer_gpu=not args.no_gpu)

    # Dataset
    if not Path("data/sample_dataset/metadata.json").exists():
        generate_sample_dataset()

    if args.use_mmsd2:
        data_cls = MMSD2Loader(batch_size=args.batch_size)
    else:
        data_cls = SampleDatasetLoader(batch_size=args.batch_size)

    _, _, test_loader = data_cls.get_loaders()

    keys = ["cnn_lstm", "clip", "vbert"] if args.model == "all" else [args.model]
    all_results = []

    for key in keys:
        if args.ckpt and args.model != "all":
            ckpt = args.ckpt
        else:
            ckpt = str(CKPT_DIR / f"{key}_best.pt")

        result = evaluate_model(key, ckpt, test_loader, device)
        all_results.append(result)

    # Summary
    print_results_table(all_results)
    if len(all_results) > 1:
        for metric in ("accuracy", "f1", "precision", "recall"):
            plot_model_comparison(all_results, metric=metric)
        logger.info("Comparison charts saved to outputs/graphs/")


if __name__ == "__main__":
    main()
