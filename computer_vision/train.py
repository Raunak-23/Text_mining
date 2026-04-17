"""
train.py
========
Training pipeline for all three models.

Usage:
    python src/train.py --model cnn_lstm --epochs 10 --batch_size 16
    python src/train.py --model clip     --epochs 10 --batch_size 8
    python src/train.py --model vbert    --epochs 10 --batch_size 8
    python src/train.py --model all      --epochs 10   # trains all 3

Author: Lab Project – Meme & Sarcasm Understanding
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add src/ to path so imports work when called from project root
sys.path.insert(0, str(Path(__file__).parent))

from data_loader   import SampleDatasetLoader, MMSD2Loader
from preprocessing import generate_sample_dataset
from model1_cnn_lstm    import build_cnn_lstm
from model2_clip         import build_clip_model
from model3_transformer  import build_visual_bert
from utils import (
    AverageMeter, EarlyStopping, EarlyStopping,
    save_checkpoint, compute_metrics, plot_training_curves,
    plot_confusion_matrix, append_results_csv, print_results_table,
    batch_predict, get_device, set_seed, Timer, logger,
    plot_model_comparison,
)

# ──────────────────────────────────────────────
# Default configurations
# ──────────────────────────────────────────────
CONFIGS = {
    "cnn_lstm": {
        "model_name"   : "CNN_LSTM",
        "num_classes"  : 2,
        "img_embed_dim": 256,
        "txt_embed_dim": 256,
        "hidden_dim"   : 256,
        "dropout"      : 0.4,
        "vocab_size"   : 30522,
        "freeze_cnn"   : False,
    },
    "clip": {
        "model_name"   : "CLIP",
        "num_classes"  : 2,
        "embed_dim"    : 512,
        "hidden_dim"   : 512,
        "dropout"      : 0.3,
        "use_pretrained": False,   # Set True if open_clip installed
        "freeze_clip"  : False,
    },
    "vbert": {
        "model_name"   : "VisualBERT_Lite",
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
    "cnn_lstm": build_cnn_lstm,
    "clip"    : build_clip_model,
    "vbert"   : build_visual_bert,
}

CKPT_DIR   = Path("outputs/checkpoints")
RESULTS_CSV = "outputs/results.csv"


# ──────────────────────────────────────────────
# Single epoch: Train
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Run one training epoch.

    Returns:
        avg_loss (float), avg_acc (float)
    """
    model.train()
    loss_meter = AverageMeter("train_loss")
    correct    = 0
    total      = 0

    for batch_idx, batch in enumerate(loader):
        images  = batch["image"].to(device)
        ids     = batch["input_ids"].to(device)
        masks   = batch["attention_mask"].to(device)
        labels  = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images, ids, masks)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping to stabilise transformer training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds  = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        loss_meter.update(loss.item(), labels.size(0))

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"  Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(loader)} | "
                f"Loss {loss_meter.avg:.4f} | Acc {correct/total:.4f}"
            )

    return loss_meter.avg, correct / max(total, 1)


# ──────────────────────────────────────────────
# Single epoch: Validate
# ──────────────────────────────────────────────
def validate(model, loader, criterion, device):
    """
    Run validation.

    Returns:
        avg_loss, avg_acc, y_true, y_pred
    """
    model.eval()
    loss_meter = AverageMeter("val_loss")
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            ids    = batch["input_ids"].to(device)
            masks  = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(images, ids, masks)
            loss   = criterion(logits, labels)

            preds  = logits.argmax(dim=-1)
            loss_meter.update(loss.item(), labels.size(0))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    import numpy as np
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
    return loss_meter.avg, acc, np.array(all_labels), np.array(all_preds)


# ──────────────────────────────────────────────
# Full Training Loop
# ──────────────────────────────────────────────
def train_model(model_key: str, args: argparse.Namespace,
                train_loader, val_loader, test_loader,
                device: torch.device) -> dict:
    """
    Complete training loop for a single model.

    Returns:
        dict of final evaluation results
    """
    cfg        = CONFIGS[model_key].copy()
    model_name = cfg.pop("model_name")
    model      = MODEL_BUILDERS[model_key](cfg).to(device)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Training {model_name}")
    logger.info(f"  Parameters: {model.count_parameters():,}")
    logger.info(f"{'='*60}")

    # Criterion
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer
    optimizer  = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )

    # LR Scheduler
    scheduler  = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Early stopping
    stopper    = EarlyStopping(patience=args.patience, mode="min")

    history    = {"train_loss": [], "val_loss": [],
                  "train_acc" : [], "val_acc" : []}
    best_ckpt  = CKPT_DIR / f"{model_key}_best.pt"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    with Timer() as timer:
        for epoch in range(1, args.epochs + 1):
            # ── Train ───────────────────────────
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )

            # ── Validate ─────────────────────────
            vl_loss, vl_acc, _, _ = validate(
                model, val_loader, criterion, device
            )
            scheduler.step()

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(vl_acc)

            logger.info(
                f"[{model_name}] Epoch {epoch:3d}/{args.epochs} | "
                f"Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} | "
                f"Val Loss={vl_loss:.4f} Acc={vl_acc:.4f}"
            )

            # ── Save best checkpoint ───────────────
            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                save_checkpoint({
                    "epoch"              : epoch,
                    "model_state_dict"   : model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss"      : best_val_loss,
                    "history"            : history,
                }, str(best_ckpt))

            # ── Early stopping ─────────────────────
            if stopper(vl_loss, epoch):
                logger.info(f"[{model_name}] Early stopping at epoch {epoch} "
                             f"(best epoch: {stopper.best_epoch})")
                break

    # ── Save training curves ────────────────────
    plot_training_curves(history, model_name)

    # ── Test set evaluation ─────────────────────
    logger.info(f"\n[{model_name}] Evaluating on test set …")
    y_true, y_pred, _ = batch_predict(model, test_loader, device)

    metrics = compute_metrics(y_true, y_pred)
    metrics["model_name"]    = model_name
    metrics["train_time_s"]  = round(timer.elapsed, 1)
    metrics["epochs"]        = epoch
    metrics["best_val_loss"] = round(best_val_loss, 4)
    metrics["num_params"]    = model.count_parameters()

    # ── Confusion matrix ────────────────────────
    plot_confusion_matrix(
        y_true, y_pred,
        class_names=["Not Sarcastic", "Sarcastic"],
        model_name=model_name,
    )

    # ── Save history JSON ───────────────────────
    hist_path = f"outputs/graphs/{model_key}_history.json"
    Path("outputs/graphs").mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w") as f:
        json.dump(history, f)

    # ── Append to results CSV ───────────────────
    append_results_csv(metrics)

    logger.info(
        f"[{model_name}] DONE | "
        f"Acc={metrics['accuracy']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"Time={timer}"
    )
    return metrics


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train meme/sarcasm detection models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",       type=str, default="all",
                        choices=["cnn_lstm", "clip", "vbert", "all"],
                        help="Which model to train")
    parser.add_argument("--epochs",      type=int, default=10)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--patience",    type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--use_mmsd2",   action="store_true",
                        help="Use full MMSD 2.0 dataset (else sample)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--no_gpu",      action="store_true",
                        help="Force CPU even if GPU is available")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(prefer_gpu=not args.no_gpu)

    # ── Ensure sample dataset exists ────────────
    sample_meta = Path("data/sample_dataset/metadata.json")
    if not sample_meta.exists():
        logger.info("Generating sample dataset …")
        generate_sample_dataset(n_samples=200, seed=args.seed)

    # ── DataLoaders (no tokenizer – vocab-based) ─
    if args.use_mmsd2:
        data_loader_cls = MMSD2Loader(batch_size=args.batch_size)
    else:
        data_loader_cls = SampleDatasetLoader(batch_size=args.batch_size)

    train_loader, val_loader, test_loader = data_loader_cls.get_loaders()

    # ── Choose which models to train ────────────
    keys = ["cnn_lstm", "clip", "vbert"] if args.model == "all" else [args.model]
    all_results = []

    for key in keys:
        result = train_model(key, args, train_loader, val_loader,
                             test_loader, device)
        all_results.append(result)

    # ── Print comparison table ───────────────────
    print_results_table(all_results)

    # ── Comparison chart ─────────────────────────
    if len(all_results) > 1:
        for metric in ("accuracy", "f1", "precision", "recall"):
            plot_model_comparison(all_results, metric=metric)
        logger.info("Model comparison charts saved to outputs/graphs/")

    logger.info("Training complete. Results saved to outputs/results.csv")


if __name__ == "__main__":
    main()
