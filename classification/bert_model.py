"""
models/bert_model.py
--------------------
BERT-based multi-label emotion classifier using HuggingFace Transformers.
Fine-tunes bert-base-uncased (or any compatible checkpoint) with a
multi-label classification head.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("BERT using device: %s", DEVICE)

DEFAULT_CHECKPOINT = "bert-base-uncased"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BERTEmotionDataset(Dataset):
    """
    Tokenises raw text strings on-the-fly and stores labels.
    """

    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer,
        max_len: int = 128,
    ):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         self.labels[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class BERTEmotionClassifier(nn.Module):
    """
    Architecture
    ------------
    BERT → [CLS] embedding → Dropout → Linear(768, num_labels)

    Note: We use the [CLS] token representation as the sentence embedding.
    """

    def __init__(
        self,
        num_labels: int = 28,
        checkpoint: str = DEFAULT_CHECKPOINT,
        dropout: float = 0.3,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(checkpoint)
        hidden_size = self.bert.config.hidden_size

        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = False
            logger.info("BERT base weights frozen.")

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use the [CLS] token embedding (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_pos_weight(y: np.ndarray, device) -> torch.Tensor:
    """Compute per-label pos_weight = (neg_count / pos_count) clipped to [1, 20]."""
    pos = y.sum(axis=0) + 1e-6
    neg = (1 - y).sum(axis=0) + 1e-6
    pw = np.clip(neg / pos, 1.0, 20.0)
    return torch.tensor(pw, dtype=torch.float32).to(device)


def train_epoch_bert(model, loader, optimizer, scheduler, criterion) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        lbls  = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = criterion(logits, lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_bert(model, loader, criterion, threshold: float = 0.5) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        lbls  = batch["labels"].to(DEVICE)

        logits = model(ids, mask)
        loss = criterion(logits, lbls)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(lbls.cpu().numpy())

    return {
        "loss":   total_loss / len(loader),
        "preds":  np.vstack(all_preds),
        "labels": np.vstack(all_labels),
    }


def train_bert(
    model: BERTEmotionClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train: np.ndarray,
    epochs: int = 3,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    patience: int = 2,
    save_path: Optional[Path] = None,
) -> dict:
    """
    Fine-tunes BERT with linear warmup scheduler + early stopping.

    Returns
    -------
    history : dict
    """
    if save_path is None:
        save_path = MODELS_DIR / "bert_best.pt"

    pos_weight = _build_pos_weight(y_train, DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01
    )
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model = model.to(DEVICE)
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t_loss = train_epoch_bert(model, train_loader, optimizer,
                                  scheduler, criterion)
        v_res = evaluate_bert(model, val_loader, criterion)
        v_loss = v_res["loss"]

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
            epoch, epochs, t_loss, v_loss
        )

        if v_loss < best_val:
            best_val = v_loss
            no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d.", epoch)
                break

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    logger.info("Best val_loss=%.4f  weights restored.", best_val)
    return history


def build_bert_loaders(
    train_texts, train_labels,
    val_texts,   val_labels,
    test_texts,  test_labels,
    tokenizer,
    max_len: int = 128,
    batch_size: int = 32,
) -> tuple:
    train_ds = BERTEmotionDataset(list(train_texts), train_labels, tokenizer, max_len)
    val_ds   = BERTEmotionDataset(list(val_texts),   val_labels,   tokenizer, max_len)
    test_ds  = BERTEmotionDataset(list(test_texts),  test_labels,  tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=DEVICE.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = BERTEmotionClassifier(num_labels=28, freeze_base=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_CHECKPOINT)
    texts = ["I am so happy!", "This is terrible and frightening."]
    labels = np.zeros((2, 28), dtype=np.float32)
    labels[0, 17] = 1   # joy
    labels[1, 14] = 1   # fear

    ds = BERTEmotionDataset(texts, labels, tok, max_len=32)
    loader = DataLoader(ds, batch_size=2)

    model = model.to(DEVICE)
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        out  = model(ids, mask)
        print("BERT output shape:", out.shape)
        break
