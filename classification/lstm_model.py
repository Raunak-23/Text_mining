"""
models/lstm_model.py
--------------------
Bidirectional LSTM with attention for multi-label emotion classification.
Built with PyTorch.
"""

import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("LSTM using device: %s", DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# Attention module
# ─────────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """Single-head additive attention over LSTM hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden*2)
        energy = torch.tanh(self.attn(lstm_out))          # (batch, seq, hidden)
        scores = self.context(energy).squeeze(-1)          # (batch, seq)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, seq, 1)
        context = (weights * lstm_out).sum(dim=1)          # (batch, hidden*2)
        return context


# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM Emotion Classifier
# ─────────────────────────────────────────────────────────────────────────────

class BiLSTMEmotionClassifier(nn.Module):
    """
    Architecture
    ------------
    Embedding → BiLSTM (stacked) → Attention → Dropout → FC → Sigmoid
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_labels: int = 28,
        dropout: float = 0.4,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # Classifier head: hidden*2 (bidirectional) → num_labels
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len) long tensor of token indices

        Returns
        -------
        logits : (batch, num_labels)
        """
        embedded = self.dropout(self.embedding(x))          # (B, L, E)
        lstm_out, _ = self.lstm(embedded)                    # (B, L, H*2)
        context = self.attention(lstm_out)                   # (B, H*2)
        logits = self.fc(self.dropout(context))              # (B, num_labels)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids: np.ndarray, labels: np.ndarray):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model: nn.Module, loader, optimizer, criterion) -> float:
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion,
             threshold: float = 0.5) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(batch_y.cpu().numpy())
    return {
        "loss": total_loss / len(loader),
        "preds": np.vstack(all_preds),
        "labels": np.vstack(all_labels),
    }


def train_lstm(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 10,
    lr: float = 1e-3,
    patience: int = 3,
    save_path: Optional[Path] = None,
) -> dict:
    """
    Full training loop with early stopping.

    Returns
    -------
    history : dict of train_loss, val_loss per epoch
    """
    if save_path is None:
        save_path = MODELS_DIR / "lstm_best.pt"

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5, verbose=True
    )
    # Binary cross-entropy with pos_weight to handle class imbalance
    pos_weight = torch.ones(model.fc[-1].out_features).to(DEVICE) * 3.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    no_improve = 0

    model = model.to(DEVICE)

    for epoch in range(1, epochs + 1):
        t_loss = train_epoch(model, train_loader, optimizer, criterion)
        v_res = evaluate(model, val_loader, criterion)
        v_loss = v_res["loss"]

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        scheduler.step(v_loss)

        logger.info(
            "Epoch %2d/%2d  train_loss=%.4f  val_loss=%.4f",
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

    # Load best weights
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    logger.info("Best val_loss=%.4f  weights restored.", best_val)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    B, L, V, C = 32, 64, 5000, 28
    ids = rng.integers(0, V, (200, L)).astype(np.int64)
    lbs = (rng.random((200, C)) > 0.8).astype(np.float32)

    ds = EmotionDataset(ids, lbs)
    loader = torch.utils.data.DataLoader(ds, batch_size=B, shuffle=True)

    model = BiLSTMEmotionClassifier(V, embed_dim=64, hidden_dim=128,
                                    num_layers=2, num_labels=C)
    print(model)
    logits = model(torch.randint(0, V, (B, L)))
    print("Output shape:", logits.shape)
