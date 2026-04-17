"""
model1_cnn_lstm.py
==================
Model 1: CNN + LSTM Multimodal Fusion Baseline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture:
  ┌──────────────────┐    ┌─────────────────────────┐
  │   IMAGE BRANCH   │    │      TEXT BRANCH         │
  │                  │    │                          │
  │  ResNet-18       │    │  Embedding → LSTM        │
  │  (pretrained)    │    │  → Mean-Pool             │
  │  → FC(256)       │    │  → FC(256)               │
  └────────┬─────────┘    └────────────┬─────────────┘
           │                           │
           └──────── Concat ───────────┘
                         │
                    FC(256) → ReLU → Dropout
                         │
                    FC(128) → ReLU → Dropout
                         │
                    FC(2)  → Softmax

Author: Lab Project – Meme & Sarcasm Understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ──────────────────────────────────────────────
# Sub-module: CNN Image Encoder
# ──────────────────────────────────────────────
class CNNImageEncoder(nn.Module):
    """
    Uses ResNet-18 (pretrained on ImageNet) as the image backbone.
    The final classification head is replaced with a projection layer.

    Args:
        output_dim (int): dimension of the projected image embedding
        freeze_base (bool): if True, freeze ResNet-18 backbone weights
    """

    def __init__(self, output_dim: int = 256, freeze_base: bool = False):
        super().__init__()
        # Load pretrained ResNet-18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove the final fully-connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # → (B, 512, 1, 1)

        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Project 512-d feature to output_dim
        self.projector = nn.Sequential(
            nn.Flatten(),                          # (B, 512)
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image tensor of shape (B, 3, H, W)
        Returns:
            image embedding (B, output_dim)
        """
        feat = self.backbone(x)          # (B, 512, 1, 1)
        return self.projector(feat)      # (B, output_dim)


# ──────────────────────────────────────────────
# Sub-module: LSTM Text Encoder
# ──────────────────────────────────────────────
class LSTMTextEncoder(nn.Module):
    """
    Embeds token IDs → LSTM → mean-pools the hidden states.

    Args:
        vocab_size   (int): vocabulary size (used for embedding)
        embed_dim    (int): embedding dimension
        hidden_dim   (int): LSTM hidden units
        output_dim   (int): projection dimension
        num_layers   (int): number of LSTM layers
        dropout      (float): dropout rate between LSTM layers
        bidirectional (bool): bidirectional LSTM flag
        padding_idx  (int): index for <PAD> token
    """

    def __init__(self, vocab_size: int = 30522, embed_dim: int = 128,
                 hidden_dim: int = 256, output_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = True, padding_idx: int = 0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                      padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.projector = nn.Sequential(
            nn.Linear(lstm_out_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids      : (B, seq_len)  token indices
            attention_mask : (B, seq_len)  1=real token, 0=padding

        Returns:
            text embedding (B, output_dim)
        """
        x, _ = self.lstm(self.embedding(input_ids))    # (B, seq_len, lstm_out)

        # Masked mean-pooling: ignore PAD positions
        mask  = attention_mask.unsqueeze(-1).float()   # (B, seq_len, 1)
        x     = (x * mask).sum(dim=1)                 # (B, lstm_out)
        denom = mask.sum(dim=1).clamp(min=1e-6)       # (B, 1)
        x     = x / denom

        return self.projector(x)                       # (B, output_dim)


# ──────────────────────────────────────────────
# Full Model: CNN + LSTM Fusion
# ──────────────────────────────────────────────
class CNNLSTMModel(nn.Module):
    """
    Multimodal Sarcasm/Meme Classifier combining CNN (image) + LSTM (text).

    Fusion strategy: concatenation of visual and textual embeddings,
    followed by a two-layer MLP classifier.

    Args:
        num_classes   (int): 2 for binary sarcasm detection
        img_embed_dim (int): CNN output dimension
        txt_embed_dim (int): LSTM output dimension
        hidden_dim    (int): MLP hidden dimension
        dropout       (float): dropout rate in the classifier head
        vocab_size    (int): text vocabulary size
        freeze_cnn    (bool): whether to freeze the ResNet-18 backbone
    """

    def __init__(self, num_classes: int = 2,
                 img_embed_dim: int = 256,
                 txt_embed_dim: int = 256,
                 hidden_dim: int = 256,
                 dropout: float = 0.4,
                 vocab_size: int = 30522,
                 freeze_cnn: bool = False):
        super().__init__()
        self.model_name = "CNN_LSTM"

        # Encoders
        self.image_encoder = CNNImageEncoder(output_dim=img_embed_dim,
                                             freeze_base=freeze_cnn)
        self.text_encoder  = LSTMTextEncoder(vocab_size=vocab_size,
                                             output_dim=txt_embed_dim)

        fusion_dim = img_embed_dim + txt_embed_dim   # concatenated

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialisation for linear layers in the classifier."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image          : (B, 3, H, W)
            input_ids      : (B, seq_len)
            attention_mask : (B, seq_len)

        Returns:
            logits (B, num_classes)
        """
        img_feat  = self.image_encoder(image)                          # (B, 256)
        txt_feat  = self.text_encoder(input_ids, attention_mask)       # (B, 256)
        fused     = torch.cat([img_feat, txt_feat], dim=-1)            # (B, 512)
        logits    = self.classifier(fused)                             # (B, 2)
        return logits

    def get_embeddings(self, image, input_ids, attention_mask):
        """Return intermediate fused embedding (useful for analysis)."""
        img_feat = self.image_encoder(image)
        txt_feat = self.text_encoder(input_ids, attention_mask)
        return torch.cat([img_feat, txt_feat], dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────
# Factory function
# ──────────────────────────────────────────────
def build_cnn_lstm(config: dict) -> CNNLSTMModel:
    """
    Construct CNNLSTMModel from a configuration dictionary.

    Example config:
        {
            "num_classes"   : 2,
            "img_embed_dim" : 256,
            "txt_embed_dim" : 256,
            "hidden_dim"    : 256,
            "dropout"       : 0.4,
            "vocab_size"    : 30522,
            "freeze_cnn"    : False,
        }
    """
    return CNNLSTMModel(**config)


# ──────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    model = CNNLSTMModel()
    print(f"Model: {model.model_name}")
    print(f"Trainable parameters: {model.count_parameters():,}")

    # Dummy forward pass
    B, SEQ = 4, 64
    imgs   = torch.randn(B, 3, 224, 224)
    ids    = torch.randint(0, 30522, (B, SEQ))
    mask   = torch.ones(B, SEQ, dtype=torch.long)

    with torch.no_grad():
        logits = model(imgs, ids, mask)

    print(f"Input image  : {imgs.shape}")
    print(f"Input ids    : {ids.shape}")
    print(f"Output logits: {logits.shape}")     # Expected: (4, 2)
    probs = F.softmax(logits, dim=-1)
    print(f"Probabilities: {probs}")
