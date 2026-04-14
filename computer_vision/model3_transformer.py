"""
model3_transformer.py
=====================
Model 3: Lightweight VisualBERT-style Multimodal Transformer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │                   Visual Token Extraction                       │
  │  Image (3×224×224)  →  ResNet-18 Grid Features (7×7×512)       │
  │                     →  FC projection  →  49 visual tokens (256-d)│
  └─────────────────────────────┬───────────────────────────────────┘
                                │
  ┌─────────────────────────────┼───────────────────────────────────┐
  │                   Token Sequence Assembly                       │
  │  [CLS] [T_1 .. T_L] [SEP]   │ [V_1 .. V_49]                   │
  │   ↑  text tokens (BERT)     ↑  visual tokens                   │
  └─────────────────────────────┴───────────────────────────────────┘
                                │
  ┌─────────────────────────────▼───────────────────────────────────┐
  │              Joint Multimodal Transformer Encoder               │
  │  4 layers  ×  {Multi-Head Attention (8 heads) + FFN}            │
  │  d_model = 256  |  Positional + Segment (text/visual) Embeddings│
  └─────────────────────────────┬───────────────────────────────────┘
                                │
                        CLS token output
                                │
                     MLP Classifier Head
                     FC(256) → GELU → Dropout → FC(2)

Author: Lab Project – Meme & Sarcasm Understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
TEXT_SEG_ID   = 0    # Segment token-type ID for text tokens
VISUAL_SEG_ID = 1    # Segment token-type ID for visual tokens


# ──────────────────────────────────────────────
# Visual Region Feature Extractor
# ──────────────────────────────────────────────
class VisualFeatureExtractor(nn.Module):
    """
    Extracts a grid of visual region features using ResNet-18.

    ResNet-18 up to the 4th block produces a 7×7 spatial feature map
    (for 224×224 input).  Each spatial location becomes a visual token.

    Args:
        out_dim (int): projection dimension per visual token
        freeze  (bool): freeze the ResNet backbone
    """

    def __init__(self, out_dim: int = 256, freeze: bool = False):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Keep everything up to (not including) avgpool
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )  # Output: (B, 512, 7, 7)  for 224×224 input

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.projector = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, 224, 224)
        Returns:
            visual_tokens : (B, 49, out_dim)
        """
        feat = self.backbone(x)                          # (B, 512, 7, 7)
        B, C, H, W = feat.shape
        feat = feat.flatten(2).transpose(1, 2)           # (B, 49, 512)
        return self.projector(feat)                      # (B, 49, out_dim)


# ──────────────────────────────────────────────
# Segment Embeddings (text vs visual)
# ──────────────────────────────────────────────
class SegmentEmbedding(nn.Module):
    def __init__(self, num_segments: int = 2, embed_dim: int = 256):
        super().__init__()
        self.seg_emb = nn.Embedding(num_segments, embed_dim)

    def forward(self, segment_ids: torch.Tensor) -> torch.Tensor:
        return self.seg_emb(segment_ids)


# ──────────────────────────────────────────────
# Positional Encoding (learned)
# ──────────────────────────────────────────────
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int = 200, embed_dim: int = 256):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, embed_dim)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device).unsqueeze(0)   # (1, L)
        return self.pos_emb(pos)                                   # (1, L, D)


# ──────────────────────────────────────────────
# Full VisualBERT-style Model
# ──────────────────────────────────────────────
class VisualBERTModel(nn.Module):
    """
    Lightweight VisualBERT-style multimodal transformer for sarcasm detection.

    The model:
        1. Extracts 49 visual region tokens from the image.
        2. Uses token embeddings for the text input.
        3. Prepends a [CLS] token, appends [SEP], then appends visual tokens.
        4. Adds positional + segment embeddings.
        5. Runs a 4-layer multi-head transformer encoder.
        6. Classifies from the final [CLS] representation.

    Args:
        num_classes  (int)   : output classes
        d_model      (int)   : transformer dimension
        nhead        (int)   : attention heads
        num_layers   (int)   : transformer encoder layers
        vocab_size   (int)   : text vocabulary size
        max_text_len (int)   : max text token length
        dropout      (float) : dropout rate
        freeze_cnn   (bool)  : freeze visual backbone
    """

    def __init__(self, num_classes: int = 2,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 vocab_size: int = 30522,
                 max_text_len: int = 64,
                 dropout: float = 0.1,
                 freeze_cnn: bool = False):
        super().__init__()
        self.model_name  = "VisualBERT_Lite"
        self.d_model     = d_model
        self.max_text_len = max_text_len

        # ── Token Embedding for text ─────────────────
        self.token_emb   = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc     = LearnedPositionalEncoding(max_len=max_text_len + 49 + 3,
                                                     embed_dim=d_model)
        self.seg_enc     = SegmentEmbedding(num_segments=2, embed_dim=d_model)
        self.input_norm  = nn.LayerNorm(d_model)
        self.input_drop  = nn.Dropout(dropout)

        # ── Visual feature extractor ──────────────────
        self.visual_extractor = VisualFeatureExtractor(out_dim=d_model,
                                                       freeze=freeze_cnn)

        # ── Special tokens (learnable) ────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.sep_token, std=0.02)

        # ── Transformer encoder ───────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── Classifier head ───────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _build_sequence(self, text_tokens, visual_tokens):
        """
        Assemble joint token sequence and segment IDs.

        Sequence:  [CLS] [text_tokens] [SEP] [visual_tokens]
        Segments:    0        0          0         1

        Args:
            text_tokens   : (B, L, D)
            visual_tokens : (B, 49, D)

        Returns:
            seq       : (B, 1+L+1+49, D)
            seg_ids   : (B, 1+L+1+49)  long tensor
        """
        B = text_tokens.size(0)

        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
        sep = self.sep_token.expand(B, -1, -1)   # (B, 1, D)

        seq = torch.cat([cls, text_tokens, sep, visual_tokens], dim=1)

        L_text = text_tokens.size(1)
        L_vis  = visual_tokens.size(1)
        total  = 1 + L_text + 1 + L_vis

        seg_ids = torch.cat([
            torch.full((B, 1 + L_text + 1), TEXT_SEG_ID,   dtype=torch.long),
            torch.full((B, L_vis),           VISUAL_SEG_ID, dtype=torch.long),
        ], dim=1).to(text_tokens.device)

        return seq, seg_ids, total

    def forward(self, image: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image          : (B, 3, H, W)
            input_ids      : (B, text_len)
            attention_mask : (B, text_len)  1=real, 0=pad

        Returns:
            logits (B, num_classes)
        """
        B, text_len = input_ids.shape

        # ── 1. Text token embeddings ─────────────────
        text_tokens = self.token_emb(input_ids)         # (B, L, D)

        # ── 2. Visual tokens ──────────────────────────
        visual_tokens = self.visual_extractor(image)    # (B, 49, D)

        # ── 3. Assemble sequence ──────────────────────
        seq, seg_ids, total_len = self._build_sequence(text_tokens, visual_tokens)

        # ── 4. Add positional + segment embeddings ────
        pos = self.pos_enc(total_len, seq.device)       # (1, total_len, D)
        seg = self.seg_enc(seg_ids)                     # (B, total_len, D)
        seq = self.input_drop(self.input_norm(seq + pos + seg))

        # ── 5. Build padding mask for transformer ─────
        # True = position should be IGNORED
        # CLS and SEP are always attended; visual tokens are always attended
        text_pad     = (attention_mask == 0)           # (B, L)   True=pad
        prefix_pad   = torch.zeros(B, 1, dtype=torch.bool, device=image.device)
        suffix_pad   = torch.zeros(B, 1 + visual_tokens.size(1),
                                   dtype=torch.bool, device=image.device)
        full_pad_mask = torch.cat([prefix_pad, text_pad, suffix_pad], dim=1)

        # ── 6. Transformer ────────────────────────────
        out = self.transformer(seq, src_key_padding_mask=full_pad_mask)

        # ── 7. Classify from CLS token ────────────────
        cls_out = out[:, 0, :]                          # (B, D)
        logits  = self.classifier(cls_out)              # (B, num_classes)
        return logits

    def get_attention_weights(self, image, input_ids, attention_mask):
        """
        Returns attention weights of the first transformer layer
        (useful for visualisation / interpretability).
        """
        # Note: nn.TransformerEncoder does not expose attention directly;
        # this is a simplified hook-based approach.
        weights = {}

        def hook_fn(module, inp, out):
            # out is (output, attn_weights) when need_weights=True
            # For TransformerEncoderLayer, out is the processed tensor
            weights["layer0_out"] = out.detach()

        handle = self.transformer.layers[0].register_forward_hook(hook_fn)
        with torch.no_grad():
            self.forward(image, input_ids, attention_mask)
        handle.remove()
        return weights

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────
def build_visual_bert(config: dict) -> VisualBERTModel:
    """
    Build VisualBERTModel from a configuration dictionary.

    Example:
        {
            "num_classes"  : 2,
            "d_model"      : 256,
            "nhead"        : 8,
            "num_layers"   : 4,
            "vocab_size"   : 30522,
            "max_text_len" : 64,
            "dropout"      : 0.1,
            "freeze_cnn"   : False,
        }
    """
    return VisualBERTModel(**config)


# ──────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    model = VisualBERTModel()
    print(f"Model: {model.model_name}")
    print(f"Trainable parameters: {model.count_parameters():,}")

    B, SEQ = 4, 64
    imgs   = torch.randn(B, 3, 224, 224)
    ids    = torch.randint(0, 30522, (B, SEQ))
    mask   = torch.ones(B, SEQ, dtype=torch.long)
    mask[:, 40:] = 0                   # Simulate padding

    with torch.no_grad():
        logits = model(imgs, ids, mask)

    print(f"Input image  : {imgs.shape}")
    print(f"Output logits: {logits.shape}")
    print(f"Probabilities: {F.softmax(logits, dim=-1)}")
