"""
model2_clip.py
==============
Model 2: CLIP-based Multimodal Sarcasm Classifier
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture:
  ┌─────────────────────────────────────────────────────┐
  │               CLIP (ViT-B/32)                       │
  │  ┌─────────────────┐    ┌──────────────────────┐   │
  │  │  Visual Encoder  │    │   Text Encoder        │   │
  │  │  (ViT-B/32)     │    │   (Transformer)       │   │
  │  └────────┬────────┘    └─────────┬────────────┘   │
  │           │  clip_img_emb (512)   │ clip_txt_emb    │
  └───────────┼───────────────────────┼─────────────────┘
              │                       │
       ┌──────┴───────────────────────┴──────┐
       │     Fusion Strategies (x3):          │
       │   • Concat : [img ‖ txt]  (1024-d)  │
       │   • Hadamard product                 │
       │   • Element-wise difference          │
       └──────────────────┬──────────────────┘
                          │
                  MLP Classifier
                  FC(512) → ReLU → Dropout
                  FC(128) → ReLU → Dropout
                  FC(2)

  Note: We use the open-source implementation via
  'open_clip_torch' if available, else fall back to a
  custom lightweight CLIP-like architecture for CPU.

Author: Lab Project – Meme & Sarcasm Understanding
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────
# Try to import open_clip; fall back gracefully
# ──────────────────────────────────────────────
try:
    import open_clip                                    # pip install open-clip-torch
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False


# ──────────────────────────────────────────────
# Fallback: Lightweight CLIP-like Backbone (CPU)
# ──────────────────────────────────────────────
class _PatchEmbedding(nn.Module):
    """Split image into 16×16 patches and linearly embed them."""

    def __init__(self, img_size=224, patch_size=32, in_channels=3, embed_dim=512):
        super().__init__()
        n_patches    = (img_size // patch_size) ** 2
        self.proj    = nn.Conv2d(in_channels, embed_dim,
                                 kernel_size=patch_size, stride=patch_size)
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)

    def forward(self, x):
        B     = x.size(0)
        x     = self.proj(x)                            # (B, D, H/P, W/P)
        x     = x.flatten(2).transpose(1, 2)            # (B, N, D)
        cls   = self.cls_tok.expand(B, -1, -1)
        x     = torch.cat([cls, x], dim=1)              # (B, N+1, D)
        return x + self.pos_emb


class _LightweightVisualEncoder(nn.Module):
    """Mini ViT-like visual encoder (2-layer transformer)."""

    def __init__(self, img_size=224, patch_size=32, embed_dim=512, nhead=8,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.patch_embed = _PatchEmbedding(img_size, patch_size, 3, embed_dim)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder     = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)          # (B, N+1, D)
        x = self.encoder(x)              # (B, N+1, D)
        return self.norm(x[:, 0])        # CLS token → (B, D)


class _LightweightTextEncoder(nn.Module):
    """Mini transformer text encoder."""

    def __init__(self, vocab_size=30522, embed_dim=512, max_len=64,
                 nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb   = nn.Embedding(max_len, embed_dim)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder   = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm      = nn.LayerNorm(embed_dim)

    def forward(self, input_ids, attention_mask):
        B, L  = input_ids.shape
        pos   = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x     = self.embedding(input_ids) + self.pos_emb(pos)   # (B, L, D)

        # Create padding mask (True = ignore)
        pad_mask = (attention_mask == 0)                          # (B, L)
        x     = self.encoder(x, src_key_padding_mask=pad_mask)  # (B, L, D)

        # Masked mean pool
        mask  = attention_mask.unsqueeze(-1).float()
        out   = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.norm(out)                                     # (B, D)


# ──────────────────────────────────────────────
# CLIP Backbone Wrapper
# ──────────────────────────────────────────────
class CLIPBackbone(nn.Module):
    """
    Wraps either the real open_clip ViT-B/32 or the lightweight fallback.

    Args:
        use_pretrained (bool): use open_clip if available
        embed_dim (int): dimension for the fallback backbone (default 512)
    """

    def __init__(self, use_pretrained: bool = True, embed_dim: int = 512):
        super().__init__()
        self.use_real_clip = use_pretrained and OPEN_CLIP_AVAILABLE
        self.embed_dim     = embed_dim

        if self.use_real_clip:
            clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            self.visual  = clip_model.visual
            self.text    = clip_model.transformer
            self.embed_dim = clip_model.visual.output_dim      # 512
            print("[CLIPBackbone] Using pretrained ViT-B/32 from open_clip")
        else:
            self.visual  = _LightweightVisualEncoder(embed_dim=embed_dim)
            self.text    = _LightweightTextEncoder(embed_dim=embed_dim)
            print("[CLIPBackbone] open_clip not found – using lightweight fallback")

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.use_real_clip:
            return self.visual(image)                         # (B, 512)
        return self.visual(image)

    def encode_text(self, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        if self.use_real_clip:
            # open_clip uses its own tokenizer; here we reuse text model directly
            return self.text(input_ids)                       # (B, 512)
        return self.text(input_ids, attention_mask)


# ──────────────────────────────────────────────
# Full Model: CLIP Sarcasm Classifier
# ──────────────────────────────────────────────
class CLIPSarcasmClassifier(nn.Module):
    """
    Sarcasm/Meme classifier built on top of CLIP features.

    Fusion:
        Three complementary interaction signals are concatenated:
        1. Concatenation          : [v ‖ t]
        2. Element-wise product   : v ⊙ t    (captures alignment)
        3. Absolute difference    : |v − t|  (captures mismatch/sarcasm cue)

    Final embedding size: embed_dim * 4 → MLP head

    Args:
        num_classes    (int)   : number of output classes
        embed_dim      (int)   : CLIP embedding size (512)
        hidden_dim     (int)   : MLP hidden dimension
        dropout        (float) : dropout rate
        use_pretrained (bool)  : use real CLIP if available
        freeze_clip    (bool)  : freeze CLIP backbone
    """

    def __init__(self, num_classes: int = 2, embed_dim: int = 512,
                 hidden_dim: int = 512, dropout: float = 0.3,
                 use_pretrained: bool = True, freeze_clip: bool = True):
        super().__init__()
        self.model_name = "CLIP"

        # CLIP backbone
        self.clip = CLIPBackbone(use_pretrained=use_pretrained,
                                 embed_dim=embed_dim)
        actual_dim = self.clip.embed_dim

        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
            print("[CLIPSarcasmClassifier] CLIP backbone frozen")

        # Projection for each modality (adapt from CLIP dim to desired dim)
        self.img_proj = nn.Sequential(
            nn.Linear(actual_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(actual_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Fusion: concat + product + difference → 4 * embed_dim
        fusion_dim = embed_dim * 4

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes),
        )

        # Learnable temperature for cross-modal similarity
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

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
        # Encode
        img_feat = self.clip.encode_image(image)                      # (B, D)
        txt_feat = self.clip.encode_text(input_ids, attention_mask)   # (B, D)

        # Project + L2-normalise
        img_emb = F.normalize(self.img_proj(img_feat), dim=-1)        # (B, D)
        txt_emb = F.normalize(self.txt_proj(txt_feat), dim=-1)        # (B, D)

        # Three-way fusion
        concat  = torch.cat([img_emb, txt_emb],         dim=-1)  # (B, 2D)
        product = img_emb * txt_emb                               # (B, D)
        diff    = torch.abs(img_emb - txt_emb)                   # (B, D)

        fused   = torch.cat([concat, product, diff], dim=-1)     # (B, 4D)
        logits  = self.classifier(fused)                         # (B, num_classes)
        return logits

    def get_similarity(self, image, input_ids, attention_mask) -> torch.Tensor:
        """Returns cosine similarity between image and text embeddings."""
        img_feat = self.clip.encode_image(image)
        txt_feat = self.clip.encode_text(input_ids, attention_mask)
        img_emb  = F.normalize(self.img_proj(img_feat), dim=-1)
        txt_emb  = F.normalize(self.txt_proj(txt_feat), dim=-1)
        scale    = self.logit_scale.exp()
        return scale * (img_emb * txt_emb).sum(dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────
def build_clip_model(config: dict) -> CLIPSarcasmClassifier:
    """
    Build CLIPSarcasmClassifier from config dict.

    Example:
        {
            "num_classes"    : 2,
            "embed_dim"      : 512,
            "hidden_dim"     : 512,
            "dropout"        : 0.3,
            "use_pretrained" : False,   # set True for real CLIP
            "freeze_clip"    : True,
        }
    """
    return CLIPSarcasmClassifier(**config)


# ──────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    # Force lightweight mode for smoke test
    model = CLIPSarcasmClassifier(use_pretrained=False, freeze_clip=False)
    print(f"Model: {model.model_name}")
    print(f"Trainable parameters: {model.count_parameters():,}")

    B, SEQ = 4, 64
    imgs   = torch.randn(B, 3, 224, 224)
    ids    = torch.randint(0, 30522, (B, SEQ))
    mask   = torch.ones(B, SEQ, dtype=torch.long)

    with torch.no_grad():
        logits = model(imgs, ids, mask)
        sim    = model.get_similarity(imgs, ids, mask)

    print(f"Output logits     : {logits.shape}")
    print(f"Image-Text sim    : {sim[:4]}")
    print(f"Probs             : {F.softmax(logits, dim=-1)}")
