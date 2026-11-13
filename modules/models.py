"""
models.py — All 6 models for the Infrared Solar Module fault classifier.

Models
------
  M1  ViTScratch       From-scratch Vision Transformer (patch 4×4 on 24×40)
  M2  FusionViT        Gated dual-branch: ViT image branch + MLP feature branch
  M3  ViTB16           Fine-tuned ViT-B/16 (ImageNet pretrained)
  B1  SwinT            Fine-tuned Swin-T  (ImageNet pretrained)
  B2  ResNet50         Fine-tuned ResNet-50 (ImageNet pretrained)
  B3  MobileNetV3      Fine-tuned MobileNetV3-Large (ImageNet pretrained)

get_model(name, num_classes) is the single entry point used by the notebook.

Complexity notes (in docstrings):
  - Self-attention: O(N² · d) where N = number of tokens
  - For M1/M2: N = (24/4)×(40/4) + 1 CLS = 60 + 1 = 61 tokens
  - For M3:    N = (224/16)² + 1         = 196 + 1 = 197 tokens
  - CNNs (B2, B3): O(K² · C_in · C_out · H · W) per conv layer — no quadratic token term
"""

import math
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T

NUM_CLASSES = 12   # defined globally; overridden by get_model(num_classes=...)


# ══════════════════════════════════════════════════════════════════════════════
#  M1 — ViT from Scratch
#  Implements: Patch Embedding, Multi-Head Self-Attention, Encoder Block, ViT
#  Reference: Dosovitskiy et al., "An Image is Worth 16×16 Words", ICLR 2021
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    """Split image into non-overlapping patches and linearly project each.

    For a 24×40 image with patch_size=4:
      grid = (24/4) × (40/4) = 6 × 10 = 60 patches
      Each patch is 4×4×1 = 16 pixels → projected to `embed_dim`.

    Complexity: O(H·W·embed_dim / patch_size²) — essentially O(N·d)
    """
    def __init__(self, img_h=40, img_w=24, patch_size=4,
                 in_channels=1, embed_dim=192):
        super().__init__()
        assert img_h % patch_size == 0 and img_w % patch_size == 0, \
            "Image dimensions must be divisible by patch_size."
        self.n_patches = (img_h // patch_size) * (img_w // patch_size)
        # A single Conv2d with kernel=patch_size and stride=patch_size
        # is mathematically identical to extracting patches + linear projection.
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) → (B, embed_dim, H/p, W/p) → (B, N, embed_dim)
        x = self.proj(x)                          # (B, D, h, w)
        x = x.flatten(2)                          # (B, D, N)
        x = x.transpose(1, 2)                     # (B, N, D)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention — implemented by hand.

    Does NOT use nn.MultiheadAttention so the attention weights are
    directly accessible for attention-rollout interpretability.

    Complexity: O(N² · d) where N = sequence length, d = head_dim.
    Memory:     O(B · heads · N²) for the attention matrix.
    """
    def __init__(self, embed_dim=192, num_heads=3, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = math.sqrt(self.head_dim)

        self.qkv     = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj    = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, N, D)
        Returns:
            out:      (B, N, D)
            attn_map: (B, heads, N, N)  — raw attention weights (after softmax)
        """
        B, N, D = x.shape
        H       = self.num_heads
        Dh      = self.head_dim

        # Compute Q, K, V in one matrix multiply then split
        qkv = self.qkv(x)                         # (B, N, 3D)
        q, k, v = qkv.chunk(3, dim=-1)            # each (B, N, D)

        # Reshape to (B, heads, N, head_dim)
        def reshape(t):
            return t.view(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention
        # attn shape: (B, H, N, N)
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Aggregate values
        out = attn @ v                             # (B, H, N, Dh)
        out = out.transpose(1, 2).contiguous()     # (B, N, H, Dh)
        out = out.view(B, N, D)                    # (B, N, D)
        out = self.proj(out)
        return out, attn                           # return attn for rollout


class TransformerBlock(nn.Module):
    """Single ViT encoder block: LayerNorm → MHSA → residual → LayerNorm → MLP → residual.

    Uses pre-norm (LayerNorm before the sub-layer), which trains more stably
    than the post-norm used in the original Transformer paper.
    """
    def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm self-attention with residual
        normed, attn = self.attn(self.norm1(x))
        x = x + normed
        # Pre-norm MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x, attn


class ViTScratch(nn.Module):
    """Vision Transformer trained from scratch on 24×40 thermal images.

    Architecture:
      PatchEmbedding  (patch_size=4) → 60 patch tokens
      + 1 learnable CLS token
      + learnable 1-D positional embeddings
      → 6 Transformer blocks (depth=6, dim=192, heads=3, mlp_ratio=4)
      → LayerNorm → CLS token → Linear classifier

    Total tokens N = 61.
    Self-attention complexity per layer: O(61² × 192) ≈ 715 k ops — very fast.
    Parameters: ~2.6 M

    Pros:  Tailored to native resolution; no upsampling artefacts; global
           receptive field from layer 1; interpretable attention maps.
    Cons:  Small dataset (20 k images) may under-train a ViT; no ImageNet
           pretraining; relies on augmentation and regularization.
    """
    def __init__(self, img_h=40, img_w=24, patch_size=4, in_channels=1,
                 embed_dim=192, depth=6, num_heads=3, mlp_ratio=4,
                 dropout=0.1, num_classes=NUM_CLASSES):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_h, img_w, patch_size,
                                          in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches       # 60

        # Learnable CLS token and positional embeddings
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed  = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token,  std=0.02)
        nn.init.trunc_normal_(self.pos_embed,  std=0.02)

        self.dropout    = nn.Dropout(dropout)
        self.blocks     = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm       = nn.LayerNorm(embed_dim)
        self.head       = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_attn=False):
        """
        Args:
            x          : (B, C, H, W)
            return_attn: if True, also return list of attention maps (one per block)
        Returns:
            logits      : (B, num_classes)
            attn_maps   : list of (B, heads, N, N)  [only if return_attn=True]
        """
        B = x.shape[0]

        # Patch embedding → (B, 60, 192)
        x = self.patch_embed(x)

        # Prepend CLS token → (B, 61, 192)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)

        # Add positional embeddings
        x = self.dropout(x + self.pos_embed)

        # Pass through Transformer blocks
        attn_maps = []
        for block in self.blocks:
            x, attn = block(x)
            if return_attn:
                attn_maps.append(attn)

        x      = self.norm(x)
        cls_out = x[:, 0]                          # CLS token only
        logits  = self.head(cls_out)

        if return_attn:
            return logits, attn_maps
        return logits


# ══════════════════════════════════════════════════════════════════════════════
#  M2 — Fusion-ViT  (Our Proposed Method)
#  Gated dual-branch: ViT image branch + MLP feature branch
# ══════════════════════════════════════════════════════════════════════════════

class FusionViT(nn.Module):
    """Custom Gated Fusion-ViT — the main proposed method.

    Two branches:
      Image branch : ViTScratch (24×40) → CLS embedding h_img  (192-dim)
      Feature branch: MLP on 33 thermal features → h_feat       (192-dim)

    Gated fusion (learned weighting of the two modalities):
      gate   = sigmoid(W · [h_img ; h_feat])    (192-dim gate vector)
      h_fuse = gate * h_img + (1 − gate) * h_feat
      logits = Linear(h_fuse)

    Why gated instead of simple concatenation?
      The gate learns which modality is more reliable per sample.
      If the image is noisy, the gate suppresses h_img and relies on features.
      This is more flexible than a fixed concat + linear.

    Pros:  Exploits both spatial and radiometric information; interpretable
           gate values indicate modality importance per prediction.
    Cons:  Feature branch requires precomputed 33-dim vectors; training
           is slightly more complex (two branches must converge together).

    Parameters: ~2.9 M (ViT ~2.6 M + feature MLP ~0.14 M + gate ~0.07 M)
    """
    def __init__(self, feat_dim=33, embed_dim=192, num_classes=NUM_CLASSES,
                 vit_kwargs=None):
        super().__init__()
        if vit_kwargs is None:
            vit_kwargs = {}

        # Image branch (scratch ViT — no head)
        self.vit = ViTScratch(embed_dim=embed_dim,
                              num_classes=num_classes, **vit_kwargs)
        self.vit.head = nn.Identity()   # remove classifier; we add our own below

        # Feature branch: 33 → 128 → 192
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim),
            nn.GELU(),
        )

        # Gated fusion layer: input = concatenated [h_img; h_feat] = 384-dim
        self.gate   = nn.Linear(2 * embed_dim, embed_dim)
        self.head   = nn.Linear(embed_dim, num_classes)

    def forward(self, img, feat, return_attn=False):
        """
        Args:
            img        : (B, C, H, W)
            feat       : (B, 33)  — standardized thermal features
            return_attn: return ViT attention maps
        Returns:
            logits: (B, num_classes)
        """
        # Image branch → CLS embedding
        if return_attn:
            h_img, attn_maps = self.vit(img, return_attn=True)
        else:
            h_img = self.vit(img)

        # Feature branch
        h_feat = self.feat_mlp(feat)                # (B, 192)

        # Gated fusion
        concat = torch.cat([h_img, h_feat], dim=-1) # (B, 384)
        gate   = torch.sigmoid(self.gate(concat))   # (B, 192)
        h_fuse = gate * h_img + (1.0 - gate) * h_feat  # (B, 192)

        logits = self.head(h_fuse)
        if return_attn:
            return logits, attn_maps
        return logits


class FeatureMLP(nn.Module):
    """Thermal-feature-only ablation baseline matching FusionViT's 33-d MLP branch + head."""
    def __init__(self, feat_dim=33, embed_dim=192, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, feat):
        """Map standardized (B, 33) thermal feature vectors to (B, num_classes) logits."""
        return self.net(feat)


# ══════════════════════════════════════════════════════════════════════════════
#  Pretrained baseline wrappers (M3, B1, B2, B3)
#  All use torchvision pretrained ImageNet weights.
#  Input: 224×224, 3-channel (grayscale repeated 3×).
#  Classifier head replaced with Linear(feat_dim, num_classes).
# ══════════════════════════════════════════════════════════════════════════════

def _freeze_backbone(model, unfreeze_last_n_layers=0):
    """Freeze all parameters; optionally unfreeze the last N named children."""
    for p in model.parameters():
        p.requires_grad = False
    if unfreeze_last_n_layers > 0:
        children = list(model.children())
        for child in children[-unfreeze_last_n_layers:]:
            for p in child.parameters():
                p.requires_grad = True


def get_vitb16(num_classes=NUM_CLASSES, freeze_backbone=True):
    """ViT-B/16 pretrained on ImageNet-1k with fine-tuned head.

    Freezing the 85.8M-parameter transformer backbone is standard transfer learning practice.
    It preserves general visual features, avoids catastrophic forgetting on small datasets,
    and reduces per-epoch training time on Apple Silicon from hours to ~1 minute.
    """
    weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1
    model   = tvm.vit_b_16(weights=weights)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    in_feat = model.heads.head.in_features
    model.heads.head = nn.Linear(in_feat, num_classes)
    for p in model.heads.head.parameters():
        p.requires_grad = True
    return model


def get_swint(num_classes=NUM_CLASSES, freeze_backbone=True):
    """Swin Transformer Tiny pretrained on ImageNet-1k with fine-tuned head."""
    weights = tvm.Swin_T_Weights.IMAGENET1K_V1
    model   = tvm.swin_t(weights=weights)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    in_feat = model.head.in_features
    model.head = nn.Linear(in_feat, num_classes)
    for p in model.head.parameters():
        p.requires_grad = True
    return model


def get_resnet50(num_classes=NUM_CLASSES, freeze_backbone=True):
    """ResNet-50 pretrained on ImageNet-1k with fine-tuned head."""
    weights = tvm.ResNet50_Weights.IMAGENET1K_V2
    model   = tvm.resnet50(weights=weights)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for p in model.fc.parameters():
        p.requires_grad = True
    return model


def get_mobilenetv3(num_classes=NUM_CLASSES, freeze_backbone=True):
    """MobileNetV3-Large pretrained on ImageNet-1k with fine-tuned head."""
    weights = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model   = tvm.mobilenet_v3_large(weights=weights)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    in_feat = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_feat, num_classes)
    for p in model.classifier[3].parameters():
        p.requires_grad = True
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Factory function — single entry point used by notebooks
# ══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "vit_scratch":  lambda nc: ViTScratch(num_classes=nc),
    "fusion_vit":   lambda nc: FusionViT(num_classes=nc),
    "vitb16":       lambda nc: get_vitb16(num_classes=nc),
    "swin_t":       lambda nc: get_swint(num_classes=nc),
    "resnet50":     lambda nc: get_resnet50(num_classes=nc),
    "mobilenetv3":  lambda nc: get_mobilenetv3(num_classes=nc),
}


def get_model(name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Return an initialised model by name.

    Valid names: 'vit_scratch', 'fusion_vit', 'vitb16',
                 'swin_t', 'resnet50', 'mobilenetv3'
    """
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. "
                         f"Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](num_classes)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

