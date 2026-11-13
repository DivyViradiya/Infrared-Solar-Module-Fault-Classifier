"""vit_fast.py — fast partial fine-tuning of ViT-B/16 (M3) via cached tokens.

Idea: freeze blocks 0..cut-1, run them ONCE per image (cached), then train only
the last (12-cut) blocks + head on the cached tokens. Input is 160x96 (aspect-
preserving 4x upsample of 40x24) -> 10x6 = 60 patches + CLS = 61 tokens
(vs 197 tokens at 224x224). Images are held as uint8 tensors and resized on
the device, so there is no PIL / DataLoader work inside the training loop.
"""
import os, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _resolve(p):
    for c in (p, os.path.join("..", p), p[3:] if p.startswith("../") else p):
        if os.path.exists(c):
            return c
    return p


def load_uint8(df):
    """All images -> uint8 tensor (N, 40, 24). ~20 MB for the full dataset."""
    return torch.from_numpy(np.stack(
        [np.array(Image.open(_resolve(p)).convert("L")) for p in df["image_filepath"]]))


def build_vitb16(input_hw=(160, 96), pretrained=True, num_classes=12):
    """torchvision ViT-B/16 with positional embeddings resized to input_hw."""
    m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
    gh, gw = input_hw[0] // 16, input_hw[1] // 16
    pe = m.encoder.pos_embedding.data
    cls, grid = pe[:, :1], pe[:, 1:].reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(gh, gw), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)
    m.encoder.pos_embedding = nn.Parameter(torch.cat([cls, grid], 1))
    m.heads.head = nn.Linear(768, num_classes)
    return m


def embed(m, x):
    """Patchify + CLS + pos-embed (bypasses torchvision's square-224 assert)."""
    t = m.conv_proj(x).flatten(2).transpose(1, 2)
    t = torch.cat([m.class_token.expand(t.shape[0], -1, -1), t], 1)
    return m.encoder.dropout(t + m.encoder.pos_embedding)


def prep(u8, hw, device, flip=(0, 0)):
    x = u8.to(device).float().div(255).unsqueeze(1)
    if flip[0]: x = x.flip(3)
    if flip[1]: x = x.flip(2)
    x = F.interpolate(x, size=hw, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    return (x - _MEAN.to(device)) / _STD.to(device)


@torch.no_grad()
def cache_tokens(m, u8, hw, cut, device, flip=(0, 0), bs=256):
    """Run embed + blocks[:cut] once; return fp16 tokens (N, 61, 768) on CPU."""
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    front = copy.deepcopy(m).to(device=device, dtype=dtype).eval()
    out = []
    for i in range(0, len(u8), bs):
        t = embed(front, prep(u8[i:i + bs], hw, device, flip).to(dtype))
        out.append(front.encoder.layers[:cut](t).half().cpu())
    return torch.cat(out)


class Tail(nn.Module):
    """Trainable part: blocks[cut:] + final LN + new head. Shares weights with m."""
    def __init__(self, m, cut, num_classes=12):
        super().__init__()
        self.blocks = nn.Sequential(*list(m.encoder.layers)[cut:])
        self.ln = m.encoder.ln
        self.head = m.heads.head   # same Linear object as in m
    def forward(self, t):
        return self.head(self.ln(self.blocks(t))[:, 0])


def forward_full(m, x):
    """Full forward pass for resized-pos-embed ViT-B/16 (bypasses torchvision 224x224 assert)."""
    t = embed(m, x)
    t = m.encoder.ln(m.encoder.layers(t))
    return m.heads.head(t[:, 0])


def forward_with_attn(m, x):
    """Full forward pass returning (logits, per-layer attention maps of shape (B, H, N, N))."""
    t = embed(m, x)
    attn_maps = []
    for blk in m.encoder.layers:
        h = blk.ln_1(t)
        attn_out, attn_w = blk.self_attention(
            h, h, h, need_weights=True, average_attn_weights=False
        )
        t = t + blk.dropout(attn_out)
        t = t + blk.mlp(blk.ln_2(t))
        attn_maps.append(attn_w)
    logits = m.heads.head(m.encoder.ln(t)[:, 0])
    return logits, attn_maps
