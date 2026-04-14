"""
Cross-Modal Attention Fusion (CMAF) — Section 3.2 of the manuscript.

Fuses EfficientNetV2L (global texture) and XceptionNet (local artifact)
feature maps using multi-head cross-attention, per-frame.

Steps:
  1. Align both feature maps to a common spatial resolution H̄ × W̄ = 14×14
     via adaptive average pooling.
  2. Project each to shared embedding dimension D = 512 with independent
     1×1 convolutions.
  3. Flatten to N = 196 tokens and add 2-D sinusoidal positional encodings.
  4. Multi-head cross-attention: Q from EfficientNetV2L, K/V from XceptionNet.
  5. Mean-pool spatial tokens → D-dimensional frame descriptor F_combined^t.

Equations (3)–(5) from the manuscript are faithfully implemented below.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding2D(nn.Module):
    """
    Fixed 2-D sinusoidal positional encoding for a sequence of H̄×W̄ spatial tokens.
    Follows the formulation from Vaswani et al. (2017), extended to 2-D grids.
    """

    def __init__(self, embed_dim: int, spatial_size: int = 14):
        super().__init__()
        H = W = spatial_size
        # Build (N, D) encoding where N = H*W
        pe = torch.zeros(H * W, embed_dim)
        y_pos = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
        x_pos = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()

        half = embed_dim // 4
        div_term = torch.exp(
            torch.arange(0, half, dtype=torch.float32) *
            -(math.log(10000.0) / half)
        )
        pe[:, 0:half]       = torch.sin(y_pos.unsqueeze(1) * div_term)
        pe[:, half:2*half]  = torch.cos(y_pos.unsqueeze(1) * div_term)
        pe[:, 2*half:3*half]= torch.sin(x_pos.unsqueeze(1) * div_term)
        pe[:, 3*half:4*half]= torch.cos(x_pos.unsqueeze(1) * div_term)
        # Remaining dimensions (if embed_dim % 4 != 0) stay zero
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, N, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) token sequence
        Returns:
            x + positional encoding, same shape
        """
        return x + self.pe


class CMAF(nn.Module):
    """
    Cross-Modal Attention Fusion (Equations 3–5, manuscript Section 3.2).

    Operates per-frame independently.  Temporal ordering is handled downstream
    by TIAM (Section 3.3).

    Architecture:
        proj_eff  : 1×1 Conv  D_E → D   (EfficientNetV2L alignment)
        proj_xcep : 1×1 Conv  D_X → D   (XceptionNet alignment)
        pos_enc   : 2-D sinusoidal PE
        attn      : nn.MultiheadAttention with h heads, dk = D/h

    Returns:
        F_combined : (B, D) — mean-pooled frame-level descriptor
    """

    def __init__(self,
                 eff_channels: int,
                 xcep_channels: int,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 spatial_size: int = 14,
                 dropout: float = 0.1):
        super().__init__()

        assert embed_dim % num_heads == 0, \
            "embed_dim must be divisible by num_heads"

        self.spatial_size = spatial_size
        self.embed_dim    = embed_dim

        # Step 1 + 2: spatial alignment + projection
        self.pool_eff  = nn.AdaptiveAvgPool2d((spatial_size, spatial_size))
        self.pool_xcep = nn.AdaptiveAvgPool2d((spatial_size, spatial_size))
        self.proj_eff  = nn.Conv2d(eff_channels,  embed_dim, kernel_size=1, bias=False)
        self.proj_xcep = nn.Conv2d(xcep_channels, embed_dim, kernel_size=1, bias=False)

        # Step 3: positional encoding
        self.pos_enc = SinusoidalPositionalEncoding2D(embed_dim, spatial_size)

        # Step 4: multi-head cross-attention
        # Q from EfficientNetV2L, K/V from XceptionNet
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q   = nn.LayerNorm(embed_dim)
        self.norm_kv  = nn.LayerNorm(embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)

        # Output projection W^O is handled internally by nn.MultiheadAttention

    def forward(self,
                feat_eff:  torch.Tensor,
                feat_xcep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_eff  : (B, D_E, H', W')  — EfficientNetV2L output
            feat_xcep : (B, D_X, H'', W'') — XceptionNet output
        Returns:
            F_combined: (B, D) — frame-level fused descriptor
        """
        B = feat_eff.size(0)

        # ── Alignment and projection ──────────────────────────────────────
        z_e = self.proj_eff(self.pool_eff(feat_eff))    # (B, D, H̄, W̄)
        z_x = self.proj_xcep(self.pool_xcep(feat_xcep)) # (B, D, H̄, W̄)

        # ── Flatten to token sequences ────────────────────────────────────
        N = self.spatial_size * self.spatial_size
        z_e = z_e.flatten(2).transpose(1, 2)   # (B, N, D)
        z_x = z_x.flatten(2).transpose(1, 2)   # (B, N, D)

        # ── Positional encoding ───────────────────────────────────────────
        z_e = self.pos_enc(z_e)   # Z_E = Flatten(F_E^proj) + PE
        z_x = self.pos_enc(z_x)   # Z_X = Flatten(F_X^proj) + PE

        # ── Layer-normalise before attention ─────────────────────────────
        q  = self.norm_q(z_e)
        kv = self.norm_kv(z_x)

        # ── Multi-head cross-attention (Eqs 4 & 5) ───────────────────────
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        # Residual + norm
        attn_out = self.out_norm(z_e + attn_out)

        # ── Mean-pool spatial tokens → frame descriptor ───────────────────
        # F_combined^t ∈ R^D
        F_combined = attn_out.mean(dim=1)   # (B, D)
        return F_combined
