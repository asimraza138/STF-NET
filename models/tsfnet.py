"""
STF-Net: Tempo-Spatial-Fusion Network — full model definition.

Architecture (Figure 1 of the manuscript):
  Input video → frame extraction & preprocessing
  ↓ (per frame, independently)
  EfficientNetV2L      → E(fᵢ) ∈ R^{H'×W'×D_E}
  XceptionNet (mod.)   → X(fᵢ) ∈ R^{H''×W''×D_X}
  ↓
  CMAF                 → F^t_combined ∈ R^D  (per frame)
  ↓ (across frames)
  TIAM (Bi-LSTM + temporal attention)  → TIAM_output ∈ R^{2H}
  ↓
  Classification head (Mish → Linear → Sigmoid)  → ŷ ∈ (0, 1)

The forward() method of TSFNet expects a pre-extracted sequence of face
frames (B, T, 3, 224, 224).  Raw video → frame extraction → MTCNN face
detection is handled in data/preprocessing.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import EfficientNetV2LBackbone, ModifiedXceptionNet
from .cmaf      import CMAF
from .tiam      import TIAM


# ---------------------------------------------------------------------------
# Mish activation (Equation 14 of the manuscript)
# ---------------------------------------------------------------------------

class Mish(nn.Module):
    """
    Mish(x) = x · tanh(ln(1 + eˣ))
    A smooth, non-monotonic activation that outperforms ReLU in many settings.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


# ---------------------------------------------------------------------------
# STFNETs
# ---------------------------------------------------------------------------

class TSFNet(nn.Module):
    """
    Full Tempo-Spatial-Fusion Network.

    Args:
        embed_dim        : Shared embedding dimension D (512)
        cmaf_heads       : Number of CMAF attention heads (8)
        spatial_size     : Spatial token grid H̄ = W̄ (14)
        bilstm_hidden    : Bi-LSTM hidden units per direction (256)
        bilstm_layers    : Number of Bi-LSTM layers (2)
        bilstm_dropout   : Bi-LSTM inter-layer dropout (0.5)
        pretrained_bb    : Load ImageNet pretrained backbone weights
    """

    def __init__(self,
                 embed_dim:      int   = 512,
                 cmaf_heads:     int   = 8,
                 spatial_size:   int   = 14,
                 bilstm_hidden:  int   = 256,
                 bilstm_layers:  int   = 2,
                 bilstm_dropout: float = 0.5,
                 pretrained_bb:  bool  = True):
        super().__init__()

        # ── Spatial feature extractors ────────────────────────────────────
        self.eff_backbone  = EfficientNetV2LBackbone(pretrained=pretrained_bb)
        self.xcep_backbone = ModifiedXceptionNet(pretrained=pretrained_bb)

        eff_channels  = self.eff_backbone.out_channels
        xcep_channels = self.xcep_backbone.out_channels

        # ── Cross-Modal Attention Fusion ──────────────────────────────────
        self.cmaf = CMAF(
            eff_channels=eff_channels,
            xcep_channels=xcep_channels,
            embed_dim=embed_dim,
            num_heads=cmaf_heads,
            spatial_size=spatial_size,
        )

        # ── Temporal Inconsistency Attention Module ───────────────────────
        self.tiam = TIAM(
            embed_dim=embed_dim,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            dropout=bilstm_dropout,
        )

        bilstm_out_dim = 2 * bilstm_hidden   # bidirectional

        # ── Classification head (Section 3.6) ────────────────────────────
        # Dense → Mish → Dense → Sigmoid
        self.classifier = nn.Sequential(
            nn.LayerNorm(bilstm_out_dim),
            nn.Linear(bilstm_out_dim, 256),
            Mish(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    # -----------------------------------------------------------------------
    # Backbone freeze / unfreeze helpers (for staged training)
    # -----------------------------------------------------------------------

    def freeze_backbones(self):
        """Freeze EfficientNetV2L and XceptionNet weights."""
        for p in self.eff_backbone.parameters():
            p.requires_grad = False
        for p in self.xcep_backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbones(self):
        """Unfreeze EfficientNetV2L and XceptionNet weights."""
        for p in self.eff_backbone.parameters():
            p.requires_grad = True
        for p in self.xcep_backbone.parameters():
            p.requires_grad = True

    # -----------------------------------------------------------------------
    # Per-frame feature extraction
    # -----------------------------------------------------------------------

    def extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract CMAF frame-level descriptors for a batch of sequences.

        Args:
            frames: (B, T, 3, H, W)
        Returns:
            seq: (B, T, D)  — sequence of F_combined^t
        """
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)   # treat all frames as independent batch

        # Spatial feature extraction
        feat_eff  = self.eff_backbone(x)    # (B*T, D_E, H', W')
        feat_xcep = self.xcep_backbone(x)   # (B*T, D_X, H'', W'')

        # CMAF → per-frame descriptor
        F_combined = self.cmaf(feat_eff, feat_xcep)  # (B*T, D)
        seq = F_combined.view(B, T, -1)              # (B, T, D)
        return seq

    # -----------------------------------------------------------------------
    # Full forward pass
    # -----------------------------------------------------------------------

    def forward(self,
                frames:  torch.Tensor,
                lengths: torch.Tensor | None = None) -> dict:
        """
        Full TSF-Net forward pass.

        Args:
            frames  : (B, T, 3, H, W) — preprocessed face-crop frame sequences
            lengths : (B,) number of valid frames per clip (None = all T valid)

        Returns:
            dict with:
              'prob'       : (B,) predicted probability of being deepfake ∈ (0,1)
              'F_combined' : (B, D) last-frame CMAF descriptor (for AALF)
              'tiam_out'   : (B, bilstm_dim) TIAM output
        """
        # ── Frame-level feature extraction ───────────────────────────────
        seq = self.extract_frame_features(frames)   # (B, T, D)

        # AALF needs a single F_combined representative per video.
        # We expose the last valid frame's descriptor.
        if lengths is not None:
            idx       = (lengths - 1).clamp(min=0).long()            # (B,)
            gather_idx = idx.view(B, 1, 1).expand(-1, 1, seq.size(2))
            F_combined_for_loss = seq.gather(1, gather_idx).squeeze(1) # (B, D)
        else:
            F_combined_for_loss = seq[:, -1, :]   # (B, D)

        B = frames.size(0)  # re-read in case sizes needed below

        # ── TIAM (temporal attention) ─────────────────────────────────────
        tiam_out = self.tiam(seq, lengths)    # (B, bilstm_dim)

        # ── Classification ────────────────────────────────────────────────
        logit = self.classifier(tiam_out).squeeze(1)   # (B,)
        prob  = torch.sigmoid(logit)                   # (B,)

        return {
            "prob"       : prob,
            "F_combined" : F_combined_for_loss,
            "tiam_out"   : tiam_out,
        }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_tsfnet(cfg: dict, pretrained_bb: bool = True) -> TSFNet:
    """
    Instantiate TSFNet from a configuration dict (loaded from default.yaml).

    Args:
        cfg          : dict — the 'model' section of default.yaml
        pretrained_bb: whether to load ImageNet-pretrained backbone weights
    Returns:
        model: TSFNet instance
    """
    return TSFNet(
        embed_dim=cfg.get("embed_dim", 512),
        cmaf_heads=cfg.get("cmaf_heads", 8),
        spatial_size=cfg.get("spatial_resolution", 14),
        bilstm_hidden=cfg.get("bilstm_hidden", 256),
        bilstm_layers=cfg.get("bilstm_layers", 2),
        bilstm_dropout=cfg.get("bilstm_dropout", 0.5),
        pretrained_bb=pretrained_bb,
    )
