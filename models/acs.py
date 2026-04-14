"""
Adaptive Computational Scaling (ACS) — Section 3.5 of the manuscript.

Three processing paths:
  Lightweight  : MobileNetV3-Small  + simplified temporal analysis
                 (90 % faster, ~5.7 % accuracy drop)
  Standard     : EfficientNetV2-S   + standard Bi-LSTM
                 (40 % faster, ~2.1 % accuracy drop)
  Full model   : Complete STF-Net architecture

Path selection:
  C(video) < τ_low               → lightweight
  τ_low ≤ C(video) < τ_high      → standard
  otherwise                       → full model

Confidence gate: if the full-model classifier's sigmoid output falls below
0.55 (near the decision boundary), the video is automatically re-routed to
the full model regardless of C(video).

Complexity Estimator:
  A lightweight CNN trained independently on SSIM-based proxy complexity
  labels (generated in utils/ssim_labels.py).  Operates on the first
  frame of a video only.  Architecture: torchvision SqueezeNet 1.1 with
  the classifier head replaced by a single Linear → Sigmoid layer.
  SqueezeNet 1.1's fire modules (squeeze + expand) naturally detect
  multi-scale texture density — precisely what defines visual complexity
  in a face frame.  The model has ≈1.2 M parameters and runs in ~1.5 ms
  per frame on an A100 GPU (batch = 1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# Complexity Estimator
# ---------------------------------------------------------------------------

class ComplexityEstimator(nn.Module):
    """
    Lightweight visual complexity estimator based on SqueezeNet 1.1.

    SqueezeNet 1.1 uses fire modules (squeeze 1×1 conv followed by parallel
    1×1 and 3×3 expand convolutions) — an architectural primitive that does not
    appear in any of STF-Net's other components, keeping the complexity estimator
    completely independent.

    The estimator outputs C(video) ∈ [0, 1]: higher values indicate more
    temporally complex videos that warrant the full processing path.

    Input : first frame of a video, (B, 3, 224, 224)
    Output: complexity score, (B,) ∈ [0, 1]
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tvm.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
        base    = tvm.squeezenet1_1(weights=weights)

        # Remove the original classifier; keep only feature extraction
        self.features = base.features   # (B, 512, 7, 7) for 224×224 input

        # Replace classifier with a compact regression head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) — first frame, normalised
        Returns:
            score: (B,) complexity ∈ [0, 1]
        """
        feat  = self.features(x)
        score = self.head(feat).squeeze(1)
        return score


# ---------------------------------------------------------------------------
# Lightweight path backbone
# ---------------------------------------------------------------------------

class LightweightDetector(nn.Module):
    """
    Lightweight detection path: MobileNetV3-Small + simple temporal pooling.
    Used when C(video) < τ_low.

    Averages frame-level sigmoid predictions (no Bi-LSTM, no CMAF).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base    = tvm.mobilenet_v3_small(weights=weights)
        # Replace classifier head with a single sigmoid output
        in_feat = base.classifier[-1].in_features
        base.classifier[-1] = nn.Linear(in_feat, 1)
        self.model = base

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, 3, H, W)
        Returns:
            prob: (B,) probability of being deepfake
        """
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)
        logits = self.model(x).squeeze(1)              # (B*T,)
        probs  = torch.sigmoid(logits).view(B, T)
        return probs.mean(dim=1)                        # (B,)


# ---------------------------------------------------------------------------
# Standard path backbone
# ---------------------------------------------------------------------------

class StandardDetector(nn.Module):
    """
    Standard detection path: EfficientNetV2-S + standard Bi-LSTM.
    Used when τ_low ≤ C(video) < τ_high.

    Processes frames through EfficientNetV2-S to get frame embeddings,
    then aggregates temporally with a standard (non-attention) Bi-LSTM.
    """

    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s", pretrained=pretrained, num_classes=0
        )
        backbone_dim = self.backbone.num_features

        self.bilstm = nn.LSTM(
            input_size=backbone_dim,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(2 * embed_dim, 1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, 3, H, W)
        Returns:
            prob: (B,) probability of being deepfake
        """
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)
        feats = self.backbone(x)             # (B*T, backbone_dim)
        feats = feats.view(B, T, -1)         # (B, T, backbone_dim)
        lstm_out, _ = self.bilstm(feats)     # (B, T, 2*embed_dim)
        video_feat  = lstm_out[:, -1, :]    # last hidden state
        logit = self.classifier(video_feat).squeeze(1)
        return torch.sigmoid(logit)          # (B,)


# ---------------------------------------------------------------------------
# ACS Controller
# ---------------------------------------------------------------------------

class ACSController(nn.Module):
    """
    Adaptive Computational Scaling controller (Section 3.5).

    Wraps the three processing paths and routes each video to the
    appropriate path at inference time.  During training, only the full
    STF-Net is trained; the lightweight and standard paths are used only
    for ACS inference routing.

    Args:
        tau_low          : lower complexity threshold (0.3)
        tau_high         : upper complexity threshold (0.7)
        confidence_gate  : classifier sigmoid threshold below which
                           the video is force-routed to the full model (0.55)
    """

    def __init__(self,
                 tau_low:         float = 0.3,
                 tau_high:        float = 0.7,
                 confidence_gate: float = 0.55,
                 pretrained:      bool  = True):
        super().__init__()

        self.tau_low         = tau_low
        self.tau_high        = tau_high
        self.confidence_gate = confidence_gate

        self.estimator  = ComplexityEstimator(pretrained=pretrained)
        self.lightweight = LightweightDetector(pretrained=pretrained)
        self.standard    = StandardDetector(pretrained=pretrained)
        # Full STF-Net is passed externally at inference time

    def estimate_complexity(self, first_frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            first_frame: (B, 3, H, W)
        Returns:
            score: (B,) ∈ [0, 1]
        """
        with torch.no_grad():
            return self.estimator(first_frame)

    def route(self,
              frames:     torch.Tensor,
              full_model: nn.Module) -> tuple[torch.Tensor, list[str]]:
        """
        Route each video in the batch to the appropriate path.

        Args:
            frames     : (B, T, 3, H, W)
            full_model : the complete STF-Net (used for full-path videos)
        Returns:
            probs  : (B,) merged probability predictions
            routes : list of B strings indicating which path was used
        """
        B      = frames.size(0)
        device = frames.device
        first  = frames[:, 0]   # (B, 3, H, W) — first frame only

        # ── Complexity estimation ─────────────────────────────────────────
        c = self.estimate_complexity(first)   # (B,)

        light_mask  = c < self.tau_low
        std_mask    = (c >= self.tau_low) & (c < self.tau_high)
        full_mask   = c >= self.tau_high

        probs  = torch.zeros(B, device=device)
        routes = [""] * B

        # ── Lightweight path ──────────────────────────────────────────────
        if light_mask.any():
            idx      = light_mask.nonzero(as_tuple=True)[0]
            p_light  = self.lightweight(frames[idx])
            # Confidence gate: near-boundary predictions → full model
            low_conf = p_light < self.confidence_gate
            if low_conf.any():
                gate_idx = idx[low_conf]
                p_full   = full_model(frames[gate_idx])
                probs[gate_idx] = p_full
                for i in gate_idx.tolist():
                    routes[i] = "full (gate)"
                keep = ~low_conf
                if keep.any():
                    probs[idx[keep]] = p_light[keep]
                    for i in idx[keep].tolist():
                        routes[i] = "lightweight"
            else:
                probs[idx] = p_light
                for i in idx.tolist():
                    routes[i] = "lightweight"

        # ── Standard path ─────────────────────────────────────────────────
        if std_mask.any():
            idx      = std_mask.nonzero(as_tuple=True)[0]
            p_std    = self.standard(frames[idx])
            low_conf = p_std < self.confidence_gate
            if low_conf.any():
                gate_idx = idx[low_conf]
                p_full   = full_model(frames[gate_idx])
                probs[gate_idx] = p_full
                for i in gate_idx.tolist():
                    routes[i] = "full (gate)"
                keep = ~low_conf
                if keep.any():
                    probs[idx[keep]] = p_std[keep]
                    for i in idx[keep].tolist():
                        routes[i] = "standard"
            else:
                probs[idx] = p_std
                for i in idx.tolist():
                    routes[i] = "standard"

        # ── Full model path ───────────────────────────────────────────────
        if full_mask.any():
            idx    = full_mask.nonzero(as_tuple=True)[0]
            p_full = full_model(frames[idx])
            probs[idx] = p_full
            for i in idx.tolist():
                routes[i] = "full"

        return probs, routes
