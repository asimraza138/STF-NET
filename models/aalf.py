"""
Artifact-Aware Loss Function (AALF) — Section 3.4 of the manuscript.

AALF = -w_real · y · log(ŷ)
      - w_fake · (1-y) · log(1-ŷ)
      + λ · R(ŷ, F_combined)

R(ŷ, F_combined) = ||ŷ - ArtifactScore(F_combined)||²₂

ArtifactScore(F_combined) = σ( (1/N) Σᵢ ArtifactDetectorᵢ(F_combined) )

Each ArtifactDetectorᵢ is a lightweight convolutional head:
  2× 3×3 depthwise-separable conv (128 channels) → GlobalAvgPool → sigmoid

Three safeguards prevent degenerate solutions where ArtifactScore mirrors ŷ:
  1. Stop-gradient on ŷ when computing R (gradients flow only through ArtifactScore)
  2. Pairwise cosine-dissimilarity penalty between detector penultimate activations
  3. Periodic frozen-detector evaluation (handled in the Trainer, not here)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ArtifactDetector
# ---------------------------------------------------------------------------

class ArtifactDetector(nn.Module):
    """
    Lightweight convolutional head that estimates artifact likelihood from
    F_combined ∈ R^D (1-D descriptor after CMAF mean-pooling).

    Because F_combined is already a 1-D vector, we treat it as a trivial
    1×1 spatial feature map and apply point-wise depthwise-separable layers,
    which reduce to:
        Linear(D, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 1) → Sigmoid

    The penultimate activation (before the final linear) is exposed as
    `self.penultimate` for diversity-loss computation.
    """

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.fc1       = nn.Linear(embed_dim, hidden_dim)
        self.bn1       = nn.BatchNorm1d(hidden_dim)
        self.fc2       = nn.Linear(hidden_dim, hidden_dim)
        self.bn2       = nn.BatchNorm1d(hidden_dim)
        self.fc_out    = nn.Linear(hidden_dim, 1)
        self.penultimate: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) frame-level CMAF descriptor
        Returns:
            score: (B,) artifact likelihood in [0, 1]
        """
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        self.penultimate = h                          # stored for diversity loss
        return torch.sigmoid(self.fc_out(h)).squeeze(1)


# ---------------------------------------------------------------------------
# AALF
# ---------------------------------------------------------------------------

class AALF(nn.Module):
    """
    Artifact-Aware Loss Function.

    Args:
        embed_dim        : Dimension of F_combined (D = 512)
        n_detectors      : N = 4 (ArtifactDetectors)
        detector_hidden  : Hidden dimension inside each ArtifactDetector
        lambda_reg       : λ — weight of the R regularisation term (0.1)
        w_real           : Weight for real class (1.0)
        w_fake           : Weight for fake class (2.0)
        diversity_weight : Weight of pairwise cosine-dissimilarity penalty (0.01)
    """

    def __init__(self,
                 embed_dim:        int   = 512,
                 n_detectors:      int   = 4,
                 detector_hidden:  int   = 128,
                 lambda_reg:       float = 0.1,
                 w_real:           float = 1.0,
                 w_fake:           float = 2.0,
                 diversity_weight: float = 0.01):
        super().__init__()

        self.n_detectors      = n_detectors
        self.lambda_reg       = lambda_reg
        self.w_real           = w_real
        self.w_fake           = w_fake
        self.diversity_weight = diversity_weight

        self.detectors = nn.ModuleList([
            ArtifactDetector(embed_dim, detector_hidden)
            for _ in range(n_detectors)
        ])

    def artifact_score(self, F_combined: torch.Tensor) -> torch.Tensor:
        """
        ArtifactScore(F_combined) = σ( (1/N) Σᵢ ArtifactDetectorᵢ(F_combined) )

        Note: each individual ArtifactDetector already applies sigmoid, so we
        compute the mean of sigmoid outputs then apply σ again as written in
        Equation (12).  This matches the manuscript's formulation.

        Args:
            F_combined: (B, D)
        Returns:
            score: (B,) ∈ [0,1]
        """
        scores = torch.stack(
            [det(F_combined) for det in self.detectors], dim=1
        )  # (B, N)
        mean_score = scores.mean(dim=1)   # (B,)
        return torch.sigmoid(mean_score)  # outer σ (Eq. 12)

    def _diversity_loss(self) -> torch.Tensor:
        """
        Pairwise cosine-dissimilarity penalty:
          L_div = -(1/N²) Σ_{i≠j} cos(vᵢ, vⱼ)

        Minimising −cosine_similarity encourages detectors to learn diverse
        feature representations.

        Penultimate activations are stored in ArtifactDetector.penultimate
        after the forward pass.
        """
        pens = [det.penultimate for det in self.detectors
                if det.penultimate is not None]
        if len(pens) < 2:
            return torch.tensor(0.0)

        pens = torch.stack(pens, dim=1)   # (B, N, H)
        # Normalise along the hidden dimension
        pens_norm = F.normalize(pens, dim=2)  # (B, N, H)
        # (B, N, N) cosine-similarity matrix
        sim = torch.bmm(pens_norm, pens_norm.transpose(1, 2))
        N   = self.n_detectors
        # Mask diagonal, sum off-diagonal
        mask = 1.0 - torch.eye(N, device=pens.device).unsqueeze(0)
        div_loss = -(sim * mask).sum() / (pens.size(0) * N * N)
        return div_loss

    def forward(self,
                y_hat:      torch.Tensor,
                y_true:     torch.Tensor,
                F_combined: torch.Tensor) -> dict:
        """
        Compute AALF.

        Args:
            y_hat      : (B,) predicted probabilities ∈ (0, 1)
            y_true     : (B,) ground-truth labels ∈ {0.0, 1.0}
            F_combined : (B, D) CMAF frame descriptor (used by ArtifactDetectors)

        Returns:
            dict with keys:
              'loss'       : total scalar loss
              'bce_loss'   : weighted BCE component
              'reg_loss'   : λ · R regularisation component
              'div_loss'   : diversity penalty component
        """
        eps = 1e-7
        y_hat_clipped = y_hat.clamp(eps, 1.0 - eps)

        # ── Weighted BCE (Eq. 10, first two terms) ───────────────────────
        bce = -(
            self.w_real * y_true       * torch.log(y_hat_clipped) +
            self.w_fake * (1 - y_true) * torch.log(1 - y_hat_clipped)
        ).mean()

        # ── Artifact regularisation R (Eq. 11) ───────────────────────────
        # STOP-GRADIENT on ŷ: gradients do NOT flow back through y_hat here.
        # Only ArtifactDetectors receive gradient from R.
        y_hat_sg   = y_hat.detach()          # stop-gradient
        art_score  = self.artifact_score(F_combined)
        reg = ((y_hat_sg - art_score) ** 2).mean()

        # ── Diversity penalty ─────────────────────────────────────────────
        div = self._diversity_loss()

        total = bce + self.lambda_reg * reg + self.diversity_weight * div

        return {
            "loss"     : total,
            "bce_loss" : bce.detach(),
            "reg_loss" : reg.detach(),
            "div_loss" : div.detach() if isinstance(div, torch.Tensor) else div,
        }
