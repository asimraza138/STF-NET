"""
Temporal Inconsistency Attention Module (TIAM) — Section 3.3 of the manuscript.

Takes a sequence of per-frame descriptors {F^t_combined}, processes through a
2-layer bidirectional LSTM, computes frame-difference-guided attention weights
αt, and returns a single attended video-level representation.

Key implementation details:
  - Δt = ||F^t_combined − F^(t−1)_combined||₂  (scalar, L2 norm)
  - Δt is min-max normalised across the video sequence
  - et = Wα · [Ht; Δ̂t] + bα       (scalar energy for each frame)
  - αt = softmax(e1, ..., eT)       (softmax over T frames)
  - TIAM_output = Σ_t αt · Ht       (weighted sum)
  - Variable-length videos: zero-padding + attention mask on padded frames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TIAM(nn.Module):
    """
    Temporal Inconsistency Attention Module.

    Args:
        embed_dim   : Dimension D of each frame descriptor F_combined^t
        hidden_size : Hidden units per direction in Bi-LSTM (256 in manuscript)
        num_layers  : Number of stacked Bi-LSTM layers (2 in manuscript)
        dropout     : Dropout between Bi-LSTM layers (0.5 in manuscript)
    """

    def __init__(self,
                 embed_dim:   int = 512,
                 hidden_size: int = 256,
                 num_layers:  int = 2,
                 dropout:     float = 0.5):
        super().__init__()

        self.embed_dim   = embed_dim
        self.hidden_size = hidden_size
        # Bi-LSTM output dimension = 2 * hidden_size
        self.bilstm_dim  = 2 * hidden_size

        # Bi-LSTM: processes the sequence of frame descriptors
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Temporal attention projection:
        # W_α ∈ R^{1 × (bilstm_dim + 1)},  bα ∈ R^1
        # Input: [H_t; Δ̂_t]  where H_t ∈ R^{bilstm_dim}, Δ̂_t ∈ R^1
        self.attn_proj = nn.Linear(self.bilstm_dim + 1, 1, bias=True)

    def _compute_normalised_delta(self,
                                   seq: torch.Tensor,
                                   mask: torch.Tensor | None) -> torch.Tensor:
        """
        Compute per-frame scalar frame difference Δ̂_t.

        Args:
            seq  : (B, T, D) frame descriptor sequence
            mask : (B, T) boolean mask — True means *valid* frame
        Returns:
            delta_hat: (B, T, 1) min-max normalised frame differences
        """
        B, T, D = seq.shape

        # ||F^t - F^(t-1)||_2  (scalar per frame)
        # For t=0: difference with itself → 0
        diff = seq[:, 1:, :] - seq[:, :-1, :]          # (B, T-1, D)
        delta = diff.norm(dim=2)                         # (B, T-1)
        # Prepend 0 for the first frame
        first = torch.zeros(B, 1, device=seq.device)
        delta = torch.cat([first, delta], dim=1)         # (B, T)

        # Min-max normalisation per video (ignoring padded frames)
        if mask is not None:
            # Set padded-frame deltas to NaN so they don't affect min/max
            delta_for_stat = delta.clone()
            delta_for_stat[~mask] = float("nan")
            d_min = torch.nanmin(delta_for_stat, dim=1, keepdim=True).values  # (B,1)
            d_max = torch.nanmax(delta_for_stat, dim=1, keepdim=True).values
        else:
            d_min = delta.min(dim=1, keepdim=True).values
            d_max = delta.max(dim=1, keepdim=True).values

        denom = (d_max - d_min).clamp(min=1e-8)
        delta_hat = (delta - d_min) / denom              # (B, T)

        # Zero out padded positions
        if mask is not None:
            delta_hat = delta_hat.masked_fill(~mask, 0.0)

        return delta_hat.unsqueeze(2)                    # (B, T, 1)

    def forward(self,
                seq:    torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            seq     : (B, T, D)  sequence of CMAF frame descriptors
            lengths : (B,) int64 actual number of valid frames per clip;
                      None means all T frames are valid for every clip.
        Returns:
            tiam_out: (B, bilstm_dim) attended temporal representation
        """
        B, T, D = seq.shape
        device  = seq.device

        # Build padding mask: True = valid, False = padded
        if lengths is not None:
            mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        else:
            mask = None

        # ── Bi-LSTM ──────────────────────────────────────────────────────
        if lengths is not None:
            # Pack for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                seq, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out_packed, _ = self.bilstm(packed)
            H, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out_packed, batch_first=True, total_length=T
            )   # (B, T, bilstm_dim)
        else:
            H, _ = self.bilstm(seq)   # (B, T, bilstm_dim)

        # ── Frame differences Δ̂_t ────────────────────────────────────────
        delta_hat = self._compute_normalised_delta(seq, mask)  # (B, T, 1)

        # ── Temporal attention (Eqs 8 & 9) ───────────────────────────────
        # et = Wα · [Ht; Δ̂t] + bα
        concat = torch.cat([H, delta_hat], dim=2)   # (B, T, bilstm_dim+1)
        e = self.attn_proj(concat).squeeze(2)        # (B, T)

        # Mask padded frames so they receive zero attention weight
        if mask is not None:
            e = e.masked_fill(~mask, float("-inf"))

        alpha = F.softmax(e, dim=1)   # (B, T)  — softmax over T

        # ── Weighted sum (TIAM_output = Σ_t αt · Ht) ─────────────────────
        tiam_out = (alpha.unsqueeze(2) * H).sum(dim=1)   # (B, bilstm_dim)
        return tiam_out
