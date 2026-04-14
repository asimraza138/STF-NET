"""
Integration test for STF-Net end-to-end forward pass.

Verifies that all components connect correctly without requiring
pretrained weights or GPU. Uses minimal dimensions for speed.

Run with:
    python -m pytest tests/test_integration.py -v
or:
    python tests/test_integration.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np


class TestTSFNetIntegration:
    """End-to-end forward pass through CMAF + TIAM + classifier."""

    @pytest.fixture(autouse=True)
    def build_model(self):
        """Build a tiny STF-Net (no pretrained weights, small dims for CPU speed)."""
        # We mock the heavy backbones by patching their forward methods
        from models.cmaf      import CMAF
        from models.tiam      import TIAM
        from models.aalf      import AALF
        from models.acs       import ComplexityEstimator

        self.B         = 2
        self.T         = 4
        self.embed_dim = 64
        self.hidden    = 32

        # Lightweight CMAF
        self.cmaf = CMAF(
            eff_channels=128, xcep_channels=64,
            embed_dim=self.embed_dim, num_heads=4, spatial_size=7,
        )
        # Lightweight TIAM
        self.tiam = TIAM(
            embed_dim=self.embed_dim, hidden_size=self.hidden,
            num_layers=1, dropout=0.0,
        )
        # Lightweight AALF
        self.aalf = AALF(
            embed_dim=self.embed_dim, n_detectors=2,
            detector_hidden=32,
        )
        # ACS estimator
        self.acs_est = ComplexityEstimator(pretrained=False)

    def test_cmaf_to_tiam_pipeline(self):
        """
        Simulate a B×T batch through CMAF → TIAM.
        """
        B, T, D = self.B, self.T, self.embed_dim
        feat_eff  = torch.randn(B * T, 128, 12, 12)
        feat_xcep = torch.randn(B * T, 64,  8,  8)

        # Per-frame CMAF
        F_combined = self.cmaf(feat_eff, feat_xcep)          # (B*T, D)
        seq        = F_combined.view(B, T, D)                  # (B, T, D)

        # Temporal attention
        tiam_out   = self.tiam(seq)                            # (B, 2H)
        assert tiam_out.shape == (B, 2 * self.hidden), \
            f"TIAM output shape mismatch: {tiam_out.shape}"

    def test_loss_backward(self):
        """
        Forward + backward through the full training graph:
        CMAF → TIAM → AALF, verify gradients exist for all parameters.
        """
        import torch.nn as nn
        B, T, D = self.B, self.T, self.embed_dim

        feat_eff  = torch.randn(B * T, 128, 12, 12)
        feat_xcep = torch.randn(B * T, 64,  8,  8)

        F_combined = self.cmaf(feat_eff, feat_xcep)
        seq        = F_combined.view(B, T, D)
        tiam_out   = self.tiam(seq)

        # Simple linear classifier head
        head  = nn.Linear(2 * self.hidden, 1)
        y_hat = torch.sigmoid(head(tiam_out).squeeze(1))
        y_true = torch.tensor([1., 0.])

        # AALF loss
        loss_dict = self.aalf(y_hat, y_true, F_combined[:B])  # last-frame descriptor
        loss      = loss_dict["loss"]

        loss.backward()

        # Verify gradient flow through CMAF
        for name, p in self.cmaf.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for CMAF param: {name}"

        # Verify gradient flow through TIAM
        for name, p in self.tiam.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for TIAM param: {name}"

    def test_variable_length_batch(self):
        """Verify the pipeline handles clips with different valid lengths."""
        B, T, D = 3, 8, self.embed_dim
        feat_eff  = torch.randn(B * T, 128, 12, 12)
        feat_xcep = torch.randn(B * T, 64,  8,  8)

        F_combined = self.cmaf(feat_eff, feat_xcep)
        seq        = F_combined.view(B, T, D)
        lengths    = torch.tensor([8, 5, 3])

        out = self.tiam(seq, lengths)
        assert out.shape == (B, 2 * self.hidden)
        assert not torch.isnan(out).any(), "NaN in variable-length TIAM output"

    def test_acs_estimator_inference(self):
        """ACS complexity estimator produces valid scores."""
        x     = torch.randn(self.B, 3, 224, 224)
        score = self.acs_est(x)
        assert score.shape == (self.B,)
        assert (score >= 0).all() and (score <= 1).all()

    def test_reproducibility_with_same_seed(self):
        """Same seed should produce identical outputs."""
        import random

        def run():
            torch.manual_seed(42)
            random.seed(42)
            np.random.seed(42)
            feat_eff  = torch.randn(4, 128, 12, 12)
            feat_xcep = torch.randn(4, 64,  8,  8)
            F = self.cmaf(feat_eff, feat_xcep)
            return F.detach()

        out1 = run()
        out2 = run()
        assert torch.allclose(out1, out2, atol=1e-6), \
            "Identical seeds produced different CMAF outputs"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    suite = TestTSFNetIntegration()
    methods = [m for m in dir(suite) if m.startswith("test_")]
    passed, failed = 0, []

    # Call build_model fixture manually
    suite.build_model()

    for method in methods:
        try:
            getattr(suite, method)()
            passed += 1
            print(f"  PASS  {method}")
        except Exception as e:
            failed.append(method)
            print(f"  FAIL  {method}: {e}")
            traceback.print_exc()

    print(f"\n{'='*55}")
    print(f"  {passed}/{len(methods)} integration tests passed")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"{'='*55}")
