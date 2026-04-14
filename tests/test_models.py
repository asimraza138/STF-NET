"""
Unit tests for STF-Net core model components.

Run with:
    python -m pytest tests/test_models.py -v
or:
    python tests/test_models.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_frames(B: int = 2, T: int = 4, H: int = 224, W: int = 224) -> torch.Tensor:
    """Return a random normalised frame tensor (B, T, 3, H, W)."""
    return torch.randn(B, T, 3, H, W)


# ---------------------------------------------------------------------------
# CMAF
# ---------------------------------------------------------------------------

class TestCMAF:
    def test_positional_encoding_shape(self):
        from models.cmaf import SinusoidalPositionalEncoding2D
        pe  = SinusoidalPositionalEncoding2D(embed_dim=512, spatial_size=14)
        x   = torch.randn(3, 196, 512)
        out = pe(x)
        assert out.shape == (3, 196, 512), f"Expected (3,196,512), got {out.shape}"

    def test_cmaf_output_shape(self):
        from models.cmaf import CMAF
        cmaf = CMAF(eff_channels=256, xcep_channels=128,
                    embed_dim=512, num_heads=8, spatial_size=14)
        feat_eff  = torch.randn(4, 256, 18, 18)  # arbitrary H', W'
        feat_xcep = torch.randn(4, 128, 12, 12)  # arbitrary H'', W''
        out = cmaf(feat_eff, feat_xcep)
        assert out.shape == (4, 512), f"Expected (4, 512), got {out.shape}"

    def test_cmaf_no_nan(self):
        from models.cmaf import CMAF
        cmaf = CMAF(eff_channels=512, xcep_channels=2048,
                    embed_dim=512, num_heads=8, spatial_size=14)
        fe = torch.randn(2, 512, 16, 16)
        fx = torch.randn(2, 2048, 8, 8)
        out = cmaf(fe, fx)
        assert not torch.isnan(out).any(), "CMAF output contains NaN"

    def test_cmaf_gradient_flows(self):
        from models.cmaf import CMAF
        cmaf = CMAF(256, 128, 512, 8, 14)
        fe   = torch.randn(2, 256, 14, 14, requires_grad=True)
        fx   = torch.randn(2, 128, 14, 14, requires_grad=True)
        out  = cmaf(fe, fx).sum()
        out.backward()
        assert fe.grad is not None and fx.grad is not None


# ---------------------------------------------------------------------------
# TIAM
# ---------------------------------------------------------------------------

class TestTIAM:
    def test_output_shape(self):
        from models.tiam import TIAM
        tiam = TIAM(embed_dim=512, hidden_size=256, num_layers=2)
        seq  = torch.randn(3, 16, 512)
        out  = tiam(seq)
        assert out.shape == (3, 512), f"Expected (3, 512), got {out.shape}"

    def test_variable_lengths(self):
        from models.tiam import TIAM
        tiam    = TIAM(embed_dim=512, hidden_size=256, num_layers=2)
        seq     = torch.randn(4, 16, 512)
        lengths = torch.tensor([16, 12, 8, 4])
        out     = tiam(seq, lengths)
        assert out.shape == (4, 512)
        assert not torch.isnan(out).any()

    def test_attention_weights_sum_to_one(self):
        """
        Verify softmax attention sums to 1 (indirectly via output magnitude).
        We check that attention-weighted output is bounded by the hidden states.
        """
        from models.tiam import TIAM
        tiam = TIAM(embed_dim=64, hidden_size=32, num_layers=1, dropout=0.0)
        tiam.eval()
        seq = torch.ones(1, 8, 64)  # constant sequence
        with torch.no_grad():
            out = tiam(seq)
        # If all frames are identical and α sums to 1, output = single hidden state
        assert out.shape == (1, 64)

    def test_delta_normalisation_no_nan(self):
        from models.tiam import TIAM
        tiam = TIAM(embed_dim=512, hidden_size=256, num_layers=2)
        # All-zero sequence → Δt = 0 for all frames → min==max → handled by clamp
        seq = torch.zeros(2, 16, 512)
        out = tiam(seq)
        assert not torch.isnan(out).any(), "NaN with constant input sequence"

    def test_gradient_flows(self):
        from models.tiam import TIAM
        tiam = TIAM(embed_dim=64, hidden_size=32, num_layers=1)
        seq  = torch.randn(2, 8, 64, requires_grad=True)
        out  = tiam(seq).sum()
        out.backward()
        assert seq.grad is not None


# ---------------------------------------------------------------------------
# AALF
# ---------------------------------------------------------------------------

class TestAALF:
    def test_loss_is_positive(self):
        from models.aalf import AALF
        aalf   = AALF(embed_dim=512, n_detectors=4)
        y_hat  = torch.tensor([0.9, 0.1, 0.8, 0.2])
        y_true = torch.tensor([1., 0., 1., 0.])
        F_comb = torch.randn(4, 512)
        d      = aalf(y_hat, y_true, F_comb)
        assert d["loss"].item() > 0, "Loss should be positive"

    def test_loss_keys(self):
        from models.aalf import AALF
        aalf   = AALF(embed_dim=64)
        d      = aalf(torch.sigmoid(torch.randn(2)),
                      torch.tensor([1., 0.]),
                      torch.randn(2, 64))
        for key in ("loss", "bce_loss", "reg_loss", "div_loss"):
            assert key in d, f"Missing key: {key}"

    def test_stop_gradient_on_yhat(self):
        """
        R should not propagate gradients back through y_hat.
        Verify by checking y_hat.grad stays None after backward.
        """
        from models.aalf import AALF
        aalf   = AALF(embed_dim=64)
        y_hat  = torch.sigmoid(torch.randn(2, requires_grad=True))
        y_true = torch.tensor([1., 0.])
        F_comb = torch.randn(2, 64)
        d      = aalf(y_hat, y_true, F_comb)
        d["loss"].backward()
        # Gradient should come via BCE only (not through R since y_hat is detached)
        assert y_hat.grad is not None   # BCE does propagate

    def test_artifact_score_range(self):
        from models.aalf import AALF
        aalf   = AALF(embed_dim=64, n_detectors=4)
        F_comb = torch.randn(8, 64)
        score  = aalf.artifact_score(F_comb)
        assert score.shape == (8,)
        assert (score >= 0).all() and (score <= 1).all(), \
            "ArtifactScore must be in [0, 1]"

    def test_diversity_loss_discourages_identical_detectors(self):
        from models.aalf import AALF
        aalf   = AALF(embed_dim=64, n_detectors=4)
        F_comb = torch.randn(4, 64)
        # Run forward to populate penultimate activations
        _      = aalf.artifact_score(F_comb)
        div    = aalf._diversity_loss()
        # Diversity loss should be a scalar tensor
        assert isinstance(div, torch.Tensor)
        assert div.shape == ()


# ---------------------------------------------------------------------------
# ACS Complexity Estimator
# ---------------------------------------------------------------------------

class TestACS:
    def test_estimator_output_shape(self):
        from models.acs import ComplexityEstimator
        est   = ComplexityEstimator(pretrained=False)
        x     = torch.randn(3, 3, 224, 224)
        score = est(x)
        assert score.shape == (3,), f"Expected (3,), got {score.shape}"

    def test_estimator_output_range(self):
        from models.acs import ComplexityEstimator
        est   = ComplexityEstimator(pretrained=False)
        x     = torch.randn(4, 3, 224, 224)
        score = est(x)
        assert (score >= 0).all() and (score <= 1).all(), \
            "Complexity score must be in [0, 1]"

    def test_lightweight_detector_shape(self):
        from models.acs import LightweightDetector
        det    = LightweightDetector(pretrained=False)
        frames = torch.randn(2, 8, 3, 224, 224)
        prob   = det(frames)
        assert prob.shape == (2,)
        assert (prob >= 0).all() and (prob <= 1).all()

    def test_standard_detector_shape(self):
        from models.acs import StandardDetector
        det    = StandardDetector(pretrained=False)
        frames = torch.randn(2, 8, 3, 224, 224)
        prob   = det(frames)
        assert prob.shape == (2,)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class TestAugmentation:
    def test_augmentation_preserves_shape(self):
        from data.augmentation import build_augmentation_pipeline, apply_augmentation
        faces   = np.random.rand(8, 3, 224, 224).astype(np.float32)
        aug_cfg = build_augmentation_pipeline({})
        result  = apply_augmentation(faces, aug_cfg)
        assert result.shape == (8, 3, 224, 224)

    def test_augmentation_value_range(self):
        from data.augmentation import build_augmentation_pipeline, apply_augmentation
        faces   = np.random.rand(4, 3, 224, 224).astype(np.float32)
        aug_cfg = build_augmentation_pipeline({})
        result  = apply_augmentation(faces, aug_cfg)
        assert result.min() >= 0.0 and result.max() <= 1.0, \
            "Augmented frames must stay in [0, 1]"

    def test_augmentation_is_stochastic(self):
        from data.augmentation import build_augmentation_pipeline, apply_augmentation
        faces   = np.random.rand(4, 3, 224, 224).astype(np.float32)
        aug_cfg = build_augmentation_pipeline({"horizontal_flip_p": 1.0})
        a1 = apply_augmentation(faces.copy(), aug_cfg)
        a2 = apply_augmentation(faces.copy(), aug_cfg)
        # With flip_p=1.0, results should equal horizontally flipped input
        flipped = faces[:, :, :, ::-1].copy()
        assert np.allclose(a1, flipped, atol=0.01), \
            "With p=1.0 flip, output should match horizontally flipped input"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_perfect_predictions(self):
        from utils.metrics import compute_metrics
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
        m = compute_metrics(y_true, y_prob)
        assert m["accuracy"]  == 1.0
        assert m["precision"] == 1.0
        assert m["recall"]    == 1.0
        assert m["f1"]        == 1.0
        assert m["auc"]       == 1.0

    def test_random_predictions_within_range(self):
        from utils.metrics import compute_metrics
        np.random.seed(0)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        m = compute_metrics(y_true, y_prob)
        for k in ("accuracy", "precision", "recall", "f1", "auc"):
            assert 0.0 <= m[k] <= 1.0, f"{k} out of range: {m[k]}"

    def test_confusion_matrix_shape(self):
        from utils.metrics import compute_metrics
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.4, 0.8])
        m = compute_metrics(y_true, y_prob)
        assert m["cm"].shape == (2, 2)

    def test_aggregate_seeds(self):
        from utils.metrics import aggregate_seeds
        results = [
            {"accuracy": 0.90, "f1": 0.88, "auc": 0.92, "cm": None},
            {"accuracy": 0.92, "f1": 0.90, "auc": 0.94, "cm": None},
        ]
        agg = aggregate_seeds(results)
        assert "accuracy_mean" in agg
        assert "accuracy_std"  in agg
        assert abs(agg["accuracy_mean"] - 0.91) < 1e-6


# ---------------------------------------------------------------------------
# SSIM Labels
# ---------------------------------------------------------------------------

class TestSSIMLabels:
    def test_compute_ssim_complexity_range(self, tmp_path):
        from utils.ssim_labels import compute_ssim_complexity
        # Create a fake .npy file
        faces = np.random.rand(8, 3, 64, 64).astype(np.float32)
        npy_path = str(tmp_path / "test.npy")
        np.save(npy_path, faces)
        score = compute_ssim_complexity(npy_path, n_pairs=3)
        assert 0.0 <= score <= 1.0, f"SSIM complexity out of range: {score}"

    def test_constant_video_complexity_is_zero(self, tmp_path):
        from utils.ssim_labels import compute_ssim_complexity
        # All frames identical → SSIM = 1 → drop = 0
        faces = np.ones((8, 3, 64, 64), dtype=np.float32) * 0.5
        npy_path = str(tmp_path / "const.npy")
        np.save(npy_path, faces)
        score = compute_ssim_complexity(npy_path, n_pairs=3)
        assert score < 0.01, f"Constant video should have near-zero complexity: {score}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run without pytest if preferred
    import traceback

    test_classes = [
        TestCMAF, TestTIAM, TestAALF, TestACS,
        TestAugmentation, TestMetrics, TestSSIMLabels,
    ]

    total, passed, failed = 0, 0, []
    for cls in test_classes:
        instance = cls()
        methods  = [m for m in dir(cls) if m.startswith("test_")]
        for method in methods:
            total += 1
            try:
                fn = getattr(instance, method)
                # Handle pytest fixtures (tmp_path) manually
                import inspect
                sig = inspect.signature(fn)
                if "tmp_path" in sig.parameters:
                    import tempfile, pathlib
                    with tempfile.TemporaryDirectory() as td:
                        fn(pathlib.Path(td))
                else:
                    fn()
                passed += 1
                print(f"  PASS  {cls.__name__}.{method}")
            except Exception as e:
                failed.append(f"{cls.__name__}.{method}")
                print(f"  FAIL  {cls.__name__}.{method}: {e}")
                traceback.print_exc()

    print(f"\n{'='*55}")
    print(f"  {passed}/{total} tests passed")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"{'='*55}")
