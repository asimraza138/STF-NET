"""
Backbone feature extractors for TSF-Net.

EfficientNetV2L  – global texture feature extractor (via timm).
XceptionNet      – local pixel-level artifact extractor, modified per the
                   manuscript: middle-flow repetitions 8 → 12, extra residual
                   connections, modified final pooling to retain spatial info.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ---------------------------------------------------------------------------
# EfficientNetV2L wrapper
# ---------------------------------------------------------------------------

class EfficientNetV2LBackbone(nn.Module):
    """
    EfficientNetV2L pretrained on ImageNet.
    Returns feature map E(f_i) ∈ R^{H' x W' x D_E}.
    The classification head is removed; spatial feature maps are returned.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            "tf_efficientnetv2_l",
            pretrained=pretrained,
            features_only=True,   # return intermediate feature maps
        )
        # Use the final feature stage only
        self.out_channels = self.model.feature_info[-1]["num_chs"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalised frame tensor
        Returns:
            feat: (B, D_E, H', W')
        """
        features = self.model(x)
        return features[-1]   # last stage feature map


# ---------------------------------------------------------------------------
# Modified XceptionNet
# ---------------------------------------------------------------------------

class SeparableConv2d(nn.Module):
    """Depthwise-separable convolution block used throughout XceptionNet."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride,
                                   padding, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class XceptionBlock(nn.Module):
    """
    Xception residual block with two separable convolutions and an
    optional skip-projection.  Used in Entry and Exit flows.
    """

    def __init__(self, in_ch: int, out_ch: int, reps: int,
                 stride: int = 1, start_with_relu: bool = True,
                 grow_first: bool = True):
        super().__init__()
        mid_ch = out_ch if grow_first else in_ch

        layers = []
        for i in range(reps):
            ch_in  = in_ch  if i == 0 else mid_ch
            ch_out = mid_ch if i < reps - 1 else out_ch
            if start_with_relu or i > 0:
                layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv2d(ch_in, ch_out))
            layers.append(nn.BatchNorm2d(ch_out))
        if stride > 1:
            layers.append(nn.MaxPool2d(3, stride, 1))
        self.rep = nn.Sequential(*layers)

        # Skip connection
        if in_ch != out_ch or stride > 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x) if self.skip else x
        return self.rep(x) + residual


class MiddleFlowBlock(nn.Module):
    """
    Middle-flow block: three 3×3 separable convolutions with a residual skip.
    An extra residual connection is added after the second convolution (manuscript
    modification) to preserve fine-grained spatial details.
    """

    def __init__(self, channels: int = 728):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConv2d(channels, channels),
            nn.BatchNorm2d(channels),
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConv2d(channels, channels),
            nn.BatchNorm2d(channels),
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConv2d(channels, channels),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out  = self.conv1(x)
        out  = self.conv2(out)
        out  = out + x          # extra residual — manuscript modification
        out  = self.conv3(out)
        return out + x          # outer residual (standard Xception)


class ModifiedXceptionNet(nn.Module):
    """
    XceptionNet modified for deepfake detection (manuscript Section 3.1.2):
      - Middle-flow repetitions: 8 → 12
      - Extra residual connections inside each middle-flow block
      - Final pooling replaced with AdaptiveAvgPool2d to retain spatial dimensions
        (no classification head; raw spatial feature maps are returned)

    Returns feature map X(f_i) ∈ R^{H'' x W'' x D_X}.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Entry flow
        self.entry_flow = nn.Sequential(
            # Block 0: initial convolutions
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.entry_block1 = XceptionBlock(64,  128, reps=2, stride=2, start_with_relu=False, grow_first=True)
        self.entry_block2 = XceptionBlock(128, 256, reps=2, stride=2, grow_first=True)
        self.entry_block3 = XceptionBlock(256, 728, reps=2, stride=2, grow_first=True)

        # Middle flow — 12 repetitions (manuscript: 8 → 12)
        self.middle_flow = nn.Sequential(
            *[MiddleFlowBlock(728) for _ in range(12)]
        )

        # Exit flow — modified final pooling
        self.exit_block = XceptionBlock(728, 1024, reps=2, stride=2, grow_first=False)
        self.exit_sep1  = nn.Sequential(
            SeparableConv2d(1024, 1536), nn.BatchNorm2d(1536), nn.ReLU(inplace=True),
        )
        self.exit_sep2  = nn.Sequential(
            SeparableConv2d(1536, 2048), nn.BatchNorm2d(2048), nn.ReLU(inplace=True),
        )
        # Retain spatial information (manuscript modification — no GlobalAvgPool)
        self.out_channels = 2048

        if pretrained:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """
        Load ImageNet-pretrained Xception weights from timm, mapping compatible
        layers.  Layers that differ (extra residual connections, middle-flow
        count) are initialised from scratch.
        """
        try:
            pretrained = timm.create_model("xception", pretrained=True)
            state_pretrained = pretrained.state_dict()
            state_ours = self.state_dict()

            matched, skipped = 0, 0
            new_state = {}
            for k, v in state_ours.items():
                if k in state_pretrained and state_pretrained[k].shape == v.shape:
                    new_state[k] = state_pretrained[k]
                    matched += 1
                else:
                    new_state[k] = v   # keep random init
                    skipped += 1
            self.load_state_dict(new_state, strict=False)
            print(f"[XceptionNet] Loaded {matched} pretrained layers, "
                  f"{skipped} layers initialised from scratch.")
        except Exception as e:
            print(f"[XceptionNet] Could not load pretrained weights: {e}. "
                  "Training from scratch.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            feat: (B, 2048, H'', W'')
        """
        x = self.entry_flow(x)
        x = self.entry_block1(x)
        x = self.entry_block2(x)
        x = self.entry_block3(x)
        x = self.middle_flow(x)
        x = self.exit_block(x)
        x = self.exit_sep1(x)
        x = self.exit_sep2(x)
        return x
