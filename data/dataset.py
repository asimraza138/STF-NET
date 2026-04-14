"""
Dataset classes for FaceForensics++ and DFDC — Section 4.1 of the manuscript.

Both datasets are expected to have been preprocessed (face extraction via MTCNN)
using scripts/preprocess_data.py before training.

Directory layout expected after preprocessing
---------------------------------------------
data/processed/
├── ffpp/
│   ├── train/
│   │   ├── real/    *.npy  (T, 3, 224, 224)
│   │   └── fake/    *.npy
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
└── dfdc/
    ├── train/
    │   ├── real/
    │   └── fake/
    ├── val/
    └── test/

Each .npy file contains a (T, 3, H, W) float32 face-frame sequence in [0, 1].
"""

import os
import random
import numpy as np
from pathlib import Path
from typing import Literal, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from .augmentation import build_augmentation_pipeline, apply_augmentation


# ---------------------------------------------------------------------------
# Base deepfake video dataset
# ---------------------------------------------------------------------------

class DeepfakeDataset(Dataset):
    """
    Generic deepfake video dataset.

    Args:
        root       : root directory (e.g. 'data/processed/ffpp/train')
        split      : 'train', 'val', or 'test'
        n_frames   : frames to use per clip (16 train, 32 inference)
        augment    : whether to apply augmentation (only during training)
        cfg        : augmentation config dict (from default.yaml)
    """

    def __init__(self,
                 root:     str,
                 split:    Literal["train", "val", "test"],
                 n_frames: int  = 16,
                 augment:  bool = False,
                 cfg:      Optional[dict] = None):
        self.n_frames = n_frames
        self.augment  = augment
        self.aug_cfg  = cfg or {}

        real_dir = os.path.join(root, split, "real")
        fake_dir = os.path.join(root, split, "fake")

        real_files = sorted(Path(real_dir).glob("*.npy"))
        fake_files = sorted(Path(fake_dir).glob("*.npy"))

        self.samples = (
            [(str(f), 0) for f in real_files] +
            [(str(f), 1) for f in fake_files]
        )
        random.shuffle(self.samples)

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No .npy files found under {real_dir} or {fake_dir}. "
                "Run scripts/preprocess_data.py first."
            )

        if augment:
            self.augmentor = build_augmentation_pipeline(self.aug_cfg)
        else:
            self.augmentor = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.samples[idx]

        # Load precomputed face sequence
        faces = np.load(path)   # (T_orig, 3, H, W) float32

        # Uniform sub-sample to n_frames
        T_orig = faces.shape[0]
        if T_orig >= self.n_frames:
            indices = np.linspace(0, T_orig - 1, self.n_frames, dtype=int)
        else:
            indices = list(range(T_orig))
            # Cyclic padding
            while len(indices) < self.n_frames:
                indices.append(indices[len(indices) % T_orig])
        faces = faces[indices]   # (n_frames, 3, H, W)

        # Augmentation (applied per-frame consistently within a clip)
        if self.augmentor is not None:
            faces = apply_augmentation(faces, self.augmentor)

        frames = torch.from_numpy(faces)   # (T, 3, H, W) float32

        # ImageNet normalisation: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        frames = (frames - mean) / std

        return {
            "frames" : frames,               # (T, 3, H, W)
            "label"  : torch.tensor(float(label)),
            "path"   : path,
            "length" : torch.tensor(self.n_frames, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Combined DFDC + FF++ dataset for joint training
# ---------------------------------------------------------------------------

class CombinedDataset(Dataset):
    """
    Combines DFDC and FF++ with stratified per-batch sampling
    (equal representation of each dataset per batch, as per Section 4.1).

    Usage:
        combined = CombinedDataset(dfdc_root, ffpp_root, split="train", ...)
    """

    def __init__(self,
                 dfdc_root: str,
                 ffpp_root: str,
                 split:     Literal["train", "val", "test"],
                 n_frames:  int  = 16,
                 augment:   bool = False,
                 cfg:       Optional[dict] = None):
        self.dfdc = DeepfakeDataset(dfdc_root, split, n_frames, augment, cfg)
        self.ffpp = DeepfakeDataset(ffpp_root, split, n_frames, augment, cfg)
        # Interleave indices so batches contain both datasets
        n = max(len(self.dfdc), len(self.ffpp))
        self.indices = []
        for i in range(n):
            self.indices.append(("dfdc", i % len(self.dfdc)))
            self.indices.append(("ffpp", i % len(self.ffpp)))
        random.shuffle(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        src, i = self.indices[idx]
        if src == "dfdc":
            return self.dfdc[i]
        return self.ffpp[i]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(cfg:       dict,
                      use_combined: bool = True) -> dict:
    """
    Build train / val / test DataLoaders according to the configuration.

    Args:
        cfg          : full config dict loaded from default.yaml
        use_combined : if True, train on DFDC + FF++ jointly
    Returns:
        dict with keys 'train', 'val', 'test'
    """
    data_cfg = cfg["data"]
    aug_cfg  = cfg["augmentation"]
    paths    = cfg["paths"]

    ffpp_root = os.path.join(paths["processed_root"], "ffpp")
    dfdc_root = os.path.join(paths["processed_root"], "dfdc")

    train_frames = data_cfg["train_frames"]
    infer_frames = data_cfg["infer_frames"]
    bs           = cfg["training"]["batch_size_per_gpu"]

    # Training set
    if use_combined:
        train_ds = CombinedDataset(
            dfdc_root, ffpp_root, split="train",
            n_frames=train_frames, augment=True, cfg=aug_cfg,
        )
    else:
        train_ds = DeepfakeDataset(
            ffpp_root, split="train",
            n_frames=train_frames, augment=True, cfg=aug_cfg,
        )

    val_ds  = DeepfakeDataset(ffpp_root, "val",  infer_frames, augment=False)
    test_ds = DeepfakeDataset(ffpp_root, "test", infer_frames, augment=False)

    num_workers = min(8, os.cpu_count() or 4)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
