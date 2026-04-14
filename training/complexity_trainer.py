"""
ACS Complexity Estimator Trainer — Stage 1 of the two-stage training pipeline.

The complexity estimator is trained independently on SSIM-based proxy labels
before main TSF-Net training begins.  Once trained, its weights are frozen and
only used for routing decisions at inference time.

Training objective: MSE regression against normalised SSIM-drop scores.
"""

import os
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.acs       import ComplexityEstimator
from utils.logging_utils import setup_logger


# ---------------------------------------------------------------------------
# Complexity Dataset
# ---------------------------------------------------------------------------

class ComplexityDataset(Dataset):
    """
    Dataset pairing the *first frame* of each video with its SSIM complexity label.

    Args:
        npy_dir      : directory containing preprocessed .npy files
        label_json   : JSON file with {npy_path: complexity_label} mapping
    """

    def __init__(self, npy_dir: str, label_json: str):
        with open(label_json) as f:
            all_labels = json.load(f)

        self.samples = []
        for fpath in Path(npy_dir).rglob("*.npy"):
            key = str(fpath)
            if key in all_labels:
                self.samples.append((key, float(all_labels[key])))

        if not self.samples:
            raise FileNotFoundError(
                f"No matching entries found for files in {npy_dir} "
                f"within {label_json}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        path, label = self.samples[idx]
        faces = np.load(path)    # (T, 3, H, W)
        first = torch.from_numpy(faces[0])   # (3, H, W)

        # ImageNet normalisation
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        first = (first - mean) / std

        return first, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def train_complexity_estimator(cfg: dict, log_dir: str = "logs") -> str:
    """
    Train the ACS complexity estimator and save checkpoint.

    Args:
        cfg     : full config dict from default.yaml
        log_dir : directory for log files
    Returns:
        checkpoint_path: path to the saved estimator weights
    """
    logger   = setup_logger("complexity_estimator", log_dir)
    acs_cfg  = cfg["acs"]
    paths    = cfg["paths"]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Dataset ───────────────────────────────────────────────────────────
    label_json = os.path.join(paths["complexity_labels"], "labels.json")
    train_ds   = ComplexityDataset(
        npy_dir=os.path.join(paths["processed_root"], "ffpp", "train"),
        label_json=label_json,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=acs_cfg["estimator_batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    logger.info(f"Complexity estimator training set: {len(train_ds)} samples")

    # ── Model ─────────────────────────────────────────────────────────────
    model = ComplexityEstimator(pretrained=True).to(device)
    optimizer = Adam(model.parameters(), lr=acs_cfg["estimator_lr"])
    scheduler = CosineAnnealingLR(
        optimizer, T_max=acs_cfg["estimator_epochs"], eta_min=1e-5
    )
    criterion = nn.MSELoss()

    # ── Training loop ──────────────────────────────────────────────────────
    best_loss = float("inf")
    ckpt_dir  = os.path.join(paths["checkpoints"], "complexity_estimator")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best.pt")

    for epoch in range(1, acs_cfg["estimator_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch}",
                                    leave=False, ncols=80):
            frames = frames.to(device)
            labels = labels.to(device)

            preds = model(frames)
            loss  = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * frames.size(0)

        scheduler.step()
        avg_loss = epoch_loss / len(train_ds)
        logger.info(f"Epoch {epoch:3d} | MSE Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  → Checkpoint saved ({ckpt_path})")

    logger.info(f"Complexity estimator training complete. Best MSE: {best_loss:.6f}")
    return ckpt_path
