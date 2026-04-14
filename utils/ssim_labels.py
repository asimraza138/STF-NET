"""
SSIM-based complexity label generation — Section 3.5 / §8 of the revision guide.

For each video in the training set, a scalar complexity label is computed as:
    C_proxy = mean SSIM drop across the first 5 consecutive frame pairs.

SSIM drop = 1 − SSIM(frame_t, frame_{t+1})

Higher C_proxy → temporally more complex video → warrants full-model processing.

The labels are min-max normalised to [0, 1] across all training videos and
saved as a JSON file for later use in ACS estimator training.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional

from skimage.metrics import structural_similarity as ssim_skimage
from tqdm import tqdm


def compute_ssim_complexity(npy_path: str,
                             n_pairs:  int = 5) -> float:
    """
    Compute the mean SSIM-drop complexity label for a preprocessed video.

    Args:
        npy_path : path to .npy file of shape (T, 3, H, W) float32 in [0,1]
        n_pairs  : number of consecutive frame pairs to average over
    Returns:
        complexity: float ∈ [0, 1]  (before global normalisation)
    """
    faces = np.load(npy_path)   # (T, 3, H, W)
    T     = faces.shape[0]
    n_actual = min(n_pairs, T - 1)
    if n_actual <= 0:
        return 0.0

    ssim_drops = []
    for t in range(n_actual):
        f1 = faces[t].transpose(1, 2, 0)      # (H, W, 3)
        f2 = faces[t + 1].transpose(1, 2, 0)

        # SSIM over RGB channels (channel_axis=-1 supported in scikit-image ≥ 0.19)
        try:
            sim = ssim_skimage(f1, f2, data_range=1.0, channel_axis=-1)
        except TypeError:
            # Older scikit-image: multichannel keyword
            sim = ssim_skimage(f1, f2, data_range=1.0, multichannel=True)

        ssim_drops.append(1.0 - float(sim))

    return float(np.mean(ssim_drops))


def generate_complexity_labels(processed_dir: str,
                                output_json:   str,
                                n_pairs:       int = 5):
    """
    Generate and save SSIM-based complexity labels for all videos in a directory.

    Args:
        processed_dir : root processed directory (e.g. 'data/processed')
        output_json   : path to save the label JSON file
        n_pairs       : number of consecutive frame pairs to use
    """
    npy_files = sorted(Path(processed_dir).rglob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found under {processed_dir}")

    raw_labels = {}
    for fpath in tqdm(npy_files, desc="Computing SSIM complexity"):
        key = str(fpath)
        raw_labels[key] = compute_ssim_complexity(str(fpath), n_pairs)

    # Min-max normalise across all videos
    vals   = np.array(list(raw_labels.values()))
    v_min  = float(vals.min())
    v_max  = float(vals.max())
    denom  = max(v_max - v_min, 1e-8)

    normalised = {
        k: float((v - v_min) / denom)
        for k, v in raw_labels.items()
    }

    os.makedirs(Path(output_json).parent, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(normalised, f, indent=2)

    print(f"Saved {len(normalised)} complexity labels to {output_json}")
    print(f"  Raw range: [{v_min:.4f}, {v_max:.4f}]")
    return normalised


def load_complexity_labels(json_path: str) -> dict:
    """Load pre-computed complexity labels from a JSON file."""
    with open(json_path) as f:
        return json.load(f)
