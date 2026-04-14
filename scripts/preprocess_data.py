#!/usr/bin/env python3
"""
Data Preprocessing Script — STF-Net

Converts raw video files (FF++ and DFDC) into preprocessed face-frame
sequences (.npy files) ready for training and evaluation.

Usage
-----
# Preprocess FaceForensics++ (c23, HQ subset):
python scripts/preprocess_data.py \
    --dataset ffpp \
    --raw_root data/raw/FaceForensics++ \
    --out_root data/processed/ffpp \
    --n_frames 32

# Preprocess DFDC:
python scripts/preprocess_data.py \
    --dataset dfdc \
    --raw_root data/raw/DFDC \
    --out_root data/processed/dfdc \
    --n_frames 32

# Generate SSIM complexity labels for ACS estimator training:
python scripts/preprocess_data.py --ssim_labels \
    --processed_root data/processed \
    --output_json data/complexity_labels/labels.json
"""

import os
import sys
import argparse
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from data.preprocessing import build_mtcnn, preprocess_dataset
from utils.ssim_labels  import generate_complexity_labels


# ---------------------------------------------------------------------------
# FF++ helpers
# ---------------------------------------------------------------------------

FFPP_MANIPULATION_TYPES = [
    "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"
]

# Official FF++ split sizes
FFPP_SPLIT = {"train": 720, "val": 140, "test": 140}


def collect_ffpp_videos(raw_root: str, subset: str = "c23") -> dict:
    """
    Collect FF++ video paths and assign real/fake labels.

    Expected directory structure (official FF++ layout):
        raw_root/
        ├── original_sequences/
        │   └── youtube/
        │       └── c23/videos/   *.mp4
        └── manipulated_sequences/
            ├── Deepfakes/
            │   └── c23/videos/   *.mp4
            ├── Face2Face/
            │   └── c23/videos/
            ├── FaceSwap/
            │   └── c23/videos/
            └── NeuralTextures/
                └── c23/videos/

    Returns:
        dict with keys 'train', 'val', 'test', each a list of (path, label) tuples
    """
    real_dir = os.path.join(raw_root, "original_sequences", "youtube",
                             subset, "videos")
    real_videos = sorted(Path(real_dir).glob("*.mp4"))
    n_real      = len(real_videos)
    if n_real == 0:
        print(f"[Warning] No real videos found in {real_dir}")

    fake_videos = []
    for mtype in FFPP_MANIPULATION_TYPES:
        fake_dir = os.path.join(raw_root, "manipulated_sequences",
                                 mtype, subset, "videos")
        vids = sorted(Path(fake_dir).glob("*.mp4"))
        if not vids:
            print(f"[Warning] No fake videos found in {fake_dir}")
        fake_videos.extend(vids)

    # Reproduce official split (by video ID ordering)
    def split_list(lst, sizes):
        splits = {}
        idx = 0
        for name, n in sizes.items():
            splits[name] = lst[idx:idx+n]
            idx += n
        return splits

    real_splits = split_list(real_videos,
                              {"train": 720, "val": 140, "test": 140})
    fake_splits = split_list(fake_videos,
                              {"train": 720*4, "val": 140*4, "test": 140*4})

    result = {}
    for split in ("train", "val", "test"):
        result[split] = (
            [(str(v), 0) for v in real_splits[split]] +
            [(str(v), 1) for v in fake_splits[split]]
        )
    return result


def collect_dfdc_videos(raw_root: str) -> dict:
    """
    Collect DFDC video paths and labels from metadata JSON files.

    Expected DFDC directory structure:
        raw_root/
        ├── train/
        │   ├── dfdc_train_part_00/
        │   │   ├── metadata.json
        │   │   └── *.mp4
        │   └── ...
        └── test/
            └── *.mp4  (labels from metadata.json if available)

    Returns:
        dict with keys 'train', 'val', 'test'
    """
    import json
    import random

    all_samples = []
    train_root  = os.path.join(raw_root, "train")

    for part_dir in sorted(Path(train_root).iterdir()):
        if not part_dir.is_dir():
            continue
        meta_path = part_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            metadata = json.load(f)
        for fname, info in metadata.items():
            vpath = str(part_dir / fname)
            if not os.path.exists(vpath):
                continue
            label = 1 if info.get("label", "REAL") == "FAKE" else 0
            all_samples.append((vpath, label))

    random.seed(42)
    random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(0.80 * n)
    n_val   = int(0.10 * n)

    return {
        "train": all_samples[:n_train],
        "val"  : all_samples[n_train:n_train+n_val],
        "test" : all_samples[n_train+n_val:],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TSF-Net Data Preprocessing")
    parser.add_argument("--dataset",       choices=["ffpp", "dfdc"], default="ffpp")
    parser.add_argument("--raw_root",      type=str, default="data/raw/FaceForensics++")
    parser.add_argument("--out_root",      type=str, default="data/processed/ffpp")
    parser.add_argument("--n_frames",      type=int, default=32)
    parser.add_argument("--image_size",    type=int, default=224)
    parser.add_argument("--ffpp_subset",   type=str, default="c23",
                        help="FF++ compression: c23 (HQ) or c40")
    parser.add_argument("--ssim_labels",   action="store_true",
                        help="Generate SSIM complexity labels for ACS")
    parser.add_argument("--processed_root",type=str, default="data/processed")
    parser.add_argument("--output_json",   type=str,
                        default="data/complexity_labels/labels.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── SSIM label generation ──────────────────────────────────────────────
    if args.ssim_labels:
        print("Generating SSIM complexity labels...")
        generate_complexity_labels(
            processed_dir=args.processed_root,
            output_json=args.output_json,
        )
        return

    # ── Video collection ───────────────────────────────────────────────────
    if args.dataset == "ffpp":
        print(f"Collecting FF++ ({args.ffpp_subset}) videos from {args.raw_root}")
        split_samples = collect_ffpp_videos(args.raw_root, args.ffpp_subset)
    else:
        print(f"Collecting DFDC videos from {args.raw_root}")
        split_samples = collect_dfdc_videos(args.raw_root)

    # ── Preprocessing ──────────────────────────────────────────────────────
    for split, samples in split_samples.items():
        real_paths = [p for p, l in samples if l == 0]
        fake_paths = [p for p, l in samples if l == 1]

        real_out = os.path.join(args.out_root, split, "real")
        fake_out = os.path.join(args.out_root, split, "fake")

        print(f"\n[{split.upper()}] {len(real_paths)} real, {len(fake_paths)} fake")
        preprocess_dataset(real_paths, real_out, args.n_frames,
                            args.image_size, device)
        preprocess_dataset(fake_paths, fake_out, args.n_frames,
                            args.image_size, device)

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
