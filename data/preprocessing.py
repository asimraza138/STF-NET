"""
Video preprocessing pipeline — Section 4.1 of the manuscript.

Steps for each video:
  1. Decode frames with OpenCV.
  2. Sample T frames at uniform temporal stride.
  3. Detect and align the primary face in each frame using MTCNN
     (facenet-pytorch) with similarity-transform normalisation.
  4. Crop to 224×224 pixels.
  5. Save as a numpy array (.npy) or individual JPEG files.

Usage
-----
Run as a standalone script (see scripts/preprocess_data.py) or import
extract_faces_from_video() directly.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

import torch
from facenet_pytorch import MTCNN
from PIL import Image


# ---------------------------------------------------------------------------
# MTCNN initialisation
# ---------------------------------------------------------------------------

def build_mtcnn(device: Optional[torch.device] = None,
                image_size: int = 224,
                margin: int = 32) -> MTCNN:
    """
    Build an MTCNN face detector.

    Args:
        device    : torch.device for MTCNN inference (GPU if available)
        image_size: output face crop size (224 per manuscript)
        margin    : context margin around the detected bounding box
    Returns:
        mtcnn: MTCNN instance ready for inference
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return MTCNN(
        image_size=image_size,
        margin=margin,
        keep_all=False,       # largest face only
        post_process=True,    # normalise to [-1, 1] internally; we renormalise
        device=device,
        select_largest=True,
    )


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def sample_frames_uniform(video_path: str,
                           n_frames: int = 32) -> list[np.ndarray]:
    """
    Decode a video and uniformly sample n_frames frames.

    Args:
        video_path: path to the video file
        n_frames  : number of frames to sample (16 training, 32 inference)
    Returns:
        frames: list of (H, W, 3) uint8 BGR numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 1

    # Uniform indices, clamped to [0, total-1]
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    indices = np.clip(indices, 0, total - 1)

    frames, prev_idx = [], -1
    for idx in indices:
        if idx != prev_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = cap.read()
        if not ret:
            # Repeat last valid frame if read fails
            if frames:
                frames.append(frames[-1].copy())
            continue
        frames.append(frame)
        prev_idx = idx

    cap.release()

    # Cyclic padding if the video has fewer frames than requested
    while len(frames) < n_frames:
        frames.append(frames[len(frames) % max(len(frames), 1)].copy())

    return frames[:n_frames]


# ---------------------------------------------------------------------------
# Face extraction from a single frame
# ---------------------------------------------------------------------------

def extract_face(mtcnn: MTCNN,
                 frame_bgr: np.ndarray,
                 image_size: int = 224) -> Optional[np.ndarray]:
    """
    Detect and crop the primary face from a single BGR frame.

    Returns:
        face: (3, H, W) float32 tensor in [0, 1], or None if no face found.
    """
    # Convert BGR → RGB PIL for MTCNN
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    # MTCNN returns a (C, H, W) FloatTensor in [-1, 1] or None
    face_tensor = mtcnn(img)

    if face_tensor is None:
        # Fallback: centre crop
        h, w = frame_bgr.shape[:2]
        side  = min(h, w)
        y0    = (h - side) // 2
        x0    = (w - side) // 2
        crop  = frame_bgr[y0:y0+side, x0:x0+side]
        crop  = cv2.resize(crop, (image_size, image_size))
        face  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return face.transpose(2, 0, 1)   # (3, H, W)

    # Renormalise from [-1, 1] → [0, 1]
    face = (face_tensor.numpy() + 1.0) / 2.0   # (3, H, W)
    face = np.clip(face, 0.0, 1.0)
    return face.astype(np.float32)


# ---------------------------------------------------------------------------
# Full pipeline: video → face sequence
# ---------------------------------------------------------------------------

def extract_faces_from_video(video_path: str,
                              mtcnn: MTCNN,
                              n_frames: int = 32,
                              image_size: int = 224) -> np.ndarray:
    """
    Full preprocessing pipeline for a single video.

    Args:
        video_path : path to the video file
        mtcnn      : pre-built MTCNN detector
        n_frames   : number of frames to extract
        image_size : face crop size
    Returns:
        faces: (T, 3, H, W) float32 array in [0, 1]
    """
    raw_frames = sample_frames_uniform(video_path, n_frames)

    face_list = []
    for frame in raw_frames:
        face = extract_face(mtcnn, frame, image_size)
        face_list.append(face)   # each is (3, H, W) float32

    return np.stack(face_list, axis=0)   # (T, 3, H, W)


# ---------------------------------------------------------------------------
# Batch preprocessing utility
# ---------------------------------------------------------------------------

def preprocess_dataset(video_list:  list[str],
                        output_dir:  str,
                        n_frames:    int  = 32,
                        image_size:  int  = 224,
                        device:      Optional[torch.device] = None,
                        num_workers: int  = 4):
    """
    Preprocess a list of videos and save extracted face sequences.

    Each video is saved as:
        <output_dir>/<video_stem>.npy   shape (T, 3, H, W) float32

    Args:
        video_list : list of absolute video file paths
        output_dir : directory to save .npy files
        n_frames   : frames to extract per video
        image_size : face crop resolution
        device     : MTCNN device
        num_workers: not used here (single-process); increase for parallel runs
                     by calling this function across multiple processes.
    """
    from tqdm import tqdm

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mtcnn = build_mtcnn(device=device, image_size=image_size)

    failed = []
    for vpath in tqdm(video_list, desc="Preprocessing videos"):
        stem    = Path(vpath).stem
        outpath = os.path.join(output_dir, f"{stem}.npy")
        if os.path.exists(outpath):
            continue   # skip already processed
        try:
            faces = extract_faces_from_video(vpath, mtcnn, n_frames, image_size)
            np.save(outpath, faces)
        except Exception as e:
            failed.append((vpath, str(e)))

    if failed:
        print(f"\n[Warning] {len(failed)} videos failed preprocessing:")
        for vpath, err in failed[:10]:
            print(f"  {vpath}: {err}")

    return failed
