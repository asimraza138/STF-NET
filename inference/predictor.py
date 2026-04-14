"""
STF-Net Inference Module — unified interface for single-video and folder-batch
prediction (Section 3.5 and Q7 specification).

Routing:
  - If ACS is enabled (default), the complexity estimator selects the processing
    path automatically.
  - If ACS is disabled (--no-acs flag), all videos run through the full TSF-Net.

Usage examples
--------------
Single video:
    predictor = TSFNetPredictor(checkpoint="checkpoints/seed_42/best.pt")
    result    = predictor.predict_video("path/to/video.mp4")
    print(result)   # {'prob': 0.93, 'label': 'FAKE', 'path': 'full', ...}

Folder (batch):
    results = predictor.predict_folder("path/to/videos/", extensions=[".mp4", ".avi"])
"""

import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from models.tsfnet import TSFNet, build_tsfnet
from models.acs    import ACSController
from data.preprocessing import build_mtcnn, extract_faces_from_video


# ---------------------------------------------------------------------------
# Normalisation constants (ImageNet)
# ---------------------------------------------------------------------------

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def _normalise(frames: torch.Tensor) -> torch.Tensor:
    """
    Normalise a (B, T, 3, H, W) float32 tensor in [0, 1] to ImageNet stats.
    """
    mean = _MEAN.to(frames.device)
    std  = _STD.to(frames.device)
    return (frames - mean) / std


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class TSFNetPredictor:
    """
    High-level inference interface for TSF-Net.

    Args:
        checkpoint      : path to a .pt checkpoint saved by the trainer
        cfg_path        : path to config YAML (default: 'config/default.yaml')
        device          : torch.device; auto-detected if None
        use_acs         : route videos through ACS paths (True) or always use
                          full model (False)
        acs_checkpoint  : path to the complexity estimator checkpoint;
                          required when use_acs=True
        threshold       : decision threshold for binary label (default 0.5)
        n_frames        : number of frames to sample per video (32 for inference)
    """

    def __init__(self,
                 checkpoint:     str,
                 cfg_path:       str  = "config/default.yaml",
                 device:         Optional[torch.device] = None,
                 use_acs:        bool = True,
                 acs_checkpoint: Optional[str] = None,
                 threshold:      float = 0.5,
                 n_frames:       int   = 32):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device    = device
        self.threshold = threshold
        self.n_frames  = n_frames

        # ── Load configuration ────────────────────────────────────────────
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg

        # ── Build and load full model ─────────────────────────────────────
        self.model = build_tsfnet(cfg["model"], pretrained_bb=False).to(device)
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        state = ckpt.get("model", ckpt)   # handle both raw and wrapped checkpoints
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        # ── ACS controller ────────────────────────────────────────────────
        self.use_acs = use_acs
        self.acs     = None
        if use_acs:
            acs_cfg = cfg["acs"]
            self.acs = ACSController(
                tau_low=acs_cfg["tau_low"],
                tau_high=acs_cfg["tau_high"],
                confidence_gate=acs_cfg["confidence_gate"],
                pretrained=False,
            ).to(device)
            if acs_checkpoint and os.path.exists(acs_checkpoint):
                est_state = torch.load(acs_checkpoint, map_location=device,
                                       weights_only=True)
                self.acs.estimator.load_state_dict(est_state)
                print(f"[ACS] Loaded complexity estimator from {acs_checkpoint}")
            else:
                print("[ACS] Warning: no complexity estimator checkpoint provided; "
                      "using ImageNet-pretrained SqueezeNet features.")
            self.acs.eval()

        # ── MTCNN face detector ───────────────────────────────────────────
        self.mtcnn = build_mtcnn(device=device, image_size=224)

        print(f"[TSFNet] Model loaded from {checkpoint}")
        print(f"[TSFNet] Running on {device} | ACS: {use_acs}")

    # -----------------------------------------------------------------------
    # Core prediction (single video)
    # -----------------------------------------------------------------------

    def predict_video(self, video_path: str) -> dict:
        """
        Predict whether a single video is real or deepfake.

        Args:
            video_path: path to the video file
        Returns:
            dict with keys:
              'path'      : input video path
              'prob'      : deepfake probability ∈ [0, 1]
              'label'     : 'FAKE' or 'REAL'
              'route'     : 'full', 'standard', 'lightweight', or 'full (gate)'
              'latency_ms': end-to-end inference time in milliseconds
        """
        t0 = time.perf_counter()

        # ── Face extraction ───────────────────────────────────────────────
        faces = extract_faces_from_video(video_path, self.mtcnn, self.n_frames)
        # faces: (T, 3, H, W) float32 in [0, 1]

        frames = torch.from_numpy(faces).unsqueeze(0).to(self.device)  # (1, T, 3, H, W)
        frames = _normalise(frames)

        # ── Inference ─────────────────────────────────────────────────────
        with torch.no_grad():
            if self.use_acs and self.acs is not None:
                probs, routes = self.acs.route(frames, self.model)
                prob  = float(probs[0])
                route = routes[0]
            else:
                out   = self.model(frames)
                prob  = float(out["prob"][0])
                route = "full"

        t1 = time.perf_counter()

        return {
            "path"       : video_path,
            "prob"       : round(prob, 4),
            "label"      : "FAKE" if prob >= self.threshold else "REAL",
            "route"      : route,
            "latency_ms" : round((t1 - t0) * 1000, 1),
        }

    # -----------------------------------------------------------------------
    # Batch prediction (folder)
    # -----------------------------------------------------------------------

    def predict_folder(self,
                        folder:     str,
                        extensions: list[str] = None,
                        output_json: Optional[str] = None,
                        recursive:  bool = False) -> list[dict]:
        """
        Predict all videos in a folder.

        Args:
            folder      : directory containing video files
            extensions  : list of file extensions to process
                          (default: ['.mp4', '.avi', '.mov', '.mkv'])
            output_json : if provided, save results to this JSON file
            recursive   : if True, search subdirectories recursively
        Returns:
            results: list of prediction dicts (one per video)
        """
        if extensions is None:
            extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

        folder = Path(folder)
        if recursive:
            video_files = [
                f for f in folder.rglob("*")
                if f.suffix.lower() in extensions
            ]
        else:
            video_files = [
                f for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in extensions
            ]

        video_files = sorted(video_files)
        if not video_files:
            print(f"[Warning] No video files found in {folder}")
            return []

        print(f"[TSFNet] Processing {len(video_files)} videos from {folder}")

        results = []
        failed  = []
        for vpath in tqdm(video_files, desc="Predicting", ncols=80):
            try:
                result = self.predict_video(str(vpath))
                results.append(result)
            except Exception as e:
                failed.append({"path": str(vpath), "error": str(e)})

        if failed:
            print(f"\n[Warning] {len(failed)} videos failed:")
            for f in failed[:5]:
                print(f"  {f['path']}: {f['error']}")

        # ── Print summary ─────────────────────────────────────────────────
        n_fake = sum(1 for r in results if r["label"] == "FAKE")
        n_real = len(results) - n_fake
        avg_lat = np.mean([r["latency_ms"] for r in results]) if results else 0

        print(f"\n{'─'*50}")
        print(f"  Total processed : {len(results)}")
        print(f"  FAKE            : {n_fake}")
        print(f"  REAL            : {n_real}")
        print(f"  Avg latency     : {avg_lat:.1f} ms/video")

        if self.use_acs:
            from collections import Counter
            route_counts = Counter(r["route"] for r in results)
            print(f"  ACS routing     : {dict(route_counts)}")
        print(f"{'─'*50}\n")

        # ── Save results ──────────────────────────────────────────────────
        if output_json:
            Path(output_json).parent.mkdir(parents=True, exist_ok=True)
            with open(output_json, "w") as f:
                json.dump({"predictions": results, "failed": failed}, f, indent=2)
            print(f"[TSFNet] Results saved to {output_json}")

        return results

    # -----------------------------------------------------------------------
    # Grad-CAM saliency (qualitative interpretability — Section 4.4.4)
    # -----------------------------------------------------------------------

    def gradcam_saliency(self,
                          video_path:  str,
                          frame_idx:   int = 0,
                          target_layer: str = "cmaf") -> np.ndarray:
        """
        Compute Grad-CAM saliency map for a specified frame.

        Args:
            video_path  : path to input video
            frame_idx   : which frame (0-indexed) to visualise
            target_layer: 'cmaf' hooks into CMAF cross-attention output
        Returns:
            saliency: (H, W) float32 heatmap normalised to [0, 1]
        """
        faces  = extract_faces_from_video(video_path, self.mtcnn, self.n_frames)
        frames = torch.from_numpy(faces).unsqueeze(0).to(self.device)  # (1,T,3,H,W)
        frames = _normalise(frames)

        # Register hooks on CMAF output projection
        activations = {}
        gradients   = {}

        def fwd_hook(module, inp, out):
            activations["feat"] = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            gradients["feat"] = grad_out[0].detach()

        # Hook into CMAF's output normalisation
        handle_fwd = self.model.cmaf.out_norm.register_forward_hook(fwd_hook)
        handle_bwd = self.model.cmaf.out_norm.register_full_backward_hook(bwd_hook)

        self.model.zero_grad()
        self.model.train()   # enable grad computation

        # Single-frame forward
        single = frames[:, frame_idx:frame_idx+1, :, :, :]  # (1,1,3,H,W)
        feat_eff  = self.model.eff_backbone(single.squeeze(1))
        feat_xcep = self.model.xcep_backbone(single.squeeze(1))
        F_comb    = self.model.cmaf(feat_eff, feat_xcep)

        # Maximise the fake class score
        score = F_comb.sum()
        score.backward()

        handle_fwd.remove()
        handle_bwd.remove()
        self.model.eval()

        # Compute saliency
        act  = activations.get("feat", None)
        grad = gradients.get("feat", None)
        if act is None or grad is None:
            return np.zeros((224, 224), dtype=np.float32)

        weights  = grad.mean(dim=1, keepdim=True)   # (B, 1, N, D) or (B, N)
        saliency = (weights * act).sum(dim=-1)       # (B, N)
        saliency = saliency.relu().cpu().numpy()[0]  # (N,)

        # Reshape tokens back to spatial grid
        H = W = int(saliency.shape[0] ** 0.5)
        saliency = saliency.reshape(H, W)

        # Resize to face-crop resolution
        import cv2 as _cv2
        saliency = _cv2.resize(saliency, (224, 224))
        s_min, s_max = saliency.min(), saliency.max()
        if s_max > s_min:
            saliency = (saliency - s_min) / (s_max - s_min)

        return saliency.astype(np.float32)
