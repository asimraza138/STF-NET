#!/usr/bin/env python3
"""
STF-Net Inference Script

Unified interface for single-video and folder-batch deepfake detection.

Examples
--------
# Single video:
python scripts/inference.py \
    --checkpoint checkpoints/seed_42/best.pt \
    --input path/to/video.mp4

# Folder (all videos):
python scripts/inference.py \
    --checkpoint checkpoints/seed_42/best.pt \
    --input path/to/folder/ \
    --output results/predictions.json

# Disable ACS (always use full model):
python scripts/inference.py \
    --checkpoint checkpoints/seed_42/best.pt \
    --input path/to/video.mp4 \
    --no_acs

# Generate Grad-CAM saliency for a specific frame:
python scripts/inference.py \
    --checkpoint checkpoints/seed_42/best.pt \
    --input path/to/video.mp4 \
    --gradcam --gradcam_frame 5 \
    --gradcam_out saliency.png
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="TSF-Net Inference")
    parser.add_argument("--checkpoint",    type=str, required=True,
                        help="Path to .pt model checkpoint")
    parser.add_argument("--input",         type=str, required=True,
                        help="Video file path or folder containing videos")
    parser.add_argument("--config",        type=str,
                        default="config/default.yaml")
    parser.add_argument("--output",        type=str, default=None,
                        help="Optional JSON file to save results")
    parser.add_argument("--no_acs",        action="store_true",
                        help="Disable ACS; always run full model")
    parser.add_argument("--acs_checkpoint",type=str, default=None,
                        help="Path to complexity estimator checkpoint")
    parser.add_argument("--threshold",     type=float, default=0.5,
                        help="Decision threshold (default: 0.5)")
    parser.add_argument("--n_frames",      type=int, default=32,
                        help="Frames to sample per video (default: 32)")
    parser.add_argument("--extensions",    type=str,
                        default=".mp4,.avi,.mov,.mkv,.webm",
                        help="Comma-separated video extensions")
    parser.add_argument("--recursive",     action="store_true",
                        help="Search for videos recursively in folder")
    parser.add_argument("--gradcam",       action="store_true",
                        help="Generate Grad-CAM saliency (single video only)")
    parser.add_argument("--gradcam_frame", type=int, default=0,
                        help="Frame index for Grad-CAM visualisation")
    parser.add_argument("--gradcam_out",   type=str, default="saliency.png",
                        help="Output path for saliency map image")
    return parser.parse_args()


def save_saliency_image(saliency, original_frame, output_path: str):
    """Overlay saliency heatmap on original frame and save."""
    import cv2
    import numpy as np

    heatmap = cv2.applyColorMap(
        (saliency * 255).astype("uint8"), cv2.COLORMAP_JET
    )
    if original_frame is not None:
        frame_bgr = cv2.cvtColor(
            (original_frame.transpose(1, 2, 0) * 255).astype("uint8"),
            cv2.COLOR_RGB2BGR,
        )
        overlay = cv2.addWeighted(frame_bgr, 0.6, heatmap, 0.4, 0)
    else:
        overlay = heatmap

    cv2.imwrite(output_path, overlay)
    print(f"[Grad-CAM] Saliency map saved to {output_path}")


def main():
    args = parse_args()

    from inference.predictor import TSFNetPredictor

    predictor = TSFNetPredictor(
        checkpoint=args.checkpoint,
        cfg_path=args.config,
        use_acs=(not args.no_acs),
        acs_checkpoint=args.acs_checkpoint,
        threshold=args.threshold,
        n_frames=args.n_frames,
    )

    input_path = Path(args.input)

    # ── Single video ───────────────────────────────────────────────────────
    if input_path.is_file():
        result = predictor.predict_video(str(input_path))

        print(f"\n{'═'*50}")
        print(f"  File    : {result['path']}")
        print(f"  Label   : {result['label']}")
        print(f"  Prob    : {result['prob']:.4f}")
        print(f"  Route   : {result['route']}")
        print(f"  Latency : {result['latency_ms']} ms")
        print(f"{'═'*50}\n")

        # ── Optional Grad-CAM ──────────────────────────────────────────
        if args.gradcam:
            from data.preprocessing import build_mtcnn, extract_faces_from_video
            import torch

            saliency = predictor.gradcam_saliency(
                str(input_path), frame_idx=args.gradcam_frame
            )
            mtcnn = build_mtcnn()
            faces = extract_faces_from_video(str(input_path), mtcnn, args.n_frames)
            frame = faces[args.gradcam_frame] if args.gradcam_frame < len(faces) else None
            save_saliency_image(saliency, frame, args.gradcam_out)

        if args.output:
            import json
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Result saved to {args.output}")

    # ── Folder batch ───────────────────────────────────────────────────────
    elif input_path.is_dir():
        extensions = args.extensions.split(",")
        predictor.predict_folder(
            folder=str(input_path),
            extensions=extensions,
            output_json=args.output,
            recursive=args.recursive,
        )
    else:
        print(f"[Error] Input path does not exist: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
