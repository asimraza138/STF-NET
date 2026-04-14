# Spatio-Temporal Fusion Learning for Robust Deepfake Video Forensics

Official implementation of the TSF-Net framework.

---

## Overview

TSF-Net is a deepfake video detection framework that integrates spatial and temporal
analysis through three core components:

- **CMAF** (Cross-Modal Attention Fusion): Fuses EfficientNetV2L global texture features
  with modified XceptionNet local artifact features using multi-head cross-attention.
- **TIAM** (Temporal Inconsistency Attention Module): Highlights frames with significant
  inter-frame discontinuities using a frame-difference-guided Bi-LSTM attention mechanism.
- **AALF** (Artifact-Aware Loss Function): Aligns classifier predictions with learned
  artifact cues through a regularisation term with stop-gradient and diversity safeguards.
- **ACS** (Adaptive Computational Scaling): Routes videos to lightweight, standard, or
  full processing paths based on estimated visual complexity.

**Reported results:** 95.36% accuracy, 0.92 F1-score on DFDC + FaceForensics++ (c23, HQ).

---

## Repository Structure

```
tsfnet/
├── config/
│   └── default.yaml          All hyperparameters (matches manuscript exactly)
├── models/
│   ├── backbones.py          EfficientNetV2L + modified XceptionNet
│   ├── cmaf.py               Cross-Modal Attention Fusion
│   ├── tiam.py               Temporal Inconsistency Attention Module
│   ├── aalf.py               Artifact-Aware Loss Function
│   ├── acs.py                Adaptive Computational Scaling
│   └── tsfnet.py             Complete TSF-Net model
├── data/
│   ├── preprocessing.py      MTCNN face extraction + frame sampling
│   ├── dataset.py            FF++ and DFDC dataset classes
│   └── augmentation.py       Training augmentation pipeline
├── training/
│   ├── trainer.py            Multi-GPU DDP trainer (10-seed protocol)
│   └── complexity_trainer.py ACS complexity estimator training
├── inference/
│   └── predictor.py          Single-video and folder-batch inference
├── utils/
│   ├── metrics.py            Accuracy, Precision, Recall, F1, AUC
│   ├── ssim_labels.py        SSIM-based ACS complexity label generation
│   └── logging_utils.py      Logging helpers
├── scripts/
│   ├── preprocess_data.py    Data preprocessing entry point
│   ├── train.py              Training entry point
│   └── inference.py          Inference entry point
├── requirements.txt
└── setup.py
```

---

## Environment

**Tested configuration:**

| Component    | Version          |
|--------------|------------------|
| Python       | 3.10 or 3.11     |
| PyTorch      | ≥ 2.1.0          |
| CUDA         | 11.8 or 12.1     |
| cuDNN        | 8.x              |
| GPU          | 4× NVIDIA A100 40GB (training) |

> **Manuscript note:** The original manuscript cited PyTorch 1.10 / CUDA 11.3.
> The codebase targets PyTorch ≥ 2.1 for full compatibility with modern CUDA
> drivers, `torch.compile`, and `weights_only` checkpoint loading.
> All architecture and hyperparameter specifications are unchanged.

**Installation:**

```bash
git clone https://github.com/<your-username>/tsfnet.git
cd tsfnet
pip install -r requirements.txt
# Or install as a package:
pip install -e .
```

---

## Dataset Setup

### FaceForensics++ (FF++)

1. Request access at: https://github.com/ondyari/FaceForensics
2. Fill out the usage agreement form.
3. Download the **c23 (HQ)** subset using the provided download script:
   ```bash
   python FaceForensics/dataset/download_FaceForensics.py \
       data/raw/FaceForensics++ -d all -c c23 -t videos
   ```
4. The expected directory layout after download:
   ```
   data/raw/FaceForensics++/
   ├── original_sequences/youtube/c23/videos/   *.mp4
   └── manipulated_sequences/
       ├── Deepfakes/c23/videos/
       ├── Face2Face/c23/videos/
       ├── FaceSwap/c23/videos/
       └── NeuralTextures/c23/videos/
   ```

### DFDC (Deepfake Detection Challenge)

1. Request access at: https://ai.meta.com/datasets/dfdc/
2. Download via the AWS CLI (credentials provided after approval):
   ```bash
   aws s3 sync s3://dfdc-dataset/ data/raw/DFDC/
   ```
3. The expected directory layout:
   ```
   data/raw/DFDC/train/
   ├── dfdc_train_part_00/
   │   ├── metadata.json
   │   └── *.mp4
   ├── dfdc_train_part_01/
   └── ...
   ```

---

## Preprocessing

Face extraction must be run before training. This applies MTCNN detection,
similarity-transform alignment, and saves each video as a (T, 3, 224, 224) `.npy` file.

**Step 1 — Extract faces from FF++:**
```bash
python scripts/preprocess_data.py \
    --dataset ffpp \
    --raw_root data/raw/FaceForensics++ \
    --out_root data/processed/ffpp \
    --n_frames 32 \
    --ffpp_subset c23
```

**Step 2 — Extract faces from DFDC:**
```bash
python scripts/preprocess_data.py \
    --dataset dfdc \
    --raw_root data/raw/DFDC \
    --out_root data/processed/dfdc \
    --n_frames 32
```

**Step 3 — Generate ACS complexity labels:**
```bash
python scripts/preprocess_data.py \
    --ssim_labels \
    --processed_root data/processed \
    --output_json data/complexity_labels/labels.json
```

This generates SSIM-based proxy complexity scores for training the ACS estimator.

---

## Training

Training runs in two stages:

### Stage 1: ACS Complexity Estimator

The complexity estimator (SqueezeNet 1.1 with regression head) is trained first on
SSIM-based proxy labels. This takes approximately 20–30 minutes on a single GPU.

```bash
# Runs automatically as part of the full training pipeline.
# To run standalone:
python -c "
import yaml
from training.complexity_trainer import train_complexity_estimator
with open('config/default.yaml') as f:
    cfg = yaml.safe_load(f)
train_complexity_estimator(cfg)
"
```

### Stage 2: TSF-Net (10 seeds, 4× A100)

**Distributed training (recommended):**
```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --config config/default.yaml \
    --distributed
```

**Single-GPU training:**
```bash
python scripts/train.py --config config/default.yaml
```

**Skip ACS estimator training (if checkpoint already exists):**
```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --config config/default.yaml \
    --distributed \
    --skip_stage1
```

**Run specific seeds only:**
```bash
python scripts/train.py --config config/default.yaml --seeds_only 0,1,2
```

Training produces:
```
checkpoints/
├── complexity_estimator/best.pt   ACS estimator checkpoint
└── seed_42/best.pt                Best TSF-Net checkpoint (seed 0)
    seed_43/best.pt                ...
    ...
logs/
├── aggregated_results.json        Mean ± std over 10 seeds
└── test_seed42.json               Per-seed test metrics
```

---

## Inference

### Single video

```bash
python scripts/inference.py \
    --checkpoint checkpoints/seed_42/best.pt \
    --input path/to/video.mp4
```

Output:
```
══════════════════════════════════════════════════
  File    : path/to/video.mp4
  Label   : FAKE
  Prob    : 0.9312
  Route   : standard
  Latency : 84.3 ms
══════════════════════════════════════════════════
```

### Folder (batch)

```bash
python scripts/inference.py \
    --checkpoint checkpoints/seed_42/best.pt \
    --input path/to/videos/ \
    --output results/predictions.json \
    --recursive
```

### Disable ACS (always use full model)

```bash
python scripts/inference.py \
    --checkpoint checkpoints/seed_42/best.pt \
    --input path/to/video.mp4 \
    --no_acs
```

### Grad-CAM saliency visualisation

```bash
python scripts/inference.py \
    --checkpoint checkpoints/seed_42/best.pt \
    --input path/to/video.mp4 \
    --gradcam \
    --gradcam_frame 10 \
    --gradcam_out saliency_frame10.png
```

### Python API

```python
from inference.predictor import TSFNetPredictor

predictor = TSFNetPredictor(
    checkpoint="checkpoints/seed_42/best.pt",
    use_acs=True,
    acs_checkpoint="checkpoints/complexity_estimator/best.pt",
)

# Single video
result = predictor.predict_video("video.mp4")
print(result)
# {'path': 'video.mp4', 'prob': 0.93, 'label': 'FAKE',
#  'route': 'standard', 'latency_ms': 84.3}

# Folder
results = predictor.predict_folder("videos/", output_json="out.json")
```

---

## Key Hyperparameters

All values match those reported in the manuscript. Edit `config/default.yaml` to change.

| Parameter              | Value  | Description                               |
|------------------------|--------|-------------------------------------------|
| `embed_dim`            | 512    | CMAF shared embedding dimension D         |
| `cmaf_heads`           | 8      | Cross-attention heads                     |
| `spatial_resolution`   | 14     | Spatial token grid H̄ = W̄                |
| `bilstm_hidden`        | 256    | Bi-LSTM hidden units per direction        |
| `bilstm_layers`        | 2      | Stacked Bi-LSTM layers                    |
| `n_artifact_detectors` | 4      | ArtifactDetectors in AALF                 |
| `epochs`               | 50     | Training epochs                           |
| `lr`                   | 2e-4   | Initial learning rate (AdamW)             |
| `batch_size_per_gpu`   | 16     | Per-GPU batch size (64 effective)         |
| `freeze_backbone_epochs`| 10    | Epochs with frozen backbones              |
| `lambda_aalf`          | 0.1    | AALF regularisation weight λ              |
| `w_fake`               | 2.0    | AALF fake-class weight                    |
| `n_seeds`              | 10     | Number of repeated training runs          |
| `tau_low`              | 0.3    | ACS lightweight-path threshold            |
| `tau_high`             | 0.7    | ACS full-model threshold                  |
| `confidence_gate`      | 0.55   | ACS confidence gate threshold             |

---

## Architecture Details

### Modified XceptionNet

Per the manuscript (Section 3.1.2), three modifications are applied to the standard
Xception architecture:
1. Middle-flow repetitions: 8 → **12** (deeper artifact feature extraction)
2. Extra residual connection inside each middle-flow block (preserves fine-grained details)
3. Final pooling replaced with `AdaptiveAvgPool2d` to retain spatial feature maps

### CMAF

Feature maps from both backbones are:
1. Spatially aligned to **14×14** via `AdaptiveAvgPool2d`
2. Projected to **D = 512** with independent 1×1 convolutions
3. Flattened to **N = 196** tokens and augmented with 2-D sinusoidal positional encodings
4. Processed with multi-head cross-attention (Q from EfficientNetV2L, K/V from XceptionNet)
5. Mean-pooled to a single **512-dim** frame descriptor

### TIAM

- **Δt** is computed as the L2-norm between consecutive frame descriptors (scalar)
- Normalised per-video with min-max normalisation
- Energy **et = Wα · [Ht; Δ̂t] + bα** is a scalar produced by a learned projection
- **αt = softmax(e₁,...,eT)** runs over the full temporal dimension
- Output: **Σt αt · Ht** (weighted sum of Bi-LSTM hidden states)
- Variable-length clips handled via zero-padding + attention mask

### ACS Complexity Estimator

- Architecture: **SqueezeNet 1.1** (fire modules; ≈1.2M parameters)
- Trained independently on SSIM-drop proxy labels (frozen during TSF-Net training)
- Operates on **first frame only** (≈1.5 ms per video on A100)
- Fire modules are architecturally distinct from all other components in TSF-Net

---

## Citation

If you use this code, please cite:

```bibtex
@article{manzoor2026tsfnet,
  title   = {Spatio-Temporal Fusion Learning for Robust Deepfake Video Forensics},
  author  = {Manzoor, Asim and Rauf, Muhammad Arslan and Ullah, Ubaid and
             Elahi, Ihsan and Iqbal, Asif},
  journal = {The Visual Computer},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This repository is released for academic and research use.
Please refer to the dataset providers (FF++ and DFDC) for their respective
terms of use before training or evaluating models on their data.
