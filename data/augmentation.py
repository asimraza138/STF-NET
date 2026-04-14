"""
Augmentation pipeline — Section 4.3 of the manuscript.

Applied per-clip during training only.  The same transform is applied to
every frame in a clip (temporal consistency of augmentation).

Augmentations:
  - Random horizontal flip     (p = 0.5)
  - Random rotation            (±10°)
  - Color jitter               (brightness 0.2, contrast 0.2, saturation 0.1)
  - Random Gaussian blur       (kernel 3–7, p = 0.3)
  - Random JPEG compression    (quality 60–95, p = 0.5)
"""

import random
import io
import numpy as np
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Individual transforms (operate on (H, W, 3) uint8 PIL Images)
# ---------------------------------------------------------------------------

def random_horizontal_flip(imgs: list[Image.Image], p: float = 0.5) -> list[Image.Image]:
    if random.random() < p:
        imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
    return imgs


def random_rotation(imgs: list[Image.Image], max_degrees: float = 10.0) -> list[Image.Image]:
    angle = random.uniform(-max_degrees, max_degrees)
    return [img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0)) for img in imgs]


def color_jitter(imgs:           list[Image.Image],
                 brightness:     float = 0.2,
                 contrast:       float = 0.2,
                 saturation:     float = 0.1) -> list[Image.Image]:
    """Apply the same random colour jitter to all frames in the clip."""
    from PIL import ImageEnhance

    # Sample enhancement factors once per clip
    b_factor = random.uniform(1 - brightness, 1 + brightness)
    c_factor = random.uniform(1 - contrast,   1 + contrast)
    s_factor = random.uniform(1 - saturation, 1 + saturation)

    result = []
    for img in imgs:
        img = ImageEnhance.Brightness(img).enhance(b_factor)
        img = ImageEnhance.Contrast(img).enhance(c_factor)
        img = ImageEnhance.Color(img).enhance(s_factor)
        result.append(img)
    return result


def random_gaussian_blur(imgs:          list[Image.Image],
                         kernel_range:  tuple[int, int] = (3, 7),
                         p:             float = 0.3) -> list[Image.Image]:
    if random.random() < p:
        radius = random.uniform(kernel_range[0] / 6, kernel_range[1] / 6)
        imgs   = [img.filter(ImageFilter.GaussianBlur(radius=radius)) for img in imgs]
    return imgs


def random_jpeg_compress(imgs:          list[Image.Image],
                          quality_range: tuple[int, int] = (60, 95),
                          p:             float = 0.5) -> list[Image.Image]:
    """Simulate JPEG compression artefacts."""
    if random.random() < p:
        quality = random.randint(*quality_range)
        result  = []
        for img in imgs:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            result.append(Image.open(buf).copy())
        return result
    return imgs


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_augmentation_pipeline(cfg: dict) -> dict:
    """
    Build augmentation configuration from the cfg['augmentation'] section.
    Returns a plain dict that apply_augmentation() will consume.
    """
    return {
        "h_flip_p"    : cfg.get("horizontal_flip_p", 0.5),
        "rot_deg"     : cfg.get("rotation_degrees",  10),
        "jitter"      : cfg.get("color_jitter",
                                 {"brightness": 0.2, "contrast": 0.2, "saturation": 0.1}),
        "blur_p"      : cfg.get("gaussian_blur_p",    0.3),
        "blur_kernel" : cfg.get("gaussian_blur_kernel", [3, 7]),
        "jpeg_p"      : cfg.get("jpeg_compress_p",    0.5),
        "jpeg_quality": cfg.get("jpeg_quality",       [60, 95]),
    }


def apply_augmentation(faces: np.ndarray, aug_cfg: dict) -> np.ndarray:
    """
    Apply augmentation to a clip.

    Args:
        faces  : (T, 3, H, W) float32 in [0, 1]
        aug_cfg: dict from build_augmentation_pipeline()
    Returns:
        faces  : (T, 3, H, W) float32 in [0, 1]  (augmented)
    """
    T = faces.shape[0]

    # Convert (T, 3, H, W) float32 → list of PIL RGB images
    imgs = []
    for t in range(T):
        arr = (faces[t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        imgs.append(Image.fromarray(arr, mode="RGB"))

    # Apply transforms (same random parameters for all frames in the clip)
    imgs = random_horizontal_flip(imgs, p=aug_cfg["h_flip_p"])
    imgs = random_rotation(imgs, max_degrees=aug_cfg["rot_deg"])
    jit  = aug_cfg["jitter"]
    imgs = color_jitter(imgs,
                        brightness=jit.get("brightness", 0.2),
                        contrast=jit.get("contrast",   0.2),
                        saturation=jit.get("saturation", 0.1))
    imgs = random_gaussian_blur(imgs,
                                kernel_range=tuple(aug_cfg["blur_kernel"]),
                                p=aug_cfg["blur_p"])
    imgs = random_jpeg_compress(imgs,
                                quality_range=tuple(aug_cfg["jpeg_quality"]),
                                p=aug_cfg["jpeg_p"])

    # Convert back to (T, 3, H, W) float32
    result = []
    for img in imgs:
        arr = np.array(img, dtype=np.float32) / 255.0
        result.append(arr.transpose(2, 0, 1))

    return np.stack(result, axis=0)
