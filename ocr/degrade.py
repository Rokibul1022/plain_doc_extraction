"""CCTV-style degradation simulator for the easy / medium / extreme benchmark tiers.

Applies, in physical order: optical blur, motion blur, downscaling to a target plate width,
exposure change, sensor noise, glare and JPEG compression.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TierSpec:
    name: str
    plate_width: tuple[int, int]
    motion_len: tuple[float, float]
    motion_angle: float
    sigma: tuple[float, float]
    brightness: tuple[float, float]
    noise: tuple[float, float]
    jpeg: tuple[int, int]
    glare_prob: float


TIERS: dict[str, TierSpec] = {
    # easy: daylight, plate ~100+ px wide, light softness
    "easy": TierSpec("easy", (95, 130), (0, 0), 0.0, (0.5, 0.9), (0.9, 1.0), (2, 5), (65, 85), 0.0),
    # medium: ~70 px plate, slight motion blur, dusk, compressed stream
    "medium": TierSpec("medium", (62, 82), (2, 4), 10.0, (0.6, 0.9), (0.75, 0.95), (3, 5), (55, 75), 0.10),
    # extreme: ~50 px plate, clear motion blur, low light, heavy compression, occasional glare
    "extreme": TierSpec("extreme", (54, 66), (3, 5), 15.0, (0.8, 1.1), (0.6, 0.85), (4, 7), (40, 55), 0.30),
}


def motion_blur(img: np.ndarray, length: float, angle_deg: float) -> np.ndarray:
    if length < 1.5:
        return img
    size = int(2 * np.ceil(length / 2) + 3) | 1
    k = np.zeros((size, size), np.float32)
    c = size // 2
    a = np.deg2rad(angle_deg)
    dx, dy = np.cos(a) * length / 2, np.sin(a) * length / 2
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1,
             cv2.LINE_AA)
    return cv2.filter2D(img, -1, k / max(k.sum(), 1e-6), borderType=cv2.BORDER_REPLICATE)


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def degrade_crop(crop: np.ndarray, tier: str, rng: np.random.Generator, plate_w_in_crop: float) -> np.ndarray:
    """Degrade a padded plate crop as a camera would have seen it at the tier's distance."""
    t = TIERS[tier]
    x = crop.astype(np.float32)
    # optics + motion happen at the camera's native (small) scale, so downscale first
    target_w = rng.uniform(*t.plate_width)
    scale = min(1.0, target_w / max(plate_w_in_crop, 1.0))
    if scale < 1.0:
        x = cv2.resize(x, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sigma = rng.uniform(*t.sigma)
    if sigma > 0.1:
        x = cv2.GaussianBlur(x, (0, 0), sigma)
    length = rng.uniform(*t.motion_len)
    if length > 1.5:
        x = motion_blur(x, length, float(rng.uniform(-t.motion_angle, t.motion_angle)))
    gain = rng.uniform(*t.brightness)
    x = x * gain
    if t.glare_prob and rng.random() < t.glare_prob:
        h, w = x.shape[:2]
        glare = np.zeros((h, w), np.float32)
        cv2.circle(glare, (int(rng.uniform(.2, .8) * w), int(rng.uniform(.2, .8) * h)), max(3, h // 4), 1.0, -1)
        x = x + cv2.GaussianBlur(glare, (0, 0), max(2, h / 8))[..., None] * 170.0
    x = x + rng.normal(0, rng.uniform(*t.noise), x.shape)
    x = np.clip(x, 0, 255).astype(np.uint8)
    return jpeg(x, int(rng.integers(t.jpeg[0], t.jpeg[1] + 1)))
