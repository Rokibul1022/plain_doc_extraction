"""Plate image quality assessment: resolution, blur, contrast, exposure, noise, compression.

All measurements run on a copy normalised to a fixed height, so scores are comparable between
a 60-px and a 400-px plate crop. The result drives which restoration steps are applied and lets
the system say honestly when a plate is unreadable.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

NORM_HEIGHT = 64


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalise_height(gray: np.ndarray, height: int = NORM_HEIGHT) -> np.ndarray:
    h, w = gray.shape[:2]
    if h == height:
        return gray
    scale = height / float(h)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(gray, (max(8, int(round(w * scale))), height), interpolation=interp)


def estimate_noise(gray: np.ndarray) -> float:
    """Noise standard deviation (Immerkaer's fast estimator), in grey levels."""
    g = gray.astype(np.float32)
    h, w = g.shape
    if h < 5 or w < 5:
        return 0.0
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    resp = cv2.filter2D(g, -1, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2.0) * np.abs(resp).sum() / (6.0 * (w - 2) * (h - 2)))


def sharpness(gray: np.ndarray) -> float:
    """Edge sharpness relative to contrast, independent of plate size and brightness.

    On real plates: reads almost never succeed below ~0.3 and mostly succeed above ~0.5."""
    g = normalise_height(gray).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 0.6)  # keep sensor noise from reading as sharpness
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0) / 16.0
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1) / 16.0
    mag = np.hypot(gx, gy)
    contrast = float(np.percentile(g, 95) - np.percentile(g, 5))
    if contrast < 8:
        return 0.0
    return float(np.percentile(mag, 98) / contrast)


def blockiness(gray: np.ndarray) -> float:
    """JPEG block-edge energy relative to ordinary edges: > 1.15 means visible 8x8 blocking."""
    g = gray.astype(np.float32)
    h, w = g.shape
    if h < 24 or w < 24:
        return 1.0
    dx = np.abs(np.diff(g, axis=1))
    cols = np.arange(dx.shape[1])
    on = dx[:, (cols % 8) == 7].mean()
    off = dx[:, (cols % 8) != 7].mean()
    return float(on / max(off, 1e-3))


@dataclass
class PlateQuality:
    score: float
    level: str
    issues: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 1),
            "level": self.level,
            "issues": list(self.issues),
            "metrics": {k: round(float(v), 3) for k, v in self.metrics.items()},
        }


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def level_for(score: float) -> str:
    if score >= 70:
        return "good"
    if score >= 45:
        return "fair"
    if score >= 25:
        return "poor"
    return "unreadable"


def assess_quality(img: np.ndarray, plate_width: float | None = None) -> PlateQuality:
    """Score a plate crop 0-100 and list what is wrong with it.

    `plate_width` is the plate's real width in source-image pixels (the crop may already be
    upscaled); defaults to the crop width."""
    gray = to_gray(img)
    h, w = gray.shape[:2]
    width_px = float(plate_width if plate_width is not None else w)
    norm = normalise_height(gray)

    sharp = sharpness(gray)
    contrast = float(np.percentile(norm, 95) - np.percentile(norm, 5))
    mean = float(norm.mean())
    dark_frac = float((norm < 12).mean())
    glare_frac = float((norm > 247).mean())
    noise = estimate_noise(norm)
    block = blockiness(norm)

    res_s = _clip01((width_px - 28.0) / (110.0 - 28.0))
    sharp_s = _clip01((sharp - 0.12) / (0.60 - 0.12))  # measured: reads fail below ~0.3, mostly succeed above ~0.5
    contrast_s = _clip01((contrast - 22.0) / (95.0 - 22.0))
    expo_s = _clip01(1.0 - max(0.0, 55.0 - mean) / 55.0 - max(0.0, mean - 205.0) / 50.0
                     - 1.5 * max(0.0, glare_frac - 0.12) - 1.5 * max(0.0, dark_frac - 0.30))
    noise_s = _clip01(1.0 - max(0.0, noise - 3.0) / 22.0)

    eps = 0.02
    score = 100.0 * ((res_s + eps) ** 0.30 * (sharp_s + eps) ** 0.30 * (contrast_s + eps) ** 0.20
                     * (expo_s + eps) ** 0.10 * (noise_s + eps) ** 0.10) / ((1 + eps) ** 1.0)

    issues: list[str] = []
    if width_px < 70:
        issues.append("low_resolution")
    if sharp < 0.40:
        issues.append("blurry")
    if contrast < 45:
        issues.append("low_contrast")
    if mean < 70 or dark_frac > 0.30:
        issues.append("dark")
    if glare_frac > 0.15:
        issues.append("glare")
    if noise > 9.0:
        issues.append("noisy")
    if block > 1.18:
        issues.append("compressed")

    metrics = {
        "width_px": width_px,
        "sharpness": sharp,
        "contrast": contrast,
        "mean": mean,
        "dark_frac": dark_frac,
        "glare_frac": glare_frac,
        "noise": noise,
        "blockiness": block,
    }
    return PlateQuality(score=float(score), level=level_for(score), issues=issues, metrics=metrics)
