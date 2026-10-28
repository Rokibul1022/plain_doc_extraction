"""Plate restoration for poor CCTV footage: low-light, glare, deblock, denoise, deblur, upscale.

`restore_plate` inspects the measured quality of a rectified plate crop and applies only the
steps that crop actually needs, in an order that never amplifies damage it has not yet cleaned
(brighten first so noise and compression blocks are measured at their true level, then clean).
Deblurring lives in `ocr.deblur` and is OCR-guided, because blind kernel estimation on
tiny, noisy plates is unreliable; `scan.read_plate_crop` runs it before these cosmetic steps.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from opencv.quality import (
    PlateQuality,
    assess_quality,
    blockiness,
    estimate_noise,
    normalise_height,
    sharpness,
    to_gray,
)

TARGET_HEIGHT = 96
MAX_SCALE = 4.0


@dataclass
class RestoreResult:
    image: np.ndarray
    steps: list[str] = field(default_factory=list)
    scale: float = 1.0
    quality_before: PlateQuality | None = None
    quality_after: PlateQuality | None = None


# ---------------------------------------------------------------- single steps
def deblock(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Soften JPEG block edges and ringing without smearing character strokes."""
    out = cv2.bilateralFilter(img, 5, 28 * strength, 3)
    if strength > 1.2:
        out = cv2.bilateralFilter(out, 5, 40, 3)
    return cv2.addWeighted(img, 0.30, out, 0.70, 0)


def denoise(img: np.ndarray, sigma: float) -> np.ndarray:
    h = float(np.clip(0.9 * sigma, 2.0, 16.0))
    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, None, h, 5, 15)
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 5, 15)


def enhance_low_light(img: np.ndarray) -> np.ndarray:
    """Lift dark footage: gamma on luminance + local-contrast (CLAHE) + a retinex-style blend."""
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) if img.ndim == 3 else None
    y = ycc[..., 0] if ycc is not None else img
    mean = max(float(y.mean()), 1.0)
    gamma = float(np.clip(np.log(0.5) / np.log(min(mean, 254.0) / 255.0), 0.35, 1.0))
    lut = (255.0 * (np.arange(256) / 255.0) ** gamma).astype(np.uint8)
    y = cv2.LUT(y, lut)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(2, 4))
    y = clahe.apply(y)
    # single-scale retinex: remove the slowly varying illumination, keep reflectance
    f = y.astype(np.float32) + 1.0
    sigma = max(6.0, 0.35 * min(y.shape[:2]))
    bg = cv2.GaussianBlur(f, (0, 0), sigma)
    ssr = np.log(f) - np.log(bg)
    ssr = (ssr - ssr.min()) / max(ssr.max() - ssr.min(), 1e-3) * 255.0
    y = cv2.addWeighted(y, 0.6, ssr.astype(np.uint8), 0.4, 0)
    if ycc is None:
        return y
    ycc[..., 0] = y
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def suppress_glare(img: np.ndarray) -> np.ndarray:
    """Fill small specular hot-spots from their surroundings; broad bloom is left to the
    local-contrast normalisation in the readable stage."""
    gray = to_gray(img)
    hot = (gray >= 250).astype(np.uint8)
    if hot.sum() == 0:
        return img
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
    small = np.zeros_like(hot)
    limit = 0.04 * gray.size
    for i in range(1, n):
        if 3 <= stats[i, cv2.CC_STAT_AREA] <= limit:
            small[labels == i] = 255
    if small.sum() == 0:
        return img
    small = cv2.dilate(small, np.ones((3, 3), np.uint8))
    return cv2.inpaint(img, small, 3, cv2.INPAINT_TELEA)


def upscale(img: np.ndarray, target_height: int = TARGET_HEIGHT) -> tuple[np.ndarray, float]:
    h = img.shape[0]
    if h >= target_height:
        return img, 1.0
    scale = min(MAX_SCALE, target_height / float(h))
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return out, scale


def stretch_contrast(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    gray = to_gray(img)
    lo, hi = np.percentile(gray, lo_pct), np.percentile(gray, hi_pct)
    if hi - lo < 10:
        return img
    lut = np.clip((np.arange(256) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def unsharp(img: np.ndarray, amount: float = 0.6, sigma: float = 1.0) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


# ---------------------------------------------------------------- deblurring helpers
def _motion_psf(length: float, angle_deg: float) -> np.ndarray:
    size = int(2 * np.ceil(length / 2) + 3) | 1
    k = np.zeros((size, size), np.float32)
    c = size // 2
    a = np.deg2rad(angle_deg)
    dx, dy = np.cos(a) * length / 2.0, np.sin(a) * length / 2.0
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1,
             cv2.LINE_AA)
    s = k.sum()
    return k / s if s > 0 else np.pad(np.ones((1, 1), np.float32), size // 2)


def _disk_psf(radius: float) -> np.ndarray:
    size = int(2 * np.ceil(radius) + 3) | 1
    k = np.zeros((size, size), np.float32)
    cv2.circle(k, (size // 2, size // 2), int(round(radius)), 1.0, -1, cv2.LINE_AA)
    return k / max(k.sum(), 1e-6)


def _gauss_psf(sigma: float) -> np.ndarray:
    size = int(2 * np.ceil(3 * sigma) + 1) | 1
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2)).astype(np.float32)
    return g / g.sum()


def _wiener(gray: np.ndarray, psf: np.ndarray, k_reg: float) -> np.ndarray:
    """Frequency-domain Wiener deconvolution with edge padding to limit boundary ringing."""
    pad = psf.shape[0]
    g = cv2.copyMakeBorder(gray.astype(np.float32), pad, pad, pad, pad, cv2.BORDER_REFLECT)
    h, w = g.shape
    kernel = np.zeros((h, w), np.float32)
    kernel[: psf.shape[0], : psf.shape[1]] = psf
    kernel = np.roll(kernel, (-(psf.shape[0] // 2), -(psf.shape[1] // 2)), axis=(0, 1))
    hf = np.fft.rfft2(kernel)
    gf = np.fft.rfft2(g)
    ff = np.conj(hf) / (np.abs(hf) ** 2 + k_reg) * gf
    out = np.fft.irfft2(ff, s=(h, w))
    return np.clip(out[pad:-pad, pad:-pad], 0, 255)


# ---------------------------------------------------------------- orchestration
def restore_plate(
    img: np.ndarray,
    quality: PlateQuality | None = None,
    *,
    plate_width: float | None = None,
    target_height: int = TARGET_HEIGHT,
) -> RestoreResult:
    """Apply only the cosmetic restoration steps a plate crop needs, in a fixed, safe order.

    Deblurring is separate (`ocr.deblur`) because it needs the plate reader."""
    quality = quality or assess_quality(img, plate_width)
    work = img.copy()
    steps: list[str] = []
    issues = set(quality.issues)

    # 1. lift darkness / glare first, so the damage that lifting exposes is measured below
    if "dark" in issues:
        work = enhance_low_light(work)
        steps.append("low-light")
    if "glare" in issues:
        work = suppress_glare(work)
        steps.append("glare")

    # 2. clean compression blocks and sensor noise at their post-lift level
    norm = normalise_height(to_gray(work))
    block, noise = blockiness(norm), estimate_noise(norm)
    if block > 1.18 or "compressed" in issues:
        work = deblock(work, 1.5 if (block > 1.4 or "dark" in issues) else 1.0)
        steps.append(f"deblock(x{block:.2f})")
    if noise > 4.5:
        work = denoise(work, noise)
        steps.append(f"denoise(sigma={noise:.1f})")

    # 3. enlarge, then final contrast and a light sharpen
    work, scale = upscale(work, target_height)
    if scale > 1.0:
        steps.append(f"upscale x{scale:.1f}")
    work = stretch_contrast(work)
    steps.append("contrast")
    if "blurry" in issues or "low_resolution" in issues:
        work = unsharp(work, 0.5, 1.0)
        steps.append("sharpen")

    after = assess_quality(work, plate_width if plate_width is not None else img.shape[1])
    return RestoreResult(image=work, steps=steps, scale=scale, quality_before=quality, quality_after=after)
