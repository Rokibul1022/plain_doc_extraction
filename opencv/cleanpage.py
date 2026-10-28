"""Sharper, cleaner plain page from a rectified plate - OpenCV + NumPy only.

Pipeline on the grey, illumination-flattened plate (dark ink, light paper):
  1. estimate the blur from the width of stroke edges (no labels, no model);
  2. undo it with Wiener deconvolution (NumPy FFT) using that estimate;
  3. local (Sauvola) thresholding, so faint strokes survive and shadows do not;
  4. smooth contour rebuild: each stroke outline is re-drawn as a smooth closed curve.
Steps 1-4 return an ink mask; the caller keeps its own frame/bolt/row clean-up.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------- 1. blur estimate
def estimate_blur_sigma(flat: np.ndarray) -> float:
    """Gaussian blur sigma (pixels) estimated from the steepest edges of a flattened grey image.

    A step edge of contrast C blurred by a Gaussian of sigma s has peak gradient C / (s * sqrt(2 pi)).
    So s ~= C / (sqrt(2 pi) * peak gradient), using robust percentiles for C and the peak."""
    g = cv2.GaussianBlur(flat.astype(np.float32), (0, 0), 0.7)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mag = np.hypot(gx, gy)
    contrast = float(np.percentile(g, 97) - np.percentile(g, 3))
    peak = float(np.percentile(mag, 99.0))
    if contrast < 20 or peak < 1e-3:
        return 0.0
    s = contrast / (np.sqrt(2 * np.pi) * peak)
    # remove the small blur this very measurement added (sigma 0.7) in quadrature
    return float(np.sqrt(max(s * s - 0.7 * 0.7, 0.0)))


# ---------------------------------------------------------------- 2. Wiener deconvolution
def _gauss_psf(sigma: float) -> np.ndarray:
    k = int(2 * np.ceil(3 * sigma) + 1) | 1
    ax = np.arange(k) - k // 2
    psf = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2)).astype(np.float32)
    return psf / psf.sum()


def wiener_deblur(gray: np.ndarray, sigma: float, k_reg: float = 0.01) -> np.ndarray:
    """Frequency-domain Wiener deconvolution of a Gaussian blur; edge-reflected padding limits ringing."""
    if sigma < 0.6:
        return gray
    psf = _gauss_psf(sigma)
    pad = psf.shape[0]
    g = cv2.copyMakeBorder(gray.astype(np.float32), pad, pad, pad, pad, cv2.BORDER_REFLECT)
    h, w = g.shape
    kern = np.zeros((h, w), np.float32)
    kern[: psf.shape[0], : psf.shape[1]] = psf
    kern = np.roll(kern, (-(psf.shape[0] // 2), -(psf.shape[1] // 2)), axis=(0, 1))
    hf = np.fft.rfft2(kern)
    out = np.fft.irfft2(np.conj(hf) / (np.abs(hf) ** 2 + k_reg) * np.fft.rfft2(g), s=(h, w))[pad:-pad, pad:-pad]
    lo, hi = float(gray.min()), float(gray.max())
    return np.clip(out, lo, hi).astype(np.float32)


# ---------------------------------------------------------------- 3. Sauvola threshold
def sauvola_ink(gray: np.ndarray, window: int, k: float = 0.22, r: float = 128.0) -> np.ndarray:
    """0..1 ink strength: how far a pixel is below the Sauvola local threshold (integral-image means)."""
    w = int(window) | 1
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, cv2.CV_32F, (w, w), borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(f * f, cv2.CV_32F, (w, w), borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    thr = mean * (1.0 + k * (std / r - 1.0))
    margin = np.maximum(thr - f, 0.0) / np.maximum(0.25 * mean, 12.0)
    return np.clip(margin, 0.0, 1.0)


# ---------------------------------------------------------------- 4. smooth contour rebuild
def _smooth_contour(pts: np.ndarray, sigma: float) -> np.ndarray:
    """Circularly Gaussian-smooth a closed contour's coordinates (keeps corners softly rounded, not jagged)."""
    n = len(pts)
    if n < 8 or sigma <= 0:
        return pts
    r = int(min(max(2, round(3 * sigma)), n // 2 - 1))
    x = np.arange(-r, r + 1)
    kern = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kern /= kern.sum()
    ext = np.concatenate([pts[-r:], pts, pts[:r]]).astype(np.float32)
    out = np.stack([np.convolve(ext[:, 0], kern, mode="valid"), np.convolve(ext[:, 1], kern, mode="valid")], 1)
    return out


def rebuild_strokes(mask: np.ndarray, sigma: float = 2.0, supersample: int = 3) -> np.ndarray:
    """Redraw every stroke outline (and every hole inside it) as a smooth closed curve at a higher resolution,
    then return it at the original size. Gives print-like edges instead of pixel staircases."""
    m = (mask > 0).astype(np.uint8)
    if m.sum() < 30:
        return m
    h, w = m.shape
    big = cv2.resize(m * 255, (w * supersample, h * supersample), interpolation=cv2.INTER_CUBIC)
    big = (cv2.GaussianBlur(big, (0, 0), 0.9 * supersample) > 127).astype(np.uint8)
    cnts, hier = cv2.findContours(big, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    canvas = np.zeros_like(big)
    sig = sigma * supersample
    for i, c in enumerate(cnts):
        if len(c) < 6:
            continue
        pts = _smooth_contour(c[:, 0, :].astype(np.float32), sig)
        is_hole = hier[0][i][3] != -1
        cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 0 if is_hole else 1, lineType=cv2.LINE_AA)
    # fillPoly paints holes after parents because RETR_CCOMP lists outer contours before their children
    return cv2.resize(canvas.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.5


# ---------------------------------------------------------------- colour projection
def fisher_channel(img: np.ndarray) -> np.ndarray:
    """Grey image along the colour direction that best separates ink from plate body (per-image Fisher LDA).

    Ink/body labels come from an Otsu split of the plain grey image; the projection w = Sigma^-1 (mu_body - mu_ink)
    then uses all three channels, which recovers faint strokes on coloured plates (teal ink on teal body)."""
    if img.ndim == 2:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = gray <= thr
    if ink.mean() > 0.5:  # strokes are the thin, minority class: on a dark plate with light text it is the bright side
        ink = ~ink
    if ink.sum() < 50 or (~ink).sum() < 50:
        return gray
    x = img.reshape(-1, 3).astype(np.float64)
    m = ink.reshape(-1)
    mu_i, mu_b = x[m].mean(0), x[~m].mean(0)
    cov = np.cov(x[m].T) * m.mean() + np.cov(x[~m].T) * (1 - m.mean()) + 1e-3 * np.eye(3)
    w = np.linalg.solve(cov, mu_b - mu_i)
    w /= (np.linalg.norm(w) + 1e-9)
    p = img.astype(np.float32) @ w.astype(np.float32)
    lo, hi = np.percentile(p, [0.5, 99.5])
    out = np.clip((p - lo) / max(hi - lo, 1e-3) * 255.0, 0, 255).astype(np.uint8)
    # convention for the rest of the pipeline: dark strokes on a light page
    return out if out[~ink].mean() >= out[ink].mean() else 255 - out
