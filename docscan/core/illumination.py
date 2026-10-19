"""Illumination and shadow correction, background normalization, and contrast enhancement."""

import cv2
import numpy as np


def estimate_background_channel(channel: np.ndarray, kernel_size: int) -> np.ndarray:
    """Estimate uneven illumination background for a single channel.

    Uses dilation to eliminate text/foreground strokes followed by median blur.
    """
    k = max(3, kernel_size | 1)  # ensure odd integer
    # Dilate with a small kernel to cover dark text strokes
    dilate_size = max(3, min(15, (k // 6) | 1))
    dilated = cv2.dilate(
        channel, cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size, dilate_size))
    )
    # Median blur smooths the illuminated background
    # OpenCV medianBlur ksize must be <= 500 and odd
    blur_k = min(51, k | 1)
    return cv2.medianBlur(dilated, blur_k)


def remove_shadows(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Remove shadows and uneven background lighting across color channels.

    Args:
        image: BGR or 1-channel uint8 array.
        strength: Blending strength of illumination correction (0.0 to 1.0).

    Returns:
        Illumination-flattened uint8 image.
    """
    if strength <= 0.0 or image.size == 0:
        return image.copy()

    h, w = image.shape[:2]
    # Optimal background kernel size is roughly 1/10 to 1/8 of the smallest dimension
    base_k = max(21, min(101, (min(h, w) // 10) | 1))

    if image.ndim == 2:
        bg = estimate_background_channel(image, base_k)
        corrected = cv2.divide(image, np.maximum(bg, 1), scale=255)
    else:
        channels = cv2.split(image)
        out_channels = []
        for ch in channels:
            bg = estimate_background_channel(ch, base_k)
            c = cv2.divide(ch, np.maximum(bg, 1), scale=255)
            out_channels.append(c)
        corrected = cv2.merge(out_channels)

    if strength >= 1.0:
        return corrected
    return cv2.addWeighted(corrected, strength, image, 1.0 - strength, 0)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization.

    On BGR images, CLAHE is applied to the L (Lightness) channel in LAB color space.
    On grayscale images, CLAHE is applied directly.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    if image.ndim == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_enhanced = clahe.apply(l_ch)
    lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
