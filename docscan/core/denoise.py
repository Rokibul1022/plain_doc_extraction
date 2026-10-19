"""Denoising filters: median, bilateral, and Non-Local Means.

Note: Over-filtering may erase thin handwriting strokes or small diacritics.
Keep denoise strength modest (0.2-0.5) for text and handwriting documents.
"""

from typing import Literal
import cv2
import numpy as np


DenoiseType = Literal["none", "median", "bilateral", "nlm"]


def denoise(
    image: np.ndarray,
    method: DenoiseType = "none",
    strength: float = 0.5,
) -> np.ndarray:
    """Apply selected denoising algorithm to the image.

    Args:
        image: BGR or grayscale uint8 array.
        method: Denoising technique ('none', 'median', 'bilateral', 'nlm').
        strength: Normalized intensity from 0.0 (no effect) to 1.0 (heavy smoothing).

    Returns:
        Denoised uint8 array.
    """
    if method == "none" or strength <= 0.0 or image.size == 0:
        return image.copy()

    s = np.clip(strength, 0.0, 1.0)

    if method == "median":
        # Kernel size 3 for mild/medium, 5 for heavy
        k = 3 if s < 0.6 else 5
        filtered = cv2.medianBlur(image, k)
        return cv2.addWeighted(filtered, s, image, 1.0 - s, 0)

    if method == "bilateral":
        # Bilateral filter preserves sharp edges while smoothing flat noise
        d = int(5 + round(s * 6))  # 5 to 11
        sigma_color = 20.0 + s * 60.0  # 20 to 80
        sigma_space = 20.0 + s * 60.0
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    if method == "nlm":
        # Non-Local Means (higher quality, computationally heavier)
        h = max(3.0, s * 12.0)
        h_color = max(3.0, s * 12.0)
        if image.ndim == 3:
            return cv2.fastNlMeansDenoisingColored(
                image, None, h=h, hColor=h_color, templateWindowSize=7, searchWindowSize=15
            )
        return cv2.fastNlMeansDenoising(
            image, None, h=h, templateWindowSize=7, searchWindowSize=15
        )

    return image.copy()
