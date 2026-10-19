"""Sharpening filters, unsharp masking, and blur-adaptive strength control."""

import cv2
import numpy as np


def estimate_blur(image: np.ndarray) -> float:
    """Estimate image sharpness using the variance of the Laplacian operator.

    Higher values indicate sharp edges; lower values (< 100) indicate blurry images.
    """
    if image.size == 0:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sharpen(
    image: np.ndarray,
    amount: float = 1.0,
    sigma: float = 1.5,
    auto_scale: bool = False,
) -> np.ndarray:
    """Enhance document edge sharpness via unsharp masking.

    Args:
        image: BGR or grayscale uint8 array.
        amount: Sharpening boost factor (0.0 to 3.0).
        sigma: Standard deviation of Gaussian blur for unsharp mask.
        auto_scale: When True, dynamically adapts amount based on measured blur.

    Returns:
        Sharpened uint8 array.
    """
    if image.size == 0 or (amount <= 0.0 and not auto_scale):
        return image.copy()

    effective_amount = float(amount)

    if auto_scale:
        blur_var = estimate_blur(image)
        if blur_var < 80.0:
            # Heavily blurred image: boost sharpening safely
            effective_amount = min(2.0, max(1.2, effective_amount * 1.5))
        elif blur_var > 600.0:
            # Already very sharp: reduce sharpening to avoid harsh halo artifacts
            effective_amount = min(0.6, effective_amount * 0.5)
        else:
            # Mild-to-moderate blur: smooth scaling
            factor = 1.0 + (300.0 - min(300.0, blur_var)) / 300.0 * 0.4
            effective_amount = min(1.8, effective_amount * factor)

    if effective_amount <= 0.01:
        return image.copy()

    # Gaussian blur for high-pass unsharp mask
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=max(0.1, sigma))
    # image * (1 + amount) - blurred * amount
    sharpened = cv2.addWeighted(
        image, 1.0 + effective_amount, blurred, -effective_amount, 0
    )
    return sharpened
