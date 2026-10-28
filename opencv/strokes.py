"""Where thin dark strokes exist (black-hat): characters, as opposed to big dark areas - OpenCV + NumPy only."""

import cv2
import numpy as np


def stroke_mask(img: np.ndarray) -> np.ndarray:
    """Where thin dark strokes exist (black-hat), slightly grown. Large dark areas give no response."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
    k = max(9, int(0.11 * gray.shape[0]) | 1)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    top = float(np.percentile(bh, 99.0))
    if top < 14.0:
        return np.zeros(gray.shape, np.uint8)
    return cv2.dilate((bh > 0.22 * top).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
