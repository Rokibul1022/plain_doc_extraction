"""Preset definitions and automatic document classification heuristics."""

from typing import Literal
import cv2
import numpy as np

ModeType = Literal[
    "plain_text",
    "magic_color",
    "grayscale",
    "original",
    "auto",
    "bw",
    "color_enhanced",
    "handwriting",
]


def classify_document_mode(
    image: np.ndarray,
) -> Literal["color_enhanced", "bw", "magic_color", "plain_text"]:
    """Classify document image using color saturation and luminance heuristics.

    Returns:
        'color_enhanced' if color content is present (stamps, signatures, graphics),
        otherwise 'bw'.
    """
    if image.size == 0:
        return "bw"

    # Color check via HSV saturation
    if image.ndim == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        mean_sat = float(np.mean(sat))
        colored_pixel_ratio = float(np.count_nonzero(sat > 45)) / float(sat.size)
        if mean_sat > 22.0 or colored_pixel_ratio > 0.06:
            return "color_enhanced"

    return "bw"
