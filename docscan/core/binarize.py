"""Binarization and clean CamScanner-style plain text extraction."""

from typing import Literal
import cv2
import numpy as np
from skimage.filters import threshold_sauvola

BinarizeMethod = Literal["plain_text", "adaptive", "sauvola", "otsu"]


def remove_small_speckles(binary_img: np.ndarray, min_area: int = 6) -> np.ndarray:
    """Remove tiny isolated dark noise speckles from a white-background binary image."""
    if min_area <= 1 or binary_img.size == 0:
        return binary_img.copy()

    inverted = cv2.bitwise_not(binary_img)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )

    cleaned_inv = np.zeros_like(inverted)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_inv[labels == i] = 255

    return cv2.bitwise_not(cleaned_inv)


def camscanner_clean_plain_text(
    image: np.ndarray,
    stroke_weight: float = 0.0,
) -> np.ndarray:
    """Extract clean, crisp, minimal plain text with snow-white background like CamScanner.

    Unlike harsh thresholding that produces thick, bold, jagged pixelated strokes,
    this normalizes the paper background to pure 255 white and applies a calibrated tone
    curve that preserves thin, sharp typography with smooth anti-aliased edges.

    Args:
        image: BGR or 1-channel grayscale uint8 image.
        stroke_weight: Slider from -1.0 (extra thin & minimal) to 1.0 (bolder).
                       0.0 is the optimal CamScanner clean plain text setting.

    Returns:
        Grayscale uint8 image with pure white background and crisp, clean text.
    """
    if image.size == 0:
        return image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    h, w = gray.shape[:2]

    # Invert dark background cards/plates so text is always dark on white paper
    if float(np.mean(gray)) < 110.0:
        gray = cv2.bitwise_not(gray)

    # 1. Background illumination estimation
    # Kernel size dynamically scaled to image dimensions
    k = max(25, min(101, (min(h, w) // 10) | 1))
    dilated = cv2.dilate(
        gray, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    )
    bg = cv2.medianBlur(dilated, k)

    # 2. Divide by background to flatten paper to pure white
    norm = cv2.divide(gray, np.maximum(bg, 1), scale=255)

    # 3. Tone curve: drives background >= 205 to pure 255 white
    # and keeps text strokes dark, thin, and crisp.
    p_white = 205.0
    # Base black point around 60; adjusting stroke_weight shifts black point
    # positive stroke_weight lowers black point -> darker/bolder
    # negative stroke_weight raises black point -> thinner/more minimal
    p_black = max(10.0, min(120.0, 60.0 + float(stroke_weight) * 35.0))

    norm_f = norm.astype(np.float32)
    clean = np.clip(
        (norm_f - p_black) * (255.0 / (p_white - p_black)), 0.0, 255.0
    ).astype(np.uint8)

    # Clean outer edge pixels that might carry dark borders from surrounding table/vehicle
    border_px = max(1, min(3, int(min(h, w) * 0.015)))
    clean[0:border_px, :] = 255
    clean[-border_px:, :] = 255
    clean[:, 0:border_px] = 255
    clean[:, -border_px:] = 255

    return clean



def clean_plate_and_label_text(
    cropped_img: np.ndarray, auto_invert: bool = True
) -> np.ndarray:
    """Extract clean binary text (backward compatibility)."""
    gray = (
        cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        if cropped_img.ndim == 3
        else cropped_img.copy()
    )
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def binarize(
    gray_image: np.ndarray,
    method: BinarizeMethod = "plain_text",
    block_size: int = 35,
    c_param: float = 14.0,
    sauvola_k: float = 0.25,
    sauvola_r: float = 128.0,
    clean_speckles: bool = True,
    min_speckle_area: int = 6,
    stroke_weight: float = 0.0,
) -> np.ndarray:
    """Convert grayscale image to crisp clean plain text or high-contrast scanner document."""
    if method == "plain_text":
        return camscanner_clean_plain_text(gray_image, stroke_weight=stroke_weight)

    if gray_image.ndim == 3:
        gray = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_image.copy()

    b_size = max(3, block_size | 1)

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "sauvola":
        thresh_map = threshold_sauvola(
            gray, window_size=b_size, k=sauvola_k, r=sauvola_r
        )
        binary = np.where(gray > thresh_map, 255, 0).astype(np.uint8)
    else:  # 'adaptive' Gaussian
        # Calibrated C offset to avoid overly bold, fused strokes
        effective_c = max(4.0, c_param - float(stroke_weight) * 6.0)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            b_size,
            int(round(effective_c)),
        )

    if clean_speckles:
        binary = remove_small_speckles(binary, min_area=min_speckle_area)

    return binary
