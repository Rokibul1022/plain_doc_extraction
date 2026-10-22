"""Orchestrates image processing stages from warping to CamScanner-style enhancement."""

from dataclasses import dataclass
import time
import cv2
import numpy as np

from docscan.config import settings
from docscan.core.binarize import binarize, camscanner_clean_plain_text
from docscan.core.denoise import denoise
from docscan.core.geometry import deshear, deskew, order_points, rotate90, warp
from docscan.core.illumination import apply_clahe, remove_shadows
from docscan.core.io_utils import resize_to_max
from docscan.core.modes import classify_document_mode
from docscan.core.sharpen import sharpen
from docscan.schemas import ProcessParams


@dataclass
class ProcessResult:
    """Encapsulates the final processed document and execution telemetry."""

    image: np.ndarray
    corners: np.ndarray
    mode_used: str
    timings_ms: dict[str, float]


def _apply_brightness_contrast(
    img: np.ndarray, brightness: float, contrast: float
) -> np.ndarray:
    """Adjust brightness and contrast on images."""
    if brightness == 0.0 and contrast == 0.0:
        return img
    alpha = max(0.1, 1.0 + (contrast / 100.0))
    beta = brightness * 1.27
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def camscanner_magic_color(image: np.ndarray) -> np.ndarray:
    """Enhance document photo into a clean CamScanner Magic Color document.

    Background paper becomes pure snow-white while keeping colors on certificates,
    forms, stamps, and signatures rich and clear.
    """
    if image.size == 0 or image.ndim != 3:
        return image.copy()

    h, w = image.shape[:2]
    k = max(25, min(101, (min(h, w) // 12) | 1))
    channels = cv2.split(image)
    out_channels = []
    for ch in channels:
        dilated = cv2.dilate(ch, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        bg = cv2.medianBlur(dilated, k)
        norm = cv2.divide(ch, np.maximum(bg, 1), scale=255)
        clean = np.clip(
            (norm.astype(np.float32) - 45.0) * (255.0 / (215.0 - 45.0)), 0.0, 255.0
        ).astype(np.uint8)
        out_channels.append(clean)
    return cv2.merge(out_channels)


def process(
    image: np.ndarray,
    corners: np.ndarray | list,
    params: ProcessParams,
    *,
    preview: bool = False,
) -> ProcessResult:
    """Run the complete document scanning pipeline.

    Args:
        image: Original full-resolution BGR image.
        corners: 4-point quadrilateral in original image coordinates.
        params: Configuration parameters and slider values.
        preview: When True, intermediate steps are run on a downscaled copy for low latency.

    Returns:
        ProcessResult containing enhanced image, mode used, and stage timings.
    """
    timings: dict[str, float] = {}
    ordered_corners = order_points(corners)

    # 1. Perspective warp
    t0 = time.perf_counter()
    warped = warp(image, ordered_corners)

    # 2. Rotation & deskew (straighten residual tilt so it is level and straight)
    if params.rotation != 0:
        warped = rotate90(warped, params.rotation)
    warped = deshear(deskew(warped))
    timings["warp_and_geom_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 3. Downscale for preview if requested
    if preview:
        warped, _ = resize_to_max(warped, settings.preview_max_dim)

    # 4. Determine operational mode
    t_mode = time.perf_counter()
    if params.mode == "auto":
        actual_mode = classify_document_mode(warped)
    else:
        actual_mode = params.mode
    timings["mode_detect_ms"] = round((time.perf_counter() - t_mode) * 1000, 2)

    # 5. Process according to mode
    t_enhance = time.perf_counter()

    if actual_mode == "original":
        output = _apply_brightness_contrast(warped, params.brightness, params.contrast)

    elif actual_mode in ("magic_color", "color_enhanced"):
        magic = camscanner_magic_color(warped)
        output = _apply_brightness_contrast(magic, params.brightness, params.contrast)

    elif actual_mode == "grayscale":
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
        flat = remove_shadows(gray, strength=0.90)
        clahe_applied = apply_clahe(flat, clip_limit=1.5)
        output = _apply_brightness_contrast(clahe_applied, params.brightness, params.contrast)

    elif actual_mode in ("plain_text", "bw"):
        # CamScanner Plain Text: snow white paper with thin, clean, crisp typography
        if params.threshold_method in ("sauvola", "otsu", "adaptive"):
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
            flat = remove_shadows(gray, strength=1.0)
            output = binarize(
                flat,
                method=params.threshold_method,
                block_size=params.threshold_block_size,
                c_param=params.threshold_c,
                clean_speckles=params.remove_speckles,
                stroke_weight=params.stroke_weight,
            )
        else:
            output = camscanner_clean_plain_text(
                warped, stroke_weight=params.stroke_weight
            )
        if params.brightness != 0.0 or params.contrast != 0.0:
            output = _apply_brightness_contrast(output, params.brightness, params.contrast)

    elif actual_mode == "handwriting":
        # Extra gentle stroke preservation for thin pencil/pen strokes
        output = camscanner_clean_plain_text(warped, stroke_weight=min(-0.2, params.stroke_weight))
        if params.brightness != 0.0 or params.contrast != 0.0:
            output = _apply_brightness_contrast(output, params.brightness, params.contrast)

    else:
        output = camscanner_clean_plain_text(warped, stroke_weight=params.stroke_weight)

    timings["enhance_ms"] = round((time.perf_counter() - t_enhance) * 1000, 2)
    timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    return ProcessResult(
        image=output,
        corners=ordered_corners,
        mode_used=actual_mode,
        timings_ms=timings,
    )
