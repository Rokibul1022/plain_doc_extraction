"""OpenCV-only plate scan: 4 corners -> straighten -> clean -> plain page. No OCR, no neural network, no training.

    corners, how = find_corners(img)       # classical detection (tight-crop colour clustering or edge/contour search)
    result = scan(img)                     # corners + rectified plate + restored plate + plain document
"""

from dataclasses import dataclass

import cv2
import numpy as np

from opencv.detect import detect_document
from opencv.geometry import order_points
from opencv.plate_find import crop_mode_quad, rectify, subpixel_corners, trim_letterbox
from opencv import quadfit
from opencv.document import make_document
from opencv.restore import restore_plate


EDGE_FIT = True  # verify / refine the colour-cluster quad against the picture's edges


@dataclass
class CvScan:
    corners: np.ndarray  # (4, 2) TL, TR, BR, BL in the input image
    method: str
    rectified: np.ndarray
    restored: np.ndarray
    document: np.ndarray | None  # the plain page: white background, clean dark characters, level (None if make_page=False)


def _full_frame(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])


def _text_inside(img: np.ndarray, quad: np.ndarray) -> float:
    """Share of the thin dark/light strokes (the characters) that lie inside `quad`."""
    from opencv.strokes import stroke_mask as _stroke_mask

    sm = _stroke_mask(img)
    if sm.sum() < 20:
        return 1.0
    m = np.zeros(sm.shape, np.uint8)
    cv2.fillConvexPoly(m, np.round(quad).astype(np.int32), 1)
    return float((sm & m).sum()) / float(sm.sum())


def find_corners(img: np.ndarray) -> tuple[np.ndarray, str]:
    """The plate's 4 corners using only classical OpenCV."""
    h, w = img.shape[:2]
    if max(h, w) <= 720 and 0.55 <= w / h <= 3.5:  # a tight crop: the plate body fills most of the frame
        old = crop_mode_quad(img)
        old = order_points(old) if old is not None and cv2.contourArea(old.astype(np.float32)) >= 0.35 * w * h else None
        new = None
        if EDGE_FIT:
            fit = quadfit.best_quad(img)
            if fit is not None and cv2.contourArea(fit[0]) >= 0.15 * w * h:
                new = order_points(fit[0])
        if new is not None and old is not None:
            # the characters must lie inside the plate: take the quad that keeps more of them, new one on a tie
            if _text_inside(img, old) > _text_inside(img, new) + 0.04:
                new = None
        if new is not None:
            return new, "edge-verified quad"
        if old is not None:
            return subpixel_corners(img, old), "colour-cluster hull"
        return _full_frame(img), "full frame"
    corners, conf, detected = detect_document(img, 600, "plate")  # wide shot: edges + contours + text-density
    if detected:
        return order_points(corners), "edge/contour search"
    return _full_frame(img), "full frame"


def scan(img: np.ndarray, relief: str = "auto", make_page: bool = True) -> CvScan:
    trimmed, (ox, oy) = trim_letterbox(img)
    corners, how = find_corners(trimmed)
    rect = rectify(trimmed, corners)
    restored = restore_plate(rect).image  # classical clean-up only: low light, glare, de-block, denoise, upscale, contrast
    document = make_document(restored, relief=relief) if make_page else None  # relief: "auto", "force" (embossed 3D plate) or "off"
    return CvScan(corners + np.float32([ox, oy]), how, rect, restored, document)
