"""Plate detection: learned detector (ONNX) + tiling for small plates + classical fallback,
followed by corner refinement and aspect-aware rectification.

The learned model (`open-image-models`, MIT) gives an axis-aligned box; the plate's own
tilted corners are then recovered inside that box with the geometric plate finder, so the
crop can be perspective-corrected instead of just cut out.
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from opencv.detect import detect_document
from opencv.geometry import order_points
from opencv.plate_find import (  # noqa: F401  (re-exported: callers import them from here)
    COMMON_ASPECTS, _iou_box, crop_mode_quad, rectify, refine_quad, subpixel_corners, true_aspect)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "yolo-v9-t-640-license-plate-end2end"


@dataclass
class PlateDetection:
    quad: np.ndarray  # (4, 2) TL, TR, BR, BL in source-image pixels
    box: tuple[int, int, int, int]
    confidence: float
    source: str

    @property
    def width_px(self) -> float:
        q = self.quad
        return float(0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])))

    @property
    def height_px(self) -> float:
        q = self.quad
        return float(0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])))

    def to_dict(self) -> dict:
        return {
            "quad": [[round(float(x), 1), round(float(y), 1)] for x, y in self.quad],
            "box": list(self.box),
            "confidence": round(float(self.confidence), 3),
            "source": self.source,
            "width_px": round(self.width_px, 1),
        }




def _nms(dets: list[tuple[tuple[int, int, int, int], float]], thr: float = 0.4):
    keep: list[tuple[tuple[int, int, int, int], float]] = []
    for box, conf in sorted(dets, key=lambda d: -d[1]):
        if all(_iou_box(box, k[0]) < thr for k in keep):
            keep.append((box, conf))
    return keep


class LearnedDetector:
    """Lazy ONNX plate detector; `available` is False if the model cannot be loaded."""

    def __init__(self, model: str = DEFAULT_MODEL, conf: float = 0.25) -> None:
        self.model_name = model
        self.conf = conf
        self._det = None
        self._tried = False

    @property
    def available(self) -> bool:
        self._load()
        return self._det is not None

    def _load(self) -> None:
        if self._tried:
            return
        self._tried = True
        try:
            from open_image_models import create_detector

            from ocr.ortutil import session_options

            self._det = create_detector(self.model_name, conf_thresh=self.conf, providers=["CPUExecutionProvider"],
                                         sess_options=session_options())
        except Exception as exc:
            logger.warning("Learned plate detector unavailable: %s", exc)
            self._det = None

    def _run(self, img: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        h, w = img.shape[:2]
        out = []
        for r in self._det.predict(img):
            b = r.bounding_box
            x1, y1, x2, y2 = max(0, b.x1), max(0, b.y1), min(w - 1, b.x2), min(h - 1, b.y2)
            if x2 - x1 >= 8 and y2 - y1 >= 4:
                out.append(((int(x1), int(y1), int(x2), int(y2)), float(r.confidence)))
        return out

    def detect(self, img: np.ndarray, tile: bool = True) -> list[tuple[tuple[int, int, int, int], float]]:
        """Boxes (x1, y1, x2, y2) with confidence; tiles large frames so small plates survive the
        model's fixed input size."""
        self._load()
        if self._det is None:
            return []
        h, w = img.shape[:2]
        dets = self._run(img)
        if tile and max(h, w) > 900:
            # tiles of ~55 % of the frame with 25 % overlap magnify distant plates ~2x
            tw, th = int(w * 0.55), int(h * 0.55)
            xs = sorted({0, (w - tw) // 2, w - tw})
            ys = sorted({0, (h - th) // 2, h - th})
            for y0 in ys:
                for x0 in xs:
                    for (bx1, by1, bx2, by2), c in self._run(img[y0:y0 + th, x0:x0 + tw]):
                        dets.append(((bx1 + x0, by1 + y0, bx2 + x0, by2 + y0), c))
        return _nms(dets)












class PlateDetector:
    """Learned detection with classical fallback; always returns tight, ordered quads."""

    def __init__(self, model: str = DEFAULT_MODEL, conf: float = 0.25) -> None:
        self.learned = LearnedDetector(model, conf)

    def detect(self, img: np.ndarray, max_plates: int = 6, refine: bool = True, tile: bool = True) -> list[PlateDetection]:
        found: list[PlateDetection] = []
        for box, conf in self.learned.detect(img, tile=tile)[:max_plates]:
            quad, ok = refine_quad(img, box) if refine else (
                np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]], np.float32), False)
            if ok:
                quad = subpixel_corners(img, quad)
            found.append(PlateDetection(quad, box, conf, "learned+quad" if ok else "learned"))
        if found:
            return found

        # tight crop (the plate fills the frame, as in cropped datasets): locate the plate body
        h, w = img.shape[:2]
        if max(h, w) <= 520 and 1.0 <= w / h <= 3.0:
            quad = crop_mode_quad(img)
            area = cv2.contourArea(quad) / float(w * h) if quad is not None else 0.0
            if quad is None or area < 0.35:
                quad = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
            x1, y1 = quad.min(axis=0)
            x2, y2 = quad.max(axis=0)
            return [PlateDetection(quad, (int(x1), int(y1), int(x2), int(y2)), 0.5, "crop")]

        # classical fallback: one plate-like region (works for tight, clean crops)
        corners, conf, detected = detect_document(img, 600, "plate")
        if detected:
            quad = order_points(corners)
            wpx = 0.5 * (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3]))
            hpx = 0.5 * (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1]))
            if 1.0 <= wpx / max(hpx, 1) <= 6.5 and wpx >= 24:
                x1, y1 = quad.min(axis=0)
                x2, y2 = quad.max(axis=0)
                return [PlateDetection(quad, (int(x1), int(y1), int(x2), int(y2)), float(conf) * 0.8, "classical")]
        return []
