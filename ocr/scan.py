"""End-to-end plate scan for one image: detect -> rectify -> assess -> restore -> read."""

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ocr.bangla_reader import BanglaRowReader
from ocr.bangla_top import BanglaTopReader
from ocr.deblur import deblur_guided, deblur_guided_bn
from opencv.document import make_document
from ocr.typeset import render_typed
from ocr.detector import PlateDetection, PlateDetector, rectify
from ocr.ocr import PlateOCR, PlateReading
from opencv.quality import PlateQuality, assess_quality
from opencv.readable import make_readable
from opencv.plate_find import trim_letterbox  # noqa: F401  (also re-exported)
from opencv.restore import restore_plate
from ocr.verdict import reliability, verdict_from
from ocr import verdict_bn


@dataclass
class PlateResult:
    detection: PlateDetection
    crop: np.ndarray
    restored: np.ndarray
    readable: np.ndarray
    quality: PlateQuality
    quality_after: PlateQuality
    steps: list[str]
    reading: PlateReading
    verdict: str
    timings_ms: dict[str, float] = field(default_factory=dict)
    document: np.ndarray | None = None  # the plain document: typed page if reliable, else the cleaned scan
    scan_document: np.ndarray | None = None  # the faithful cleaned scan (always present)
    typed: np.ndarray | None = None  # best-guess typed page (shown with the verdict; may be wrong if not 'readable')

    def to_dict(self) -> dict:
        return {
            "detection": self.detection.to_dict(),
            "quality": self.quality.to_dict(),
            "quality_after": self.quality_after.to_dict(),
            "steps": list(self.steps),
            "reading": self.reading.to_dict(),
            "verdict": self.verdict,
            "document_kind": "typed" if (self.document is not None and self.document is not self.scan_document) else "scan",
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
        }


_detector: PlateDetector | None = None
_ocr: PlateOCR | None = None


def get_detector() -> PlateDetector:
    global _detector
    if _detector is None:
        _detector = PlateDetector()
    return _detector


_bangla: BanglaRowReader | None = None


def get_bangla() -> BanglaRowReader:
    global _bangla
    if _bangla is None:
        _bangla = BanglaRowReader()
    return _bangla


_bangla_top: BanglaTopReader | None = None


def get_bangla_top() -> BanglaTopReader:
    global _bangla_top
    if _bangla_top is None:
        _bangla_top = BanglaTopReader()
    return _bangla_top


def get_ocr() -> PlateOCR:
    global _ocr
    if _ocr is None:
        _ocr = PlateOCR()
    return _ocr


def verdict_for(reading: PlateReading, quality: PlateQuality, quality_after: PlateQuality | None = None) -> str:
    """Honest readability call from a calibrated reliability model (see `verdict.py`).

    `quality` must be the plate's quality *before* restoration (its true size and sharpness)."""
    p = reliability(reading, quality)
    reading.reliability = p
    if not reading.text:
        return "unreadable"
    return verdict_from(p)


def read_plate_crop(
    crop: np.ndarray,
    *,
    plate_width: float | None = None,
    ocr: PlateOCR | None = None,
    do_restore: bool = True,
    do_ocr: bool = True,
) -> tuple[np.ndarray, np.ndarray, PlateQuality, PlateQuality, list[str], PlateReading, dict[str, float]]:
    """Quality -> restore -> readable -> OCR for an already-rectified plate crop."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    quality = assess_quality(crop, plate_width)
    timings["quality"] = (time.perf_counter() - t0) * 1000

    engine = ocr or (get_ocr() if do_ocr else None)
    steps: list[str] = []
    variants = [crop]

    t0 = time.perf_counter()
    work = crop
    if do_restore and engine is not None and engine.available:
        guided = deblur_guided(crop, engine)  # OCR-validated: kept only if the read clearly improves
        if guided is not None:
            work, desc, _ = guided
            steps.append(f"deblur({desc})")
            variants.append(work)
    timings["deblur"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    if do_restore:
        rr = restore_plate(work, assess_quality(work, plate_width), plate_width=plate_width)
        restored, q_after = rr.image, rr.quality_after
        steps += rr.steps
        variants.append(restored)
    else:
        restored, q_after = crop, quality
    timings["restore"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    readable = make_readable(restored)
    timings["readable"] = (time.perf_counter() - t0) * 1000

    reading = PlateReading()
    if do_ocr and engine is not None:
        t0 = time.perf_counter()
        reading = engine.read_variants(variants) if len(variants) > 1 else engine.read(crop)
        timings["ocr"] = (time.perf_counter() - t0) * 1000
    return restored, readable, quality, q_after, steps, reading, timings


@dataclass
class FullRead:
    """Everything produced for one rectified plate crop."""

    restored: np.ndarray
    readable: np.ndarray
    quality: PlateQuality
    quality_after: PlateQuality
    steps: list[str]
    reading: PlateReading
    verdict: str
    timings: dict[str, float]
    document: np.ndarray  # the plain document: typed page when the reading is reliable, else the cleaned scan
    scan_document: np.ndarray  # the faithful cleaned scan
    typed: np.ndarray | None  # best-guess typed page (any reading), None if there is no text or no font


def bengali_text(digits: str) -> str:
    return f"{digits[:2]}-{digits[2:]}" if len(digits) == 6 else digits


def read_plate_full(
    crop: np.ndarray,
    *,
    plate_width: float | None = None,
    ocr: PlateOCR | None = None,
    bangla: BanglaRowReader | None = None,
    do_restore: bool = True,
    do_ocr: bool = True,
    latin: bool = False,
) -> FullRead:
    """Everything for one rectified plate crop: restore, read, and produce the plain document.

    This product reads Bangladeshi (Bengali) plates, so the Bengali readers are the default. The
    Latin reader only runs with `latin=True` (for foreign plates) and then competes on reliability."""
    engine = (ocr or get_ocr()) if (latin and do_ocr) else None
    restored, readable, q, qa, steps, latin_read, timings = read_plate_crop(
        crop, plate_width=plate_width, ocr=engine, do_restore=do_restore, do_ocr=bool(engine)
    )
    width = plate_width if plate_width is not None else crop.shape[1]
    latin_read.reliability = reliability(latin_read, q) if latin_read.text else 0.0
    reading = latin_read if latin else PlateReading(model="bangla-row", script="bengali")

    reader = bangla or get_bangla()
    if do_ocr and reader.available:
        t0 = time.perf_counter()
        row = reader.read([crop, restored])
        p_bn = verdict_bn.reliability(row, width)
        if do_restore and p_bn < 0.93:  # remove blur, judged by the reader's own reliability (no labels)
            g = deblur_guided_bn(crop, reader, width)
            if g is not None:
                row2 = reader.read([crop, restored, g[0]])
                p2 = verdict_bn.reliability(row2, width)
                if row2.ok and p2 > p_bn:
                    row, p_bn = row2, p2
                    steps.append(f"deblur({g[1]})")
        timings["bengali"] = (time.perf_counter() - t0) * 1000
        if row.ok and p_bn > (latin_read.reliability or 0.0 if latin else 0.0):
            reading = PlateReading(
                text=bengali_text(row.text), confidence=row.confidence, char_confidence=row.char_confidence,
                model="bangla-row", format_ok=True, votes=round(row.agreement * row.n_views), variants=row.n_views,
                reliability=p_bn, script="bengali",
            )
        top_reader = get_bangla_top()
        if reading.script == "bengali" and top_reader.available:
            top = top_reader.read([crop, restored])
            reading.top = top.text
    if reading.script == "bengali":
        verdict = verdict_bn.verdict(reading.reliability or 0.0) if reading.text else "unreadable"
    else:
        verdict = verdict_from(reading.reliability or 0.0) if reading.text else "unreadable"

    scan_doc = make_document(restored)
    typed = None
    if reading.text:
        try:
            typed = render_typed([reading.top, reading.text] if reading.top else [reading.text])
        except Exception:  # no shaping stack / font available: the cleaned scan is the document
            typed = None
    document = typed if (verdict == "readable" and typed is not None) else scan_doc
    return FullRead(restored, readable, q, qa, steps, reading, verdict, timings, document, scan_doc, typed)




def scan_image(
    img: np.ndarray,
    *,
    detector: PlateDetector | None = None,
    ocr: PlateOCR | None = None,
    max_plates: int = 6,
    do_restore: bool = True,
    do_ocr: bool = True,
    assume_crop: bool = True,
    latin: bool = False,
) -> list[PlateResult]:
    """Find every plate in `img` and return restored, readable crops with text and a verdict.

    `assume_crop`: if nothing is detected and the image itself looks like a plate-shaped crop,
    treat the whole image as the plate."""
    det = detector or get_detector()
    img, (ox, oy) = trim_letterbox(img)  # detection runs on the trimmed image; results are shifted back
    t0 = time.perf_counter()
    detections = det.detect(img, max_plates=max_plates)
    t_det = (time.perf_counter() - t0) * 1000

    if not detections and assume_crop:
        h, w = img.shape[:2]
        if 1.1 <= w / h <= 6.5 and max(h, w) <= 700:
            quad = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
            detections = [PlateDetection(quad, (0, 0, w - 1, h - 1), 0.3, "whole-image")]

    results: list[PlateResult] = []
    for d in detections:
        crop = rectify(img, d.quad)
        fr = read_plate_full(crop, plate_width=d.width_px, ocr=ocr, do_restore=do_restore, do_ocr=do_ocr, latin=latin)
        timings = fr.timings
        timings["detect"] = t_det
        if ox or oy:  # report coordinates in the caller's (untrimmed) image
            shift = np.float32([ox, oy])
            d = PlateDetection(d.quad + shift, (d.box[0] + ox, d.box[1] + oy, d.box[2] + ox, d.box[3] + oy),
                               d.confidence, d.source)
        results.append(
            PlateResult(d, crop, fr.restored, fr.readable, fr.quality, fr.quality_after, fr.steps, fr.reading,
                        fr.verdict, timings, fr.document, fr.scan_document, fr.typed)
        )
    return results
