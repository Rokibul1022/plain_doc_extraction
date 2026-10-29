"""The OCR pipeline with the merged reader: same steps and decision rules as `ocr.pipeline`, one model call per plate.

    from ocr_merge.pipeline import read_plate
    report = read_plate(cv2.imread("car.jpg"))     # YOLO (large frames) -> opencv scan -> PlateNet -> checked result
    report.text, report.verdict
"""

from pathlib import Path

import numpy as np

from ocr.pipeline import PlateReport, build_report, locate_plate, render_report  # noqa: F401  (render_report re-exported)
from ocr_merge.reader import MODEL_PATH, MergedReader
from opencv.scan import scan as cv_scan

_reader: dict = {}


def use_model(path) -> None:
    """Read with another merged ONNX file (used to evaluate held-out models)."""
    _reader["merged"] = MergedReader(Path(path))


def reader() -> MergedReader:
    if "merged" not in _reader:
        _reader["merged"] = MergedReader(MODEL_PATH)
    return _reader["merged"]


def read_plate(img: np.ndarray, force_mode: str | None = None) -> PlateReport:
    crop = locate_plate(img)
    r = cv_scan(crop, relief="off", make_page=False)
    return build_report(r, reader().candidates([r.rectified, r.restored]), force_mode)
