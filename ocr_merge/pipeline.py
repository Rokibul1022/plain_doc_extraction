"""The OCR pipeline with the merged reader: same steps and decision rules as `ocr.pipeline`, one model call per plate.

    from ocr_merge.pipeline import read_plate
    report = read_plate(cv2.imread("car.jpg"))     # YOLO (large frames) -> opencv scan -> PlateNet -> checked result
    report.text, report.verdict
"""

import os
from pathlib import Path

import numpy as np

from ocr.pipeline import PlateReport, build_report, locate_plate, render_report  # noqa: F401  (render_report re-exported)
from ocr_merge.prior import PATH as PRIORS_PATH, Priors
from ocr_merge.reader import MODEL_PATH, MergedReader
from ocr_merge.views import VIEW_SETS
from opencv.scan import scan as cv_scan

_reader: dict = {}
# MERGE_DEVICE=cpu forces the CPU model.
RETRY_DEFAULT = "55"  # a plate that is not "readable" after the first read is read again with this many views (MERGE_RETRY_VIEWS=0 turns it off)
VIEWS_DEFAULT = "20"  # digit-row views per crop (ocr_merge/views.py); MERGE_VIEWS=6|12|20|36|55 trades accuracy for speed


def make_reader(path=MODEL_PATH) -> MergedReader:
    """A merged reader with the deployed settings: MERGE_VIEWS (default 20) and, unless MERGE_PRIOR=0, the class priors in priors.json."""
    r = MergedReader(Path(path))
    r.views = VIEW_SETS[os.environ.get("MERGE_VIEWS", VIEWS_DEFAULT)]
    if os.environ.get("MERGE_PRIOR", "1") != "0" and PRIORS_PATH.exists():
        r.priors = Priors.from_json(PRIORS_PATH)
    if os.environ.get("MERGE_DEVICE", "auto") != "cpu":  # GPU when PyTorch + CUDA + the weights are there, else the ONNX model on the CPU
        from ocr_merge.gpu import WEIGHTS, attach

        twin = Path(path).with_suffix(".pt")
        attach(r, twin if twin.exists() else WEIGHTS if Path(path) == MODEL_PATH else Path("missing.pt"))
    return r


def use_model(path) -> None:
    """Read with another merged ONNX file (used to evaluate held-out models)."""
    _reader["merged"] = make_reader(path)


def reader() -> MergedReader:
    if "merged" not in _reader:
        _reader["merged"] = make_reader()
    return _reader["merged"]


def read_scan(r, force_mode: str | None = None) -> PlateReport:
    """Read one scanned plate: a first vote over `MERGE_VIEWS` views, and a denser vote (`MERGE_RETRY_VIEWS`) if it is not "readable"."""
    views = [r.rectified, r.restored]
    rd = reader()
    cands = rd.candidates(views)
    retry = os.environ.get("MERGE_RETRY_VIEWS", RETRY_DEFAULT)
    retried = retry in VIEW_SETS and build_report(r, cands, force_mode, make_page=False).verdict != "readable"
    if retried:
        cands = rd.candidates(views, VIEW_SETS[retry])  # only the doubtful plates pay for the denser vote
    rep = build_report(r, cands, force_mode)
    rep.retried, rep.views = retried, len(VIEW_SETS[retry] if retried else rd.views)  # (for the UI: how this reading was made)
    return rep


def read_plate(img: np.ndarray, force_mode: str | None = None) -> PlateReport:
    crop = locate_plate(img)
    return read_scan(cv_scan(crop, relief="off", make_page=False), force_mode)
