"""Calibrated reliability of a Bengali digit-row reading (same idea as `verdict.py`).

Fitted by `ocr/scripts/calibrate_bangla.py` on labelled plates (positives and negatives, including
non-Bengali plates the reader cannot read, so garbage reads of foreign plates score low).
"""

import json
from pathlib import Path

import numpy as np

from ocr.bangla_reader import RowReading

MODEL_PATH = Path(__file__).with_name("verdict_bn_model.json")
FEATURES = ["bias", "conf", "min_char", "agree", "six", "log_width"]
_PRIOR = np.array([-7.0, 3.0, 4.0, 1.5, 1.0, 0.2])
# Thresholds measured on 84 verified plates read by models that never saw them: at >= 0.80 the system
# calls ~42 % of plates readable and ~94 % of those are exactly right.
READABLE_AT = 0.80
UNCERTAIN_AT = 0.40


def verdict(p: float) -> str:
    return "readable" if p >= READABLE_AT else "uncertain" if p >= UNCERTAIN_AT else "unreadable"

_model: np.ndarray | None = None


def features(r: RowReading, width_px: float) -> np.ndarray:
    return np.array([
        1.0,
        r.confidence,
        min(r.char_confidence) if r.char_confidence else 0.0,
        r.agreement,
        1.0 if r.ok else 0.0,
        float(np.log(max(width_px, 8.0) / 80.0)),
    ])


def weights() -> np.ndarray:
    global _model
    if _model is None:
        try:
            w = np.array(json.loads(MODEL_PATH.read_text())["weights"], dtype=float)
            _model = w if w.shape == _PRIOR.shape else _PRIOR
        except (OSError, ValueError, KeyError):
            _model = _PRIOR
    return _model


def reliability(r: RowReading, width_px: float) -> float:
    if not r.ok:
        return 0.0
    z = float(features(r, width_px) @ weights())
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))
