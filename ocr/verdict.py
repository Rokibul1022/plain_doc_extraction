"""Calibrated reliability of a plate reading: P(text is exactly right), and the verdict from it.

The OCR's own confidence is not trustworthy on degraded plates (it can be confidently wrong), so
reliability is a logistic model over several independent signals - confidence, weakest character,
agreement between enhanced variants, format validity, plate size and sharpness. The weights are
fitted by `ocr/scripts/calibrate_verdict.py` on labelled real and degraded plates and stored in
`verdict_model.json`; a conservative hand-set prior is used if that file is missing.
"""

import json
from pathlib import Path

import numpy as np

from ocr.ocr import PlateReading
from opencv.quality import PlateQuality

MODEL_PATH = Path(__file__).with_name("verdict_model.json")
FEATURES = ["bias", "conf", "min_char", "agree", "format_ok", "len_ok", "log_width", "sharp", "q_raw", "multi"]

# Conservative prior used until a fitted model exists.
_PRIOR = np.array([-6.0, 1.5, 3.0, 1.0, 0.8, 0.5, 0.5, 2.0, 1.0, 0.3])

READABLE_AT = 0.90  # cross-validated: reads at P>=0.90 are exactly right ~96 % of the time
UNCERTAIN_AT = 0.45

_model: np.ndarray | None = None


def features(reading: PlateReading, q0: PlateQuality) -> np.ndarray:
    n = max(1, reading.variants)
    return np.array([
        1.0,
        reading.confidence,
        min(reading.char_confidence) if reading.char_confidence else 0.0,
        reading.votes / n,
        1.0 if reading.format_ok else 0.0,
        1.0 if 4 <= len(reading.text) <= 9 else 0.0,
        float(np.log(max(q0.metrics.get("width_px", 40.0), 8.0) / 60.0)),
        min(q0.metrics.get("sharpness", 0.0), 0.6),
        q0.score / 100.0,
        1.0 if n > 1 else 0.0,
    ])


def _weights() -> np.ndarray:
    global _model
    if _model is None:
        try:
            w = np.array(json.loads(MODEL_PATH.read_text())["weights"], dtype=float)
            _model = w if w.shape == _PRIOR.shape else _PRIOR
        except (OSError, ValueError, KeyError):
            _model = _PRIOR
    return _model


def reliability(reading: PlateReading, q0: PlateQuality) -> float:
    if not reading.text:
        return 0.0
    z = float(features(reading, q0) @ _weights())
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))


def verdict_from(p: float) -> str:
    return "readable" if p >= READABLE_AT else "uncertain" if p >= UNCERTAIN_AT else "unreadable"


def fit(x: np.ndarray, y: np.ndarray, l2: float = 0.05, iters: int = 400) -> np.ndarray:
    """L2-regularised logistic regression by Newton steps (small, dependency-free)."""
    w = np.zeros(x.shape[1])
    reg = l2 * np.eye(x.shape[1])
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
        g = x.T @ (p - y) / len(y) + reg @ w / len(y)
        h = (x.T * (p * (1 - p))) @ x / len(y) + reg / len(y) + 1e-6 * np.eye(x.shape[1])
        step = np.linalg.solve(h, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w
