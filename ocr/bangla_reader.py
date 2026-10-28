"""Bengali plate digit-row reader: ONNX CRNN with test-time voting and per-character confidence.

Reads the six digits of the lower row of a Bangladeshi plate (`NN-NNNN`) from a rectified plate
crop. Several slightly different views of the digit region are read and combined, so one unlucky
crop does not decide the result; disagreement lowers the confidence instead.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ocr.digits import DIGITS
from ocr.rowprep import IN_H, IN_W

logger = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).with_name("bangla_row.onnx")
BLANK = 10
EXPECTED = 6

# views of the digit region: (top fraction of the plate where the region starts, left/right trim)
# views of the digit region: (top fraction of the plate where it starts, left/right trim). The reader votes over them.
VIEWS = [(0.40, 0.0), (0.34, 0.0), (0.46, 0.0), (0.40, 0.04), (0.37, 0.025), (0.43, 0.025)]


@dataclass
class RowReading:
    text: str = ""  # Bengali digits, "" if the row could not be read as six digits
    confidence: float = 0.0
    char_confidence: list[float] = field(default_factory=list)
    agreement: float = 0.0  # fraction of views that agreed with the final text
    n_views: int = 0
    raw: list[str] = field(default_factory=list)
    alts: list[list[tuple[str, float]]] = field(default_factory=list)  # per position: (digit, share of the vote), best first

    @property
    def ok(self) -> bool:
        return len(self.text) == EXPECTED

    def to_dict(self) -> dict:
        return {"text": self.text, "confidence": round(self.confidence, 3),
                "char_confidence": [round(float(c), 3) for c in self.char_confidence],
                "agreement": round(self.agreement, 3), "views": self.n_views}


def region(crop: np.ndarray, top: float = 0.40, trim: float = 0.0) -> np.ndarray:
    """Grey digit region of a rectified plate crop, resized to the network input (IN_H x IN_W)."""
    h, w = crop.shape[:2]
    x0, x1 = int(trim * w), w - int(trim * w)
    part = crop[int(top * h):, x0:x1]
    g = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY) if part.ndim == 3 else part
    return cv2.resize(g, (IN_W, IN_H), interpolation=cv2.INTER_AREA if g.shape[1] > IN_W else cv2.INTER_CUBIC)


def decode_probs(p: np.ndarray) -> tuple[str, list[float]]:
    """Greedy CTC decode of (T, 11) probabilities -> (digits, per-digit confidence)."""
    ids = p.argmax(-1)
    chars, conf, prev, best = [], [], -1, 0.0
    for t, k in enumerate(ids):
        if k != prev:
            if prev not in (-1, BLANK):
                conf.append(best)
            if k != BLANK:
                chars.append(DIGITS[k])
                best = 0.0
        if k != BLANK:
            best = max(best, float(p[t, k]))
        prev = k
    if prev not in (-1, BLANK):
        conf.append(best)
    return "".join(chars), conf


class BanglaRowReader:
    def __init__(self, path: Path = MODEL_PATH, view=None) -> None:
        self.path = path
        self.view = view  # optional transform of each crop before reading (e.g. opencv.relief.relief_view)
        self._sess = None
        self._tried = False

    @property
    def available(self) -> bool:
        self._load()
        return self._sess is not None

    def _load(self) -> None:
        if self._tried:
            return
        self._tried = True
        try:
            import onnxruntime as ort
            from ocr.ortutil import session_options

            self._sess = ort.InferenceSession(str(self.path), sess_options=session_options(), providers=["CPUExecutionProvider"])
        except Exception as exc:
            logger.warning("Bengali digit reader unavailable (%s): %s", self.path.name, exc)
            self._sess = None

    def probs(self, regions: list[np.ndarray]) -> np.ndarray:
        x = np.stack(regions).astype(np.float32)[:, None] / 255.0
        return np.exp(self._sess.run(None, {"x": x})[0])

    def read(self, crop: np.ndarray | list[np.ndarray]) -> RowReading:
        """Read a rectified plate crop (or several versions of it, whose views are pooled)."""
        self._load()
        crops = [c for c in (crop if isinstance(crop, list) else [crop]) if c is not None and c.size]
        if self._sess is None or not crops:
            return RowReading()
        if self.view is not None:
            crops = [self.view(c) for c in crops]
        p = self.probs([region(c, top, trim) for c in crops for top, trim in VIEWS])
        decoded = [decode_probs(pi) for pi in p]
        raw = [d[0] for d in decoded]
        six = [(t, c) for t, c in decoded if len(t) == EXPECTED and len(c) == EXPECTED]
        if not six:
            return RowReading(raw=raw, n_views=len(decoded))
        # per-position vote, each view weighted by its own confidence in that character
        text, conf, alts = [], [], []
        for i in range(EXPECTED):
            tally: dict[str, float] = {}
            for t, c in six:
                tally[t[i]] = tally.get(t[i], 0.0) + c[i]
            ch = max(tally, key=tally.get)
            text.append(ch)
            total = sum(tally.values()) or 1.0
            alts.append([(k, v / total) for k, v in sorted(tally.items(), key=lambda kv: -kv[1])])
            members = [c[i] for t, c in six if t[i] == ch]
            conf.append(float(np.mean(members)) * len(members) / len(six))
        final = "".join(text)
        agree = sum(1 for t, _ in six if t == final) / len(decoded)
        return RowReading(final, float(np.mean(conf)), conf, agree, len(decoded), raw, alts)
