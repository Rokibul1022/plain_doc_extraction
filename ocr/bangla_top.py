"""Bengali plate top-row reader: city/district word + class letter (two-head CNN, ONNX)."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ocr.synth_row import CITY_WORDS, LETTERS, to_top_region

logger = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).with_name("bangla_top.onnx")
# views: crop at slightly different heights so one bad framing does not decide the answer
HEIGHTS = (0.0, 0.03, -0.03)
# Printed only when reliable. Measured on 102 verified plates the model never trained on: the city word is
# 92 % correct at confidence >= 0.95 (shown for 75 % of plates); the class letter never exceeds ~85 %, so it
# is not printed by default (set LETTER_MIN <= 1 to enable). A wrong city on a clean page is worse than none.
CITY_MIN = 0.95
LETTER_MIN = 1.01


@dataclass
class TopReading:
    city: str = ""
    letter: str = ""
    city_conf: float = 0.0
    letter_conf: float = 0.0

    @property
    def text(self) -> str:
        return f"{self.city}-{self.letter}" if self.city and self.letter else self.city

    @property
    def confidence(self) -> float:
        return min(self.city_conf, self.letter_conf) if self.letter else self.city_conf


class BanglaTopReader:
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
            logger.warning("Bengali top-row reader unavailable (%s): %s", self.path.name, exc)

    def read(self, crops: np.ndarray | list[np.ndarray], min_conf: float | None = None) -> TopReading:
        """`min_conf=None` applies the reliability cut-offs; `0.0` returns the raw best guess (for scoring)."""
        city_min = CITY_MIN if min_conf is None else min_conf
        letter_min = LETTER_MIN if min_conf is None else min_conf
        self._load()
        crops = crops if isinstance(crops, list) else [crops]
        crops = [c for c in crops if c is not None and c.size]
        if self._sess is None or not crops:
            return TopReading()
        if self.view is not None:
            crops = [self.view(c) for c in crops]
        x = np.stack([to_top_region(c) for c in crops]).astype(np.float32)[:, None] / 255.0
        city_logits, letter_logits = self._sess.run(None, {"x": x})

        def avg_softmax(z):
            e = np.exp(z - z.max(1, keepdims=True))
            return (e / e.sum(1, keepdims=True)).mean(0)

        pc, pl = avg_softmax(city_logits), avg_softmax(letter_logits)
        ci, li = int(pc.argmax()), int(pl.argmax())
        r = TopReading(CITY_WORDS[ci], LETTERS[li], float(pc[ci]), float(pl[li]))
        if r.city_conf < city_min:
            return TopReading()
        if r.letter_conf < letter_min:
            r.letter, r.letter_conf = "", 0.0  # city word is reliable, the single letter is not
        return r
