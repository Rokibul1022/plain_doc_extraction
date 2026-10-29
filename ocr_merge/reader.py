"""Run-time side of the merged reader: one ONNX session, one call per plate (no torch).

All the views the four separate readers looked at (6 digit-row views x 2 crops, and the top row of both crops, each in
the ink look and the relief look) go into a single batched call; the outputs are split and decoded with the same
voting code as `ocr.bangla_reader` / `ocr.bangla_top`, so the rest of the pipeline is unchanged.
"""

import logging
from pathlib import Path

import numpy as np

from ocr.bangla_reader import VIEWS, region, vote
from ocr.bangla_top import from_logits
from ocr.pipeline import Candidate
from ocr.synth_row import to_top_region
from opencv.relief import relief_view

logger = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).with_name("plate_net.onnx")
LOOKS = {"ink": None, "relief": relief_view}


class MergedReader:
    def __init__(self, path: Path = MODEL_PATH) -> None:
        self.path = Path(path)
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
            logger.warning("merged plate reader unavailable (%s): %s", self.path.name, exc)

    def candidates(self, crops: list[np.ndarray]) -> dict:
        """{"ink": Candidate | None, "relief": Candidate | None} for the crops of one plate (rectified, restored)."""
        self._load()
        crops = [c for c in crops if c is not None and c.size]
        if self._sess is None or not crops:
            return {m: None for m in LOOKS}
        rows, tops = [], []
        for view in LOOKS.values():
            looked = [view(c) if view else c for c in crops]
            rows += [region(c, top, trim) for c in looked for top, trim in VIEWS]
            tops += [to_top_region(c) for c in looked]
        logp, city, letter = self._sess.run(None, {"row": np.stack(rows).astype(np.float32)[:, None] / 255.0,
                                                   "top": np.stack(tops).astype(np.float32)[:, None] / 255.0})
        probs = np.exp(logp)
        nr, nt = len(crops) * len(VIEWS), len(crops)
        out = {}
        for k, mode in enumerate(LOOKS):
            r = vote(probs[k * nr:(k + 1) * nr])
            t = from_logits(city[k * nt:(k + 1) * nt], letter[k * nt:(k + 1) * nt], 0.0, 0.0)
            c = Candidate(mode)
            if r.ok:
                c.digits, c.conf, c.agreement, c.alts = r.text, list(r.char_confidence), r.agreement, r.alts
            c.city, c.city_conf, c.letter, c.letter_conf = t.city, t.city_conf, t.letter, t.letter_conf
            out[mode] = c if c.digits or c.city else None
        return out
