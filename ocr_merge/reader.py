"""Run-time side of the merged reader: one ONNX session, one call per plate (no torch).

All the views the four separate readers looked at (6 digit-row views x 2 crops, and the top row of both crops, each in
the ink look and the relief look) go into a single batched call; the outputs are split and decoded with the same
voting code as `ocr.bangla_reader` / `ocr.bangla_top`, so the rest of the pipeline is unchanged.
"""

import logging
import os
import threading
from pathlib import Path

import numpy as np

from ocr.bangla_reader import VIEWS, region, vote
from ocr.bangla_top import from_logits
from ocr.pipeline import Candidate
from ocr.synth_row import CITY_WORDS, LETTERS, to_top_region
from opencv.relief import relief_view

logger = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).with_name("plate_net.onnx")
LOOKS = {"ink": None, "relief": relief_view}
CHUNK = int(os.environ.get("MERGE_CHUNK", "48"))  # digit-row views per model call (memory vs. call overhead)


def _avg_softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True)).mean(0)


class MergedReader:
    def __init__(self, path: Path = MODEL_PATH) -> None:
        self.path = Path(path)
        self._sess = None
        self._tried = False
        self.views = VIEWS  # digit-row views per crop: (top fraction, left/right trim); more views = a steadier vote but a bigger batch
        self.priors = None  # ocr_merge.prior.Priors: decide the city word and class letter with class priors (None = the model alone)
        self.a_city, self.a_letter = 0.5, 1.0
        self.decode = None  # e.g. ocr_merge.decode.decode6 (always six digits per view); None = greedy CTC like the separate readers
        self._lock = threading.Lock()  # two first requests at once must not both see a half-loaded reader

    @property
    def device(self) -> str:
        self._load()
        return getattr(self._sess, "name", "cpu (ONNX Runtime)") if self._sess is not None else "unavailable"

    @property
    def available(self) -> bool:
        self._load()
        return self._sess is not None

    def _load(self) -> None:
        with self._lock:
            if self._tried:
                return
            try:
                import onnxruntime as ort
                from ocr.ortutil import session_options

                self._sess = ort.InferenceSession(str(self.path), sess_options=session_options(), providers=["CPUExecutionProvider"])
            except Exception as exc:
                logger.warning("merged plate reader unavailable (%s): %s", self.path.name, exc)
            self._tried = True

    def candidates(self, crops: list[np.ndarray], views=None) -> dict:
        """{"ink": Candidate | None, "relief": Candidate | None} for the crops of one plate (rectified, restored); `views` overrides self.views."""
        views = views or self.views
        self._load()
        crops = [c for c in crops if c is not None and c.size]
        if self._sess is None or not crops:
            return {m: None for m in LOOKS}
        rows, tops = [], []
        for view in LOOKS.values():
            looked = [view(c) if view else c for c in crops]
            rows += [region(c, top, trim) for c in looked for top, trim in views]
            tops += [to_top_region(c) for c in looked]
        rows_in = np.stack(rows).astype(np.float32)[:, None] / 255.0
        tops_in = np.stack(tops).astype(np.float32)[:, None] / 255.0
        parts = []  # digit-row views go through in chunks: the network's activations scale with the batch (216 views at once = ~400 MB)
        for i in range(0, len(rows_in), CHUNK):
            lp, c, l = self._sess.run(None, {"row": rows_in[i:i + CHUNK], "top": tops_in if i == 0 else tops_in[:1]})
            parts.append(lp)
            if i == 0:
                city, letter = c, l
        logp = np.concatenate(parts)
        probs = np.exp(logp)
        nr, nt = len(crops) * len(views), len(crops)
        out = {}
        for k, mode in enumerate(LOOKS):
            r = vote(probs[k * nr:(k + 1) * nr], self.decode)
            t = from_logits(city[k * nt:(k + 1) * nt], letter[k * nt:(k + 1) * nt], 0.0, 0.0)
            c = Candidate(mode)
            if r.ok:
                c.digits, c.conf, c.agreement, c.alts = r.text, list(r.char_confidence), r.agreement, r.alts
            c.city, c.city_conf, c.letter, c.letter_conf = t.city, t.city_conf, t.letter, t.letter_conf
            c.city_probs, c.letter_probs = _avg_softmax(city[k * nt:(k + 1) * nt]), _avg_softmax(letter[k * nt:(k + 1) * nt])  # for re-deciding
            if self.priors is not None:
                ci, cc, li, lc = self.priors.decide(c.city_probs, c.letter_probs, self.a_city, self.a_letter)
                c.city, c.city_conf, c.letter, c.letter_conf = CITY_WORDS[ci], cc, LETTERS[li], lc
            out[mode] = c if c.digits or c.city else None
        return out
