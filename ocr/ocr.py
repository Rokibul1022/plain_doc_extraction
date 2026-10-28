"""Plate text recognition with confidence, multi-variant voting and format checking.

The default backend is `fast-plate-ocr` (MIT, ONNX, Latin alphabet + digits). Its input is a
rectified plate crop. Reading several differently-restored variants of the same plate and voting
is more robust than trusting any single enhancement.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cct-xs-v1-global-model"

# Generic shapes of plates around the world: 4-9 letters/digits. Region packs can extend this.
FORMATS: dict[str, re.Pattern[str]] = {
    "generic": re.compile(r"^[A-Z0-9]{4,9}$"),
    "us": re.compile(r"^[A-Z0-9]{5,8}$"),
    "eu": re.compile(r"^[A-Z0-9]{4,9}$"),
}


@dataclass
class PlateReading:
    text: str = ""
    confidence: float = 0.0
    char_confidence: list[float] = field(default_factory=list)
    model: str = ""
    format_ok: bool = False
    votes: int = 1
    variants: int = 1
    reliability: float | None = None
    script: str = "latin"  # "latin" or "bengali"
    top: str = ""  # city/district + class letter (Bengali plates only)
    alternatives: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "char_confidence": [round(float(c), 3) for c in self.char_confidence],
            "model": self.model,
            "format_ok": self.format_ok,
            "votes": self.votes,
            "variants": self.variants,
            "script": self.script,
            "top": self.top,
            "reliability": None if self.reliability is None else round(self.reliability, 3),
            "alternatives": [(t, round(float(c), 3)) for t, c in self.alternatives],
        }


def check_format(text: str, region: str = "generic") -> bool:
    pat = FORMATS.get(region, FORMATS["generic"])
    return bool(pat.match(text))


class PlateOCR:
    """Lazy wrapper around the ONNX recogniser; `available` is False if it cannot be loaded."""

    def __init__(self, model: str = DEFAULT_MODEL, region: str = "generic") -> None:
        self.model_name = model
        self.region = region
        self._rec = None
        self._tried = False

    @property
    def available(self) -> bool:
        self._load()
        return self._rec is not None

    def _load(self) -> None:
        if self._tried:
            return
        self._tried = True
        try:
            from fast_plate_ocr import LicensePlateRecognizer

            from ocr.ortutil import session_options

            self._rec = LicensePlateRecognizer(self.model_name, device="cpu", sess_options=session_options())
        except Exception as exc:  # missing dependency, offline model download, ...
            logger.warning("Plate OCR unavailable: %s", exc)
            self._rec = None

    def read(self, img: np.ndarray) -> PlateReading:
        """Read one rectified plate crop (BGR or grey)."""
        self._load()
        if self._rec is None or img is None or img.size == 0:
            return PlateReading(model="unavailable")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        pred = self._rec.run(img, return_confidence=True)[0]
        text = pred.plate or ""
        probs = [] if pred.char_probs is None else [float(p) for p in np.asarray(pred.char_probs)[: len(text)]]
        conf = float(np.mean(probs)) if probs else 0.0
        return PlateReading(
            text=text,
            confidence=conf,
            char_confidence=probs,
            model=self.model_name,
            format_ok=check_format(text, self.region),
        )

    def read_many(self, imgs: list[np.ndarray]) -> list[PlateReading]:
        """Batch-read several crops (one ONNX call; much faster than looping `read`)."""
        self._load()
        if self._rec is None or not imgs:
            return [PlateReading(model="unavailable") for _ in imgs]
        batch = [cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) if i.ndim == 2 else i for i in imgs]
        out = []
        for pred in self._rec.run(batch, return_confidence=True):
            text = pred.plate or ""
            probs = [] if pred.char_probs is None else [float(p) for p in np.asarray(pred.char_probs)[: len(text)]]
            out.append(PlateReading(
                text=text,
                confidence=float(np.mean(probs)) if probs else 0.0,
                char_confidence=probs,
                model=self.model_name,
                format_ok=check_format(text, self.region),
            ))
        return out

    def read_variants(self, variants: list[np.ndarray]) -> PlateReading:
        """Read several versions of one plate and vote (agreement raises confidence)."""
        readings = [r for r in (self.read(v) for v in variants) if r.text]
        if not readings:
            return PlateReading(model=self.model_name)
        return vote(readings, self.region)


def vote(readings: list[PlateReading], region: str = "generic") -> PlateReading:
    """Combine readings: group identical strings, score by summed confidence, then repair
    individual characters by confidence-weighted vote among same-length candidates."""
    groups: dict[str, list[PlateReading]] = defaultdict(list)
    for r in readings:
        groups[r.text].append(r)
    scored = sorted(
        ((sum(x.confidence for x in g), t) for t, g in groups.items()), reverse=True
    )
    best_text = scored[0][1]
    same_len = [r for r in readings if len(r.text) == len(best_text)]

    chars, char_conf = [], []
    for i in range(len(best_text)):
        tally: dict[str, float] = defaultdict(float)
        for r in same_len:
            c = r.char_confidence[i] if i < len(r.char_confidence) else r.confidence
            tally[r.text[i]] += c
        ch = max(tally, key=tally.get)
        chars.append(ch)
        char_conf.append(tally[ch] / sum(tally.values()) * max(r.char_confidence[i] if i < len(r.char_confidence) else r.confidence
                                                              for r in same_len if r.text[i] == ch))
    text = "".join(chars)
    agree = sum(1 for r in readings if r.text == text)
    mean_conf = float(np.mean(char_conf)) if char_conf else 0.0
    # agreement among variants is evidence; a lone reading keeps its own confidence
    conf = mean_conf * (0.75 + 0.25 * agree / len(readings))
    alts = [(t, s / len(groups[t])) for s, t in scored if t != text][:3]
    return PlateReading(
        text=text,
        confidence=float(conf),
        char_confidence=[float(c) for c in char_conf],
        model=readings[0].model,
        format_ok=check_format(text, region),
        votes=agree,
        variants=len(readings),
        alternatives=alts,
    )
