"""Plate reading pipeline: YOLO finds the plate -> OpenCV scan (corners, straighten, restore) -> readers -> checked result.

    report = read_plate(cv2.imread("car.jpg"))
    report.text        # "ঢাকা মেট্রো  ১৪-১০০৬"  (the best reading of every digit; the verdict says how far to trust it)
    report.verdict     # "readable" | "uncertain" | "unreadable"
    cv2.imwrite("result.png", render_report(report))

Two readers look at every plate and the better one wins:
  ink     the shipped readers, on the straightened colour plate (plates with dark ink)
  relief  readers trained on embossed plates (characters stamped in the plate colour, no ink), looking at the plate's
          relief outlines (`opencv.relief.relief_view`)
Plates follow `[city]-[class letter]  NN-NNNN`. Every digit shows the best-supported reading of both readers (no confidence
numbers are shown); the verdict (readable / uncertain / unreadable) says how far to trust it. The thresholds are heuristics:
no labelled set of embossed plates exists to calibrate them on.
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ocr.bangla_reader import BanglaRowReader
from ocr.bangla_top import BanglaTopReader
from ocr.digits import DIGITS
from opencv.document import make_document
from opencv.relief import relief_page, relief_view
from opencv.scan import scan as cv_scan

HERE = Path(__file__).parent
LETTER_SHOW = 0.60  # the class letter (the weakest head) is printed only when it is this sure; the plate picture shows it anyway
READABLE_MIN = 0.80  # every digit at least this confident (and the views agree) -> "readable"
UNCERTAIN_MIN = 0.55  # mean digit confidence at least this -> "uncertain", else "unreadable"
DIGIT_MIN = 0.55  # a digit this sure counts towards "uncertain"
PAGE_H = 128  # the plain-page step is tuned for plate crops about this high; bigger ones are only slower, not better
TOP_RULE = "ink_first"  # which reader names the top-row words (see read_plate)
TIGHT_CROP_MAX = 720  # images this small (longest side) are treated as a plate crop, larger ones go through YOLO


@dataclass
class Candidate:
    mode: str
    digits: str = ""
    conf: list[float] = field(default_factory=list)
    agreement: float = 0.0
    alts: list = field(default_factory=list)
    city: str = ""
    city_conf: float = 0.0
    letter: str = ""
    letter_conf: float = 0.0

    @property
    def score(self) -> float:
        return float(np.mean(self.conf)) * (0.5 + 0.5 * self.agreement) if self.conf else 0.0


@dataclass
class PlateReport:
    mode: str  # "ink" | "relief"
    digits: str  # six Bengali digits: the best-supported reading of every position
    digit_conf: list[float]
    top: str  # city word ("" if no reader named one)
    letter: str  # class letter ("" unless confident)
    top_conf: float
    letter_conf: float
    verdict: str
    format_ok: bool
    corners: np.ndarray
    corner_method: str
    page: np.ndarray  # the cleaned plate (plain page for ink plates, relief outlines for embossed ones)
    rectified: np.ndarray
    restored: np.ndarray
    candidates: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        head = f"{self.top}-{self.letter}" if self.top and self.letter else self.top
        return f"{head}  {self.digits[:2]}-{self.digits[2:]}".strip()


_readers: dict = {}
MODEL_PATHS: dict[str, Path] = {}  # optional overrides, e.g. {"ink_row": Path(...)}; call `use_models` so cached readers reset


def use_models(**paths) -> None:
    """Point named readers (ink_row, ink_top, relief_row, relief_top) at other ONNX files (used to evaluate held-out models)."""
    MODEL_PATHS.update({k: Path(v) for k, v in paths.items()})
    for k in paths:
        _readers.pop(k, None)


def _reader(name: str):
    if name not in _readers:
        if name == "ink_row":
            _readers[name] = BanglaRowReader(MODEL_PATHS.get(name, HERE / "bangla_row.onnx"))
        elif name == "relief_row":
            _readers[name] = BanglaRowReader(MODEL_PATHS.get(name, HERE / "bangla_row_relief.onnx"), view=relief_view)
        elif name == "ink_top":
            _readers[name] = BanglaTopReader(MODEL_PATHS.get(name, HERE / "bangla_top.onnx"))
        elif name == "relief_top":
            _readers[name] = BanglaTopReader(MODEL_PATHS.get(name, HERE / "bangla_top_relief.onnx"), view=relief_view)
        elif name == "detector":
            from ocr.detector import LearnedDetector

            _readers[name] = LearnedDetector()
    return _readers[name]


def locate_plate(img: np.ndarray, margin: float = 0.08) -> np.ndarray:
    """A crop that holds the plate: the image itself if it already is a crop, else the best YOLO box plus a margin."""
    h, w = img.shape[:2]
    if max(h, w) <= TIGHT_CROP_MAX and 0.55 <= w / h <= 3.5:
        return img
    det = _reader("detector")
    boxes = det.detect(img) if det.available else []
    if not boxes:
        return img
    (x1, y1, x2, y2), _ = max(boxes, key=lambda b: b[1])
    mx, my = int(margin * (x2 - x1)), int(margin * (y2 - y1) * 1.6)
    return img[max(0, y1 - my): min(h, y2 + my), max(0, x1 - mx): min(w, x2 + mx)]


def _page_input(restored: np.ndarray) -> np.ndarray:
    if restored.shape[0] <= PAGE_H * 1.25:
        return restored
    s = PAGE_H / restored.shape[0]
    return cv2.resize(restored, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def _candidate(mode: str, crops: list[np.ndarray]) -> Candidate | None:
    row, top = _reader(f"{mode}_row"), _reader(f"{mode}_top")
    c = Candidate(mode)
    if row.available:
        r = row.read(crops)
        if r.ok:
            c.digits, c.conf, c.agreement, c.alts = r.text, list(r.char_confidence), r.agreement, r.alts
    if top.available:
        t = top.read(crops, min_conf=0.0)
        c.city, c.city_conf, c.letter, c.letter_conf = t.city, t.city_conf, t.letter, t.letter_conf
    return c if c.digits or c.city else None


def _verdict(digits: str, conf: list[float], agreement: float) -> str:
    if len(digits) != 6 or any(d not in DIGITS for d in digits.replace("?", "")) or not conf:
        return "unreadable"
    sure = [c >= DIGIT_MIN for c in conf]
    if all(c >= READABLE_MIN for c in conf) and agreement >= 0.6:
        return "readable"
    if np.mean(conf) >= UNCERTAIN_MIN and sum(sure) >= 4:
        return "uncertain"
    return "unreadable"


def read_plate(img: np.ndarray, force_mode: str | None = None) -> PlateReport:
    """Run the whole pipeline on a frame or a plate crop."""
    crop = locate_plate(img)
    r = cv_scan(crop, relief="off", make_page=False)  # corners, straighten, restore (OpenCV only); the page comes later
    views = [r.rectified, r.restored]
    return build_report(r, {m: _candidate(m, views) for m in ("ink", "relief")}, force_mode)


def build_report(r, cands: dict, force_mode: str | None = None) -> PlateReport:
    """Pick the reading from the ink and relief candidates of one scanned plate (shared with ocr_merge)."""
    avail = {m: c for m, c in cands.items() if c is not None}
    if force_mode in avail:
        best = avail[force_mode]
    elif avail:
        best = max(avail.values(), key=lambda c: (c.score, c.mode == "ink"))
    else:
        best = Candidate("ink")
    # top row. "ink_first" (default): the ink reader, which was tuned on real normal plates, names the words; the relief
    # reader (embossed plates) takes over only when the ink reader is unsure and the relief reader is surer.
    # "winner": the reader of the mode that read the digits better. "relief_first": the relief reader unless very unsure.
    rel, ink = avail.get("relief"), avail.get("ink")
    if TOP_RULE == "relief_first":
        pick = rel if rel and rel.city and (rel.city_conf >= 0.35 or not (ink and ink.city_conf >= 0.5)) else (ink if ink and ink.city else None)
    elif TOP_RULE == "winner":
        pick = best if best.city else next((c for c in avail.values() if c.city), None)
    else:
        pick = ink if ink and ink.city else rel
        if ink and rel and ink.city and rel.city and ink.city_conf < 0.5 and rel.city_conf > ink.city_conf:
            pick = rel
    city, city_conf = (pick.city, pick.city_conf) if pick else ("", 0.0)
    letter, letter_conf = (pick.letter, pick.letter_conf) if pick else ("", 0.0)
    # digits: every position takes the best-supported digit over both readers' views (confidence-weighted vote)
    digits = ""
    for k in range(6):
        tally: dict[str, float] = {}
        for c in avail.values():
            if c.alts and k < len(c.alts):
                for ch, share in c.alts[k]:
                    tally[ch] = tally.get(ch, 0.0) + share * c.conf[k]
        digits += max(tally, key=tally.get) if tally else "?"
    verdict = _verdict(best.digits, best.conf, best.agreement)
    if verdict == "readable" and digits != best.digits:
        verdict = "uncertain"  # the two readers disagree about a digit
    page = relief_page(r.restored) if best.mode == "relief" else make_document(_page_input(r.restored))
    return PlateReport(
        mode=best.mode, digits=digits, digit_conf=best.conf or [0.0] * 6,
        top=city, letter=letter if letter_conf >= LETTER_SHOW else "",
        top_conf=city_conf, letter_conf=letter_conf, verdict=verdict, format_ok=len(best.digits) == 6,
        corners=r.corners, corner_method=r.method, page=page, rectified=r.rectified, restored=r.restored,
        candidates={m: {"digits": c.digits, "score": round(c.score, 3), "agreement": round(c.agreement, 2),
                        "city": c.city, "city_conf": round(c.city_conf, 2)} for m, c in avail.items()})


# ---------------------------------------------------------------- the result picture
VERDICT_COLOUR = {"readable": (80, 190, 80), "uncertain": (0, 170, 255), "unreadable": (70, 70, 230)}  # BGR


def _bengali(text: str, height: int) -> np.ndarray:
    """Dark-on-white mask of `text`, correctly shaped (HarfBuzz); a blank if the font stack is missing."""
    try:
        from ocr import textrender

        return textrender.render(text, height)
    except Exception:
        return np.full((height, max(8, height // 2 * len(text))), 255, np.uint8)


def render_report(rep: PlateReport, width: int = 1000) -> np.ndarray:
    """One picture: the cleaned plate on the left, the reading (top row + number) and the verdict on the right."""
    left_w, right_w = int(width * 0.58), int(width * 0.42)
    page = rep.page if rep.page.ndim == 3 else cv2.cvtColor(rep.page, cv2.COLOR_GRAY2BGR)
    s = left_w / page.shape[1]
    page = cv2.resize(page, (left_w, max(40, int(page.shape[0] * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    height = max(page.shape[0], 300)
    canvas = np.full((height, width, 3), 255, np.uint8)
    canvas[(height - page.shape[0]) // 2:(height - page.shape[0]) // 2 + page.shape[0], :left_w] = page
    x0, room = left_w + 20, right_w - 30

    def put(mask: np.ndarray, x: int, y: int) -> None:
        m = mask[:, :room]
        roi = canvas[y: y + m.shape[0], x: x + m.shape[1]]
        roi[:] = np.minimum(roi, cv2.cvtColor(m[: roi.shape[0], : roi.shape[1]], cv2.COLOR_GRAY2BGR))

    cv2.rectangle(canvas, (x0, 14), (width - 14, 52), VERDICT_COLOUR[rep.verdict], -1)
    cv2.putText(canvas, rep.verdict.upper(), (x0 + 12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    top = f"{rep.top}-{rep.letter}" if rep.top and rep.letter else rep.top
    if top:
        put(_bengali(top, 54), x0, 76)
    number = f"{rep.digits[:2]}-{rep.digits[2:]}"
    put(_bengali(number, 84), x0, 160)
    return canvas
