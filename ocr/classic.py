"""Classical (no neural network) Bengali plate-number reader.

1. `rowprep.row_roi` cuts the number row out as black ink on white.
2. `split_digits` finds the hyphen/dot, then cuts the six digits at an equal pitch around it (the plate font
   is monospaced); a class letter in front of the number is skipped automatically.
3. Each digit becomes a shape feature (HOG + a coarse ink map) and is matched to reference digits.

Reference digits come from Bengali fonts and, when available, from verified plates.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.feature import hog

from ocr.digits import DIGITS
from ocr.rowprep import row_roi

CELL = 28


@dataclass
class DigitCell:
    ink: np.ndarray  # CELL x CELL float32, ink = 1
    box: tuple[int, int, int, int]
    fill: float


@dataclass
class Segmentation:
    cells: list[DigitCell]
    sep: tuple[int, int, int, int] | None
    pitch: float
    row: np.ndarray  # the row ink (0..1) the cells were cut from
    note: str = ""

    @property
    def ok(self) -> bool:
        return len(self.cells) == 6 and all(c.fill > 0.04 for c in self.cells)


def _normalise_cell(ink: np.ndarray) -> DigitCell | None:
    """Tight-crop a digit's ink, scale it to fit CELL x CELL keeping its shape, centre it on its ink mass."""
    ys, xs = np.where(ink > 0.5)
    if len(ys) < 12:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch = ink[y0:y1, x0:x1]
    h, w = patch.shape
    s = (CELL - 4) / max(h, w)
    patch = cv2.resize(patch, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((CELL, CELL), np.float32)
    ph, pw = patch.shape
    # centre of mass at the cell centre keeps strokes comparable across slightly different crops
    m = patch.sum()
    cy = (patch.sum(axis=1) * np.arange(ph)).sum() / max(m, 1e-6)
    cx = (patch.sum(axis=0) * np.arange(pw)).sum() / max(m, 1e-6)
    oy, ox = int(round(CELL / 2 - cy)), int(round(CELL / 2 - cx))
    oy, ox = max(0, min(CELL - ph, oy)), max(0, min(CELL - pw, ox))
    canvas[oy: oy + ph, ox: ox + pw] = patch
    return DigitCell(canvas, (int(x0), int(y0), int(x1), int(y1)), float((ink > 0.5).mean()))


def split_digits(roi: np.ndarray) -> Segmentation | None:
    """Cut the six digits out of a number-row crop (uint8, ink dark on white)."""
    ink = 1.0 - roi.astype(np.float32) / 255.0
    b = (ink > 0.5).astype(np.uint8)
    h, w = b.shape
    if b.sum() < 40 or h < 12:
        return None
    n, lab, st, cen = cv2.connectedComponentsWithStats(b, connectivity=8)
    comps = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 0.0006 * h * w]
    if len(comps) < 3:
        return None
    # digit-sized pieces define the row: tall relative to the typical piece
    areas = np.array([st[i, cv2.CC_STAT_AREA] for i in comps])
    heights = np.array([st[i, cv2.CC_STAT_HEIGHT] for i in comps])
    big = heights[areas >= np.median(areas)]
    dh = float(np.median(big)) if len(big) else float(np.median(heights))
    tall = [i for i in comps if st[i, cv2.CC_STAT_HEIGHT] >= 0.55 * dh and st[i, cv2.CC_STAT_HEIGHT] <= 1.6 * dh]
    if len(tall) < 3:
        return None
    xs0 = min(st[i, cv2.CC_STAT_LEFT] for i in tall)
    xs1 = max(st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH] for i in tall)
    ys0 = min(st[i, cv2.CC_STAT_TOP] for i in tall)
    ys1 = max(st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] for i in tall)
    rh = ys1 - ys0
    span = max(xs1 - xs0, 1)

    # separator: a short mark (hyphen / dot) low-middle in the row, between the two digit groups; of several
    # candidates take the one nearest where the gap normally sits (about 38 % along the row)
    cands = []
    for i in comps:
        x, y, cw, ch = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
        cx, cy = x + cw / 2, y + ch / 2
        if ch < 0.45 * dh and 0.25 * rh < cy - ys0 < 0.85 * rh and 0.16 * span < cx - xs0 < 0.60 * span:
            cands.append((abs((cx - xs0) / span - 0.38), (int(x), int(y), int(x + cw), int(y + ch))))
    sep = min(cands)[1] if cands else None

    if sep is not None:
        pitch = (xs1 - sep[2]) / 4.0
        spans = [(sep[0] - (2 - k) * pitch, sep[0] - (1 - k) * pitch) for k in range(2)] +                 [(sep[2] + k * pitch, sep[2] + (k + 1) * pitch) for k in range(4)]
        note = "sep"
    else:
        pitch = span / 6.7
        spans = [(xs0 + k * pitch, xs0 + (k + 1) * pitch) for k in range(2)] +                 [(xs0 + 2.7 * pitch + k * pitch, xs0 + 2.7 * pitch + (k + 1) * pitch) for k in range(4)]
        note = "no-sep"

    cells: list[DigitCell] = []
    for a, bb in spans:
        # a little slack on each side because the pitch estimate is not exact; only ink inside the row counts
        xa, xb = int(max(0, a - 0.12 * pitch)), int(min(w, bb + 0.12 * pitch))
        piece = ink[:, xa:xb].copy()
        cell = _normalise_cell(piece) if xb > xa else None
        if cell is None:
            cell = DigitCell(np.zeros((CELL, CELL), np.float32), (xa, 0, xb, h), 0.0)
        cells.append(cell)
    return Segmentation(cells, sep, float(pitch), ink, note)


def features(cell: np.ndarray) -> np.ndarray:
    """Shape descriptor: HOG on the centred digit plus a coarse ink map, L2-normalised."""
    c = np.clip(cell.astype(np.float32), 0, 1)
    c = cv2.GaussianBlur(c, (0, 0), 0.8)
    hg = hog(c, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2), block_norm="L2-Hys")
    coarse = cv2.resize(c, (9, 9), interpolation=cv2.INTER_AREA).ravel()
    f = np.concatenate([hg, 0.6 * coarse]).astype(np.float32)
    return f / (np.linalg.norm(f) + 1e-6)


class TemplateBank:
    """Reference digits with their features; classifies a digit by weighted nearest neighbours."""

    def __init__(self) -> None:
        self.feats: list[np.ndarray] = []
        self.labels: list[int] = []
        self.weights: list[float] = []
        self._mat: np.ndarray | None = None

    def add(self, cell: np.ndarray, label: int, weight: float = 1.0) -> None:
        self.feats.append(features(cell))
        self.labels.append(label)
        self.weights.append(weight)
        self._mat = None

    def classify(self, cell: np.ndarray, k: int = 3) -> tuple[int, float, np.ndarray]:
        """(digit index, confidence 0..1, class scores): class score = mean of its top-k similarities."""
        if self._mat is None:
            self._mat = np.stack(self.feats)
            self._lab = np.array(self.labels)
            self._w = np.array(self.weights, np.float32)
        sims = (self._mat @ features(cell)) * self._w
        scores = np.zeros(10, np.float32)
        for d in range(10):
            s = np.sort(sims[self._lab == d])[::-1][:k]
            scores[d] = float(s.mean()) if len(s) else 0.0
        order = np.argsort(scores)[::-1]
        best, second = scores[order[0]], scores[order[1]]
        conf = float(np.clip((best - second) / max(best, 1e-6) * 4.0, 0.0, 1.0)) * float(np.clip(best, 0, 1))
        return int(order[0]), conf, scores


@dataclass
class ClassicReading:
    text: str = ""
    confidence: float = 0.0
    digit_conf: list[float] = field(default_factory=list)
    ok: bool = False


def read_cells(seg: Segmentation, bank: TemplateBank) -> ClassicReading:
    digs, confs = [], []
    for c in seg.cells:
        d, conf, _ = bank.classify(c.ink)
        digs.append(DIGITS[d])
        confs.append(conf)
    ok = seg.ok
    return ClassicReading("".join(digs), float(np.mean(confs)) if confs else 0.0, confs, ok)


def font_templates(bank: TemplateBank, n_variants: int = 6, seed: int = 0) -> int:
    """Add reference digits rendered from the Bengali fonts, with mild variations of stroke and blur."""
    from ocr.synth_row import _FONT_DIR, _FONT_SPECS, FONT_PATH
    from pathlib import Path
    from PIL import Image, ImageDraw, ImageFont

    rng = np.random.default_rng(seed)
    faces = [(str(_FONT_DIR / name), idx) for name, idx, _ in _FONT_SPECS if (_FONT_DIR / name).exists()]
    faces += [(FONT_PATH, i) for i in (0, 1, 2, 3, 4) if Path(FONT_PATH).exists()]
    added = 0
    for path, idx in faces:
        font = ImageFont.truetype(path, 120, index=idx)
        for d, ch in enumerate(DIGITS):
            for v in range(n_variants):
                im = Image.new("L", (200, 200), 0)
                ImageDraw.Draw(im).text((100, 100), ch, font=font, fill=255, anchor="mm")
                a = np.asarray(im, np.float32) / 255.0
                k = int(rng.integers(-2, 4))
                if k > 0:
                    a = cv2.dilate(a, np.ones((k, k), np.uint8))
                elif k < 0:
                    a = cv2.erode(a, np.ones((-k, -k), np.uint8))
                m = cv2.getRotationMatrix2D((100, 100), rng.uniform(-5, 5), rng.uniform(0.9, 1.1))
                m[0, 1] += rng.uniform(-0.15, 0.15)
                a = cv2.warpAffine(a, m, (200, 200))
                a = cv2.GaussianBlur(a, (0, 0), rng.uniform(1.0, 4.0))
                cell = _normalise_cell((a > 0.45).astype(np.float32))
                if cell is not None:
                    bank.add(cell.ink, d, 1.0)
                    added += 1
    return added
