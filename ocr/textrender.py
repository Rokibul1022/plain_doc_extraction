"""Render Bengali (or any complex-script) text correctly: HarfBuzz shaping + outline rasterising.

Pillow builds without libraqm cannot reorder/join Bengali vowel signs and conjuncts, so words such as
"মেট্রো" come out scrambled. This shapes with HarfBuzz and fills the glyph outlines directly, so it
works wherever uharfbuzz and fontTools are installed and a Bengali font is available.
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

_FONT_CANDIDATES = [
    (Path(__file__).with_name("fonts") / "NotoSansBengali-Bold.ttf", 0),
    (Path("C:/Windows/Fonts/Nirmala.ttc"), 1),  # Nirmala UI Bold
    (Path("/usr/share/fonts/truetype/noto/NotoSansBengali-Bold.ttf"), 0),
    (Path("/usr/share/fonts/truetype/lohit-bengali/Lohit-Bengali.ttf"), 0),
]


def find_font() -> tuple[Path, int] | None:
    for path, index in _FONT_CANDIDATES:
        if path.exists():
            return path, index
    return None


@lru_cache(maxsize=1)
def _hb_font():
    import uharfbuzz as hb

    found = find_font()
    if found is None:
        return None
    path, index = found
    return hb.Font(hb.Face(hb.Blob.from_file_path(str(path)), index))


def available() -> bool:
    try:
        return _hb_font() is not None
    except Exception:
        return False


def _contours(font, gid: int, scale: float, ox: float, oy: float) -> list[np.ndarray]:
    from fontTools.pens.basePen import BasePen

    class Pen(BasePen):
        def __init__(self) -> None:
            super().__init__(None)
            self.polys: list[list[tuple[float, float]]] = []
            self.cur: list[tuple[float, float]] = []

        def _moveTo(self, p):
            self.cur = [p]

        def _lineTo(self, p):
            self.cur.append(p)

        def _curveToOne(self, p1, p2, p3):
            p0 = self.cur[-1]
            for t in np.linspace(0, 1, 9)[1:]:
                u = 1 - t
                self.cur.append((u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                                 u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]))

        def _qCurveToOne(self, p1, p2):
            p0 = self.cur[-1]
            for t in np.linspace(0, 1, 7)[1:]:
                u = 1 - t
                self.cur.append((u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
                                 u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]))

        def _closePath(self):
            if len(self.cur) > 2:
                self.polys.append(self.cur)
            self.cur = []

        _endPath = _closePath

    pen = Pen()
    font.draw_glyph_with_pen(gid, pen)
    return [np.array([(x * scale + ox, -y * scale + oy) for x, y in poly], np.float32) for poly in pen.polys]


def render(text: str, height_px: int, *, supersample: int = 3) -> np.ndarray:
    """Black-on-white uint8 image of `text`, `height_px` being the em height. Raises if unavailable."""
    import uharfbuzz as hb

    font = _hb_font()
    if font is None:
        raise RuntimeError("no Bengali font found")
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf, {"kern": True, "liga": True})
    scale = height_px * supersample / font.face.upem
    pen_x, polys = 0.0, []
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        polys += _contours(font, info.codepoint, scale, pen_x + pos.x_offset * scale, height_px * supersample * 0.85 - pos.y_offset * scale)
        pen_x += pos.x_advance * scale
    width = int(pen_x + height_px * supersample * 0.2)
    h = int(height_px * supersample * 1.2)
    mask = np.zeros((h, max(width, 8)), np.uint8)
    for poly in polys:  # even-odd fill: XOR each contour so counters (holes) stay open
        layer = np.zeros_like(mask)
        cv2.fillPoly(layer, [np.round(poly).astype(np.int32)], 1)
        mask ^= layer
    img = 255 - cv2.resize(mask * 255, (mask.shape[1] // supersample, mask.shape[0] // supersample), interpolation=cv2.INTER_AREA)
    return img
