"""Typeset recognised plate text as a clean page (needs the OCR route font and HarfBuzz)."""

import numpy as np


def render_typed(lines: list[str], *, out_width: int = 960) -> np.ndarray:
    """Typeset recognised plate text as a clean page: black text on pure white, correctly shaped.

    Used when the reading is reliable, so the page is perfectly crisp. The faithful scan produced
    by `make_document` remains the fallback and the way to check the reading against the plate."""
    from ocr import textrender

    lines = [ln for ln in lines if ln]
    if not lines:
        return np.full((int(out_width * 0.3), out_width), 255, np.uint8)
    em = int(out_width * 0.16)
    rows = [textrender.render(ln, em) for ln in lines]
    widest = max(r.shape[1] for r in rows)
    if widest > 0.86 * out_width:  # shrink to fit
        em = int(em * 0.86 * out_width / widest)
        rows = [textrender.render(ln, em) for ln in lines]
    gap = int(em * 0.25)
    height = sum(r.shape[0] for r in rows) + gap * (len(rows) - 1) + int(em * 0.5)
    page = np.full((height, out_width), 255, np.uint8)
    y = int(em * 0.25)
    for r in rows:
        x = (out_width - r.shape[1]) // 2
        page[y: y + r.shape[0], x: x + r.shape[1]] = np.minimum(page[y: y + r.shape[0], x: x + r.shape[1]], r)
        y += r.shape[0] + gap
    return page
