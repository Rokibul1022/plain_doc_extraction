"""Render a scanned plate as a clean plain document, the way a document scanner would.

The result is a flat white page that carries only the plate's characters: crisp, dark and level.
Everything that belongs to the object rather than the text - the plate frame and edges, bolt holes,
background, shadows, speckle - is removed. It is derived from the real pixels (never re-typed), so
what you read on the page is what was on the plate.
"""

import cv2
import numpy as np

from opencv.geometry import deshear, deskew
from opencv.layout import clean_ink
from opencv.readable import _smoothstep, _text_is_light, best_channel

RELIEF_COV, RELIEF_COH = 0.08, 0.30  # normal ink map weaker than this -> try the 3D relief reading
PAGE_H, PAGE_MAX_H = 128, 200  # the page step is tuned for plates ~100-130 px high; taller crops are downscaled to PAGE_H first (slower, not better)
WORK_SCALE = 3.0  # work at 3x so strokes come out smooth after thresholding


def _flatten(ch: np.ndarray) -> np.ndarray:
    """Divide out slowly varying illumination with a closing wider than a stroke."""
    h = ch.shape[0]
    k = max(9, int(0.30 * h) | 1)
    bg = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), max(2.0, k / 4.0))
    return np.clip(ch.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0) * 255.0, 0, 255)


def _soft_ink(flat: np.ndarray) -> np.ndarray:
    """0..1 ink strength with data-driven black/white points from the plate's own tones."""
    u8 = flat.astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    body, text = flat[flat > thr], flat[flat <= thr]
    if body.size < 8 or text.size < 8:
        return np.zeros_like(flat)
    body_med, text_med = float(np.median(body)), float(np.median(text))
    if body_med - text_med < 12:
        return np.zeros_like(flat)
    lo = text_med + 0.10 * (thr - text_med)
    hi = thr + 0.45 * (body_med - thr)
    return 1.0 - _smoothstep((flat - lo) / max(hi - lo, 1.0))


def _hysteresis(ink: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Pixels above `lo` that connect to a pixel above `hi`: keeps faint stroke ends, drops lone noise."""
    weak = (ink > lo).astype(np.uint8)
    strong = (ink > hi).astype(np.uint8)
    n, lab = cv2.connectedComponents(weak, connectivity=8)
    if n <= 1:
        return weak
    keep = np.zeros(n, np.uint8)
    keep[np.unique(lab[strong > 0])] = 1
    keep[0] = 0
    return keep[lab]


def _smooth_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Round off jagged outlines: blur the binary mask and re-threshold (curvature smoothing)."""
    f = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    return (f > 0.5).astype(np.uint8)


def _row_bands(mask: np.ndarray) -> list[tuple[int, int]]:
    """Vertical bands of ink rows, split at the deepest valley of the row profile (plates have 1-2 rows)."""
    h = mask.shape[0]
    prof = cv2.GaussianBlur(mask.sum(axis=1).astype(np.float32).reshape(-1, 1), (1, 0), max(1.5, h * 0.015)).ravel()
    nz = np.where(prof > 0.03 * prof.max())[0]
    if len(nz) == 0:
        return []
    top, bot = int(nz[0]), int(nz[-1]) + 1
    span = bot - top
    lo, hi = top + int(0.30 * span), top + int(0.70 * span)
    valley = lo + int(np.argmin(prof[lo:hi])) if hi > lo else (top + bot) // 2
    top_peak, bot_peak = prof[top:valley].max() if valley > top else 0, prof[valley:bot].max() if bot > valley else 0
    if min(top_peak, bot_peak) > 0 and prof[valley] < 0.45 * min(top_peak, bot_peak) and valley - top > 0.15 * span and bot - valley > 0.15 * span:
        return [(top, valley), (valley, bot)]
    return [(top, bot)]


def _straighten_row(mask: np.ndarray) -> np.ndarray:
    """Level one text row and remove its slant: same estimators the page-level deskew uses, on the row alone."""
    ys, xs = np.where(mask > 0)
    if len(ys) < 30:
        return mask
    pad = int(0.15 * (ys.max() - ys.min() + 1)) + 4
    row = mask[max(0, ys.min() - pad): ys.max() + 1 + pad, max(0, xs.min() - pad): xs.max() + 1 + pad]
    page = ((1 - row) * 255).astype(np.uint8)
    out = deshear(deskew(page, max_angle=18.0))
    return _drop_slivers((out < 128).astype(np.uint8))


def _drop_slivers(row: np.ndarray) -> np.ndarray:
    """Remove frame fragments left after levelling. They are hair-thin bars/lines sitting at the very edge
    of the row; the same shape in the middle of the row is a real glyph (e.g. the digit one) and stays."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(row, connectivity=8)
    if n <= 2:
        return row
    H, W = row.shape
    ys, xs = np.where(row > 0)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    heights = st[1:, cv2.CC_STAT_HEIGHT]
    areas = st[1:, cv2.CC_STAT_AREA]
    big = heights[areas >= np.median(areas)]
    char_h = float(np.median(big)) if len(big) else float(np.median(heights))
    ex, ey = 0.05 * (x1 - x0), 0.12 * (y1 - y0)
    keep = np.zeros_like(row)
    for i in range(1, n):
        x, y, w, h = st[i, 0], st[i, 1], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        at_side = x <= x0 + ex or x + w >= x1 - ex
        at_top_bottom = y <= y0 + ey or y + h >= y1 - ey
        if (w < 0.12 * char_h and h > 0.8 * char_h and at_side) or (h < 0.14 * char_h and w > 0.6 * char_h and at_top_bottom):
            continue
        keep[lab == i] = 1
    return keep if keep.sum() >= 30 else row


def _strip_rules(row: np.ndarray) -> np.ndarray:
    """After a text row is levelled, the plate frame's rules are full-width thin horizontal runs hugging its top or
    bottom edge. Delete those rows of pixels; digit strokes never span most of the row's width in one pixel row."""
    h, w = row.shape
    ys, xs = np.where(row > 0)
    if len(ys) < 30:
        return row
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    span, rh = x1 - x0, y1 - y0
    prof = row[:, x0:x1].sum(axis=1)
    full = prof > 0.62 * span
    out = row.copy()
    edge, thin = max(2, int(0.22 * rh)), max(3, int(0.17 * rh))
    y = y0
    while y < y1:
        if not full[y]:
            y += 1
            continue
        z = y
        while z < y1 and full[z]:
            z += 1
        # only a THIN full-width run at the edge is a rule; a thick one is the row's own joined digits
        if z - y <= thin and (y - y0 < edge or y1 - z < edge):
            out[y:z] = 0
        y = z
    # a rule leaves a thin ragged crust beside the deleted rows: remove ink that is only a few pixels thick there
    return out if out.sum() >= 30 else row


def _align_rows(mask: np.ndarray) -> np.ndarray:
    """Re-lay the ink as a plain document: every row levelled, de-slanted and centred on a common width."""
    bands = _row_bands(mask)
    if not bands:
        return mask
    rows = []
    for y0, y1 in bands:
        band = np.zeros_like(mask)
        band[y0:y1] = mask[y0:y1]
        r = _strip_rules(_straighten_row(band))
        ys, xs = np.where(r > 0)
        if len(ys) < 30:
            continue
        rows.append(r[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1])
    if not rows:
        return mask
    # characters in a row should be the same size on the page: scale each row to the common median height
    width = max(r.shape[1] for r in rows)
    gap = int(0.35 * float(np.median([r.shape[0] for r in rows])))
    height = sum(r.shape[0] for r in rows) + gap * (len(rows) - 1)
    out = np.zeros((height, width), np.uint8)
    y = 0
    for r in rows:
        x = (width - r.shape[1]) // 2
        out[y: y + r.shape[0], x: x + r.shape[1]] = r
        y += r.shape[0] + gap
    return out


def _drop_edge_fragments(binary: np.ndarray) -> np.ndarray:
    """Delete ink pieces that touch the crop border and are long/thin or very large: frame and shadow remnants.
    Real characters that merely reach the border are compact and survive."""
    h, w = binary.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return binary
    areas = st[1:, cv2.CC_STAT_AREA]
    ref = float(np.median(st[1:, cv2.CC_STAT_HEIGHT][areas >= np.median(areas)])) if len(areas) else 1.0
    out = binary.copy()
    for i in range(1, n):
        x, y, cw, ch, a = st[i]
        touches = x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1
        if not touches:
            continue
        elongated = max(cw, ch) > 1.8 * ref and min(cw, ch) < 0.5 * max(cw, ch)
        if elongated or a > 3.5 * ref * ref:
            out[lab == i] = 0
    return out


def _char_height(mask: np.ndarray) -> float:
    """Typical character height: median height of the larger ink pieces."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return 1.0
    areas, heights = st[1:, cv2.CC_STAT_AREA], st[1:, cv2.CC_STAT_HEIGHT]
    big = heights[areas >= np.median(areas)]
    return float(np.median(big)) if len(big) else float(np.median(heights))


def _drop_bars(mask: np.ndarray) -> np.ndarray:
    """Remove stand-alone bars: very long, hair-thin pieces (plate edge, shadow line). Characters are compact,
    and a headline joined to letters is part of a bigger piece, so neither is touched."""
    ch = _char_height(mask)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = mask.copy()
    for i in range(1, n):
        w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        long_side, short_side = max(w, h), min(w, h)
        # a bar is thin in ABSOLUTE terms (a fraction of a character's height); a whole digit row that has
        # joined up is wide but as tall as a character, and must survive
        if long_side > 1.5 * ch and short_side < 0.22 * ch:
            out[lab == i] = 0
        elif h > 0.55 * ch and w < 0.16 * h and (st[i, cv2.CC_STAT_LEFT] < 0.07 * mask.shape[1]
                                                  or st[i, cv2.CC_STAT_LEFT] + w > 0.93 * mask.shape[1]):
            out[lab == i] = 0  # a thin upright bar hugging the left/right edge: plate edge, not a digit
    return out


def _even_strokes(mask: np.ndarray) -> np.ndarray:
    """Give thin, broken strokes (soft plates) a consistent printed weight: if the typical stroke is much thinner
    than print at this character size, thicken it a little, then bridge the tiny gaps that remain."""
    ch = _char_height(mask)
    if mask.sum() < 30:
        return mask
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    ridge = dt[(dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-6) & (mask > 0)]
    stroke = 2.0 * float(np.median(ridge)) if len(ridge) else 0.0
    target = 0.12 * ch
    out = mask
    if 0 < stroke < 0.75 * target:
        r = int(min(4, round((target - stroke) / 2)))
        if r >= 1:
            out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    gap = max(3, int(0.03 * ch)) | 1
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap)))
    # print never covers more than ~35 % of its own bounding box; if the clean-up did, it merged strokes: undo it
    ys, xs = np.where(out > 0)
    if len(ys) and out.sum() / float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) > 0.35:
        return mask
    return out


def _local_ink(flat: np.ndarray, height: int) -> np.ndarray:
    """0..1 'darker than its neighbourhood' map: catches faint strokes a single global threshold misses."""
    f = flat.astype(np.float32) / 255.0
    win = max(5.0, 0.05 * height)
    mu = cv2.GaussianBlur(f, (0, 0), win)
    sd = np.sqrt(np.maximum(cv2.GaussianBlur((f - mu) ** 2, (0, 0), win), 1e-6))
    z = (mu - f) / (sd + 0.02)
    ink = _smoothstep((z - 0.35) / 1.0)
    return ink * ((mu - f) * 255.0 > 9.0)  # a stroke must be clearly darker than the paper around it


def _finish_rows(keep: np.ndarray, guide: np.ndarray, h0: int, thin_top: bool) -> np.ndarray:
    """Stroke clean-up per row. Bold digits get evened strokes; the small Bangla first row is thinned back to a
    print weight with its counters re-opened (see toprow.py) so the words stay legible."""
    bands = _row_bands(keep)
    if not thin_top or len(bands) != 2:
        return _smooth_mask(_even_strokes(keep), 0.0028 * h0 + 0.6)
    from opencv import toprow

    out = np.zeros_like(keep)
    for i, (y0, y1) in enumerate(bands):
        pad = int(0.04 * (y1 - y0))
        a, b = max(0, y0 - pad), min(keep.shape[0], y1 + pad)
        band = np.zeros_like(keep)
        band[y0:y1] = keep[y0:y1]
        if i == 0:
            part = toprow.thin_row(band[a:b], guide[a:b])
        else:
            part = _smooth_mask(_even_strokes(band[a:b]), 0.0028 * h0 + 0.6)
        out[a:b] = np.maximum(out[a:b], part)
    return out


def make_document(img: np.ndarray, *, out_width: int = 960, margin: float = 0.07, local: bool = True,
                  deblur: bool = True, sauvola: bool = False, rebuild: bool = True, fisher: bool = True, thin_top: bool = True, relief: str = "auto",
                  info: dict | None = None) -> np.ndarray:
    """Return a uint8 grey page: white background, smooth dark characters, level and tightly cropped.

    Blurry CCTV plates give noisy ink maps; instead of thresholding them directly the strokes are
    extracted with hysteresis, speckle and frame are removed, and the outlines are smoothed so the
    page looks like clean print rather than torn paper."""
    if relief == "force":  # embossed plate: there is no ink to threshold, so skip the whole ink pipeline
        from opencv import relief as RL

        if info is not None:
            info["relief_used"] = True
        return RL.relief_page(img, out_width)
    full = img
    if img.shape[0] > PAGE_MAX_H:
        img = cv2.resize(img, None, fx=PAGE_H / img.shape[0], fy=PAGE_H / img.shape[0], interpolation=cv2.INTER_AREA)
    work = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    work = deskew(work)
    s = 4.0 if max(work.shape[:2]) < 500 else (WORK_SCALE if max(work.shape[:2]) < 700 else 1.0)
    if s != 1.0:
        work = cv2.resize(work, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    work = cv2.bilateralFilter(work, 7, 25, 5)  # smooth noise but keep stroke edges

    from opencv.strokes import stroke_mask as _stroke_mask

    grow = max(3, int(0.035 * work.shape[0])) | 1
    near_stroke = cv2.dilate(_stroke_mask(work), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))

    def ink_binary(ch0: np.ndarray):
        """Ink mask from one grey projection of the plate: polarity, flatten, (deblur), threshold, hysteresis."""
        ch = cv2.GaussianBlur(ch0, (0, 0), 1.0)
        if _text_is_light(ch):
            ch = 255 - ch
        flat = _flatten(ch)
        if deblur:
            from opencv import cleanpage as CP

            sig = CP.estimate_blur_sigma(flat)
            if info is not None:
                info["blur_sigma"] = round(sig, 2)
            if sig >= 1.2:
                flat = CP.wiener_deblur(flat, min(sig * 0.8, 3.5), k_reg=0.012)
        ink = _soft_ink(flat)
        if sauvola:
            from opencv import cleanpage as CP

            ink = np.maximum(ink, CP.sauvola_ink(flat, int(0.12 * flat.shape[0])) * 0.9)
        if local:
            ink = np.maximum(ink, _local_ink(flat, work.shape[0]))
        # print only: keep ink near thin dark strokes (never big shadows or plate edges); the mask responds on stroke
        # edges, so it is grown by about half a stroke width to keep bold strokes solid instead of hollow outlines
        ink = ink * near_stroke.astype(np.float32)
        if ink.max() <= 0:
            return None
        ink = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), 0.0035 * ink.shape[0])
        return _hysteresis(ink, 0.42, 0.60), flat

    def coherence(b: np.ndarray) -> float:
        """Share of the ink that sits in solid, character-sized pieces (speckle scores low)."""
        total = float(b.sum())
        if total < 30:
            return 0.0
        n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        big = st[1:, cv2.CC_STAT_AREA] >= 0.0012 * b.size
        return float(st[1:, cv2.CC_STAT_AREA][big].sum()) / total

    candidates = [best_channel(work)]
    if fisher:  # also try the colour direction that best separates ink from the plate body
        from opencv import cleanpage as CP

        candidates.insert(0, CP.fisher_channel(work))
    best_b, best_c = None, -1.0
    for c0 in candidates:
        res = ink_binary(c0)
        if res is None:
            continue
        bb, fl = res
        sc = coherence(bb)
        if sc > best_c + 0.02 or best_b is None:  # a later (grey) candidate must be clearly better to win
            best_b, best_c, guide = bb, sc, fl
    if best_b is None:
        return np.full((int(out_width * 0.35), out_width), 255, np.uint8)
    binary = best_b
    h0, w0 = binary.shape
    rel_used = False
    if relief != "off":
        from opencv import relief as RL

        yy, xx = int(0.09 * h0), int(0.07 * w0)
        inner = binary[yy: h0 - yy, xx: w0 - xx]
        cov, coh = float(inner.mean()), best_c
        gi = guide[yy: h0 - yy, xx: w0 - xx].astype(np.uint8)
        thr_i, _ = cv2.threshold(gi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        body_i, text_i = gi[gi > thr_i], gi[gi <= thr_i]
        contrast = float(np.median(body_i) - np.median(text_i)) / max(float(np.median(body_i)), 1.0) if body_i.size > 8 and text_i.size > 8 else 0.0
        # the same contrast measured only on strokes that sit well inside the plate (not rim, bolts, background)
        core = cv2.erode(binary, np.ones((3, 3), np.uint8))[yy: h0 - yy, xx: w0 - xx] > 0
        stroke_dark = float(np.median(gi[core])) if core.sum() > 30 else float(np.median(gi))
        depth = (float(np.median(gi)) - stroke_dark) / max(float(np.median(gi)), 1.0)
        if info is not None:
            info["contrast"], info["depth"] = round(contrast, 3), round(depth, 3)
        if info is not None:
            info["ink_cov"], info["ink_coh"] = round(cov, 3), round(coh, 3)
        if relief == "force" or cov < RELIEF_COV or coh < RELIEF_COH:
            rb, rinfo = RL.relief_ink(work)
            if info is not None:
                info["relief"] = rinfo
            if relief == "force" or rb[yy: h0 - yy, xx: w0 - xx].mean() > cov:
                rel_used = True
    if info is not None:
        info["relief_used"] = rel_used
    if rel_used:  # embossed plate: no ink to threshold, draw the relief contours instead
        return RL.relief_page(full, out_width)
    inset = np.zeros_like(binary)
    iy, ix = int(0.085 * h0), int(0.065 * w0)  # the plate edge sits ~5-8 % inside the rectified frame
    inset[iy: h0 - iy, ix: w0 - ix] = binary[iy: h0 - iy, ix: w0 - ix]
    inset = _drop_edge_fragments(inset)
    keep, _ = clean_ink(inset)
    if keep.sum() < 30:
        keep, _ = clean_ink(binary)
    if keep.sum() < 30:
        keep = binary

    # drop specks (far below a character's ink) and plug pin-holes inside strokes, then round the outlines
    n, lab, st, _ = cv2.connectedComponentsWithStats(keep, connectivity=8)
    if n > 1:
        areas = st[1:, cv2.CC_STAT_AREA]
        floor = max(20, 0.10 * float(np.percentile(areas, 75)))
        keep = np.isin(lab, 1 + np.where(areas >= floor)[0]).astype(np.uint8)
    keep = _drop_bars(keep)
    keep = _finish_rows(keep, guide, h0, thin_top)
    keep = _align_rows(keep)
    if rebuild:
        from opencv import cleanpage as CP

        keep = CP.rebuild_strokes(keep, sigma=1.6, supersample=2).astype(np.uint8)
    h0 = keep.shape[0]
    if keep.sum() < 30:
        return np.full((int(out_width * 0.35), out_width), 255, np.uint8)

    ys, xs = np.where(keep > 0)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    mx, my = int(margin * (x1 - x0)), int(margin * 1.6 * (y1 - y0))
    box = keep[max(0, y0 - my): y1 + my, max(0, x0 - mx): x1 + mx].astype(np.float32)
    canvas = np.zeros((box.shape[0] + 2 * my, box.shape[1] + 2 * mx), np.float32)
    canvas[my: my + box.shape[0], mx: mx + box.shape[1]] = box
    scale = out_width / canvas.shape[1]
    canvas = cv2.resize(canvas, (out_width, max(16, int(canvas.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    crisp = _smoothstep((cv2.GaussianBlur(canvas, (0, 0), 1.0) - 0.30) / 0.40)
    page = np.clip((1.0 - crisp) * 255.0, 0, 255).astype(np.uint8)
    page[page > 245] = 255
    return page


