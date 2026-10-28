"""3D relief reading for embossed plates (no ink contrast) - OpenCV + NumPy only.

Stamped letters are the same colour as the plate. They only show as light and shadow: the face turned towards the
light has a bright edge, the face turned away a dark one. To first order the brightness is the derivative of the
surface height h along the light direction l:    I_hp ~ dh/dl.

So, walking along l across a raised stroke, the signed derivative D = dI/dl goes  + (ridge, lit)  ...  - (shadow).
A stroke pixel is therefore one that has a lit edge a short way behind it and a shadow edge a short way ahead of it.
That pairing is what separates a character from grime (single edges), peeling paint (random signs) and the rim.

  1. luminance, high-pass and contrast-normalise (colour and vignetting play no role);
  2. light direction: highlight prior (bolt heads and rim) + the direction whose ridge/shadow pairing gives the most
     solid, character-sized coverage; the pairing order (raised vs recessed) is chosen the same way;
  3. signed directional derivative -> ridge map P and shadow map N -> stroke = P-behind AND N-ahead (fills the
     stroke between ridge and shadow);
  4. strokes that run parallel to the light have no signed pair: they are added from the edge energy, closed at stroke
     width;
  5. hysteresis, solid components only, rim lines and frame removed.
"""

import cv2
import numpy as np

ANGLES = 8  # light azimuths tried over the full circle


def _luminance(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)


def _highpass(lum: np.ndarray, sigma: float) -> np.ndarray:
    """Remove slow illumination and paint tone, then scale by local contrast so faint relief counts like strong."""
    hp = lum - cv2.GaussianBlur(lum, (0, 0), sigma)
    sd = np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), sigma * 1.5)) + 1.5
    return hp / sd


def _shift_or(m: np.ndarray, dx: float, dy: float, span: int) -> np.ndarray:
    """OR of the mask shifted by s*(dx,dy) for s = 0..span (a one-sided dilation along the light direction)."""
    out = m.copy()
    h, w = m.shape
    for s in range(1, span + 1):
        M = np.float32([[1, 0, dx * s], [0, 1, dy * s]])
        out |= cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return out


def _stroke_pairs(hs: np.ndarray, ang: float, span: int, order: int, k: float) -> np.ndarray:
    """Pixels with a ridge behind and a shadow ahead along `ang` (order=+1), or the reverse (order=-1)."""
    gx = cv2.Sobel(hs, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(hs, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    lx, ly = np.cos(ang), np.sin(ang)
    d = gx * lx + gy * ly * 1.0
    t = k * float(np.percentile(np.abs(d), 75))
    pos = (order * d > t).astype(np.uint8)
    neg = (order * d < -t).astype(np.uint8)
    behind = _shift_or(pos, lx, ly, span)  # a ridge exists within `span` px behind the pixel
    ahead = _shift_or(neg, -lx, -ly, span)  # a shadow exists within `span` px ahead of it
    return behind & ahead


def _solidity(m: np.ndarray, stroke: float) -> float:
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return 0.0
    a = st[1:, cv2.CC_STAT_AREA].astype(np.float32)
    return float(a[a >= (1.5 * stroke) ** 2].sum()) / max(float(a.sum()), 1.0)


def estimate_light(img: np.ndarray, hs: np.ndarray, stroke: float, interior: np.ndarray) -> tuple[float, int, float]:
    """(azimuth radians, order, score): the direction/order whose ridge-shadow pairing looks most like characters."""
    lum = _luminance(img)
    h, w = lum.shape
    thr = np.percentile(lum, 98.5)
    ys, xs = np.where(lum >= thr)
    prior = None
    if len(xs) > 20:
        v = np.array([xs.mean() - w / 2.0, ys.mean() - h / 2.0])
        if np.linalg.norm(v) > 0.05 * max(h, w):
            prior = float(np.arctan2(v[1], v[0]))
    span = max(3, int(round(1.1 * stroke)))
    best = (-1.0, 0.0, 1)
    for k in range(ANGLES):
        a = 2 * np.pi * k / ANGLES
        for order in (1, -1):
            m = _stroke_pairs(hs, a, span, order, 1.0) * interior
            cov = float(m.sum()) / max(float(interior.sum()), 1.0)
            if cov < 0.01 or cov > 0.6:
                continue
            sc = _solidity(m, stroke) * (1.0 - abs(cov - 0.20))
            if prior is not None:
                sc += 0.04 * np.cos(a - prior)  # mild pull towards the highlight side
            if sc > best[0]:
                best = (sc, a, order)
    return best[1], best[2], best[0]


def relief_ink(img: np.ndarray, stroke: float | None = None) -> tuple[np.ndarray, dict]:
    """Binary stroke map (uint8 0/1, same size as `img`) from the plate's relief, plus diagnostics."""
    lum = _luminance(img)
    h, w = lum.shape
    stroke = float(stroke) if stroke else max(2.0, 0.03 * h)
    hs = cv2.GaussianBlur(_highpass(cv2.GaussianBlur(lum, (0, 0), 0.7), max(8.0, 5.0 * stroke)), (0, 0), 0.6 * stroke * 0.5 + 0.4)
    interior = np.zeros((h, w), np.uint8)
    iy, ix = int(0.09 * h), int(0.07 * w)
    interior[iy: h - iy, ix: w - ix] = 1
    ang, order, score = estimate_light(img, hs, stroke, interior)
    span = max(3, int(round(1.1 * stroke)))
    pair = _stroke_pairs(hs, ang, span, order, 0.9)
    # strokes parallel to the light: edge energy, closed at stroke width so ridge+shadow edges become one stroke
    gx = cv2.Sobel(hs, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(hs, cv2.CV_32F, 0, 1, ksize=3)
    en = np.hypot(gx, gy)
    en_t = float(np.percentile(en[interior > 0], 82))
    edge = (en > en_t).astype(np.uint8)
    k = max(3, int(0.8 * stroke)) | 1
    closed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # hysteresis: paired strokes are the strong seeds, closed edge energy may extend them
    m = _grow_from(pair, closed)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    m = (m * interior).astype(np.uint8)
    # solid, character-sized pieces only; thin long pieces along the border are rim/frame
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    ok = np.zeros(n, np.uint8)
    iy_ix = (iy + 2, ix + 2)
    for i in range(1, n):
        x, y, cw, chh, a = st[i]
        if a < (1.1 * stroke) ** 2:
            continue
        thin_long = max(cw, chh) > 0.5 * w and min(cw, chh) < 1.6 * stroke
        if thin_long:
            continue
        touches = x <= iy_ix[1] or y <= iy_ix[0] or x + cw >= w - iy_ix[1] or y + chh >= h - iy_ix[0]
        if touches and a < (4.0 * stroke) ** 2:
            continue
        ok[i] = 1
    out = ok[lab]
    return out, {"angle_deg": round(float(np.degrees(ang)), 1), "order": order, "score": round(float(score), 3),
                 "coverage": round(float(out.mean()), 3)}


def _grow_from(seed: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Connected components of (seed | region) that contain a seed pixel."""
    u = (seed | region).astype(np.uint8)
    n, lab = cv2.connectedComponents(u, connectivity=8)
    keep = np.zeros(n, np.uint8)
    keep[np.unique(lab[seed > 0])] = 1
    keep[0] = 0
    return keep[lab]


# ---------------------------------------------------------------- readable page for embossed plates
def _plate_mask(h: int, w: int) -> np.ndarray:
    """Where characters can be: the plate interior minus the corners that carry the mounting bolts."""
    m = np.zeros((h, w), np.float32)
    iy, ix = int(0.09 * h), int(0.09 * w)
    m[iy: h - iy, ix: w - ix] = 1.0
    cy, cx = int(0.30 * h), int(0.20 * w)  # the mounting bolts sit in the top corners
    m[0:cy, 0:cx] = 0.0
    m[0:cy, w - cx: w] = 0.0
    return cv2.GaussianBlur(m, (0, 0), 2.5)


def relief_lines(img: np.ndarray, work_h: int = 360) -> np.ndarray:
    """Line strength (float32, 0 flat .. 1 strong contour) of the plate's relief, at `work_h` pixels height.

    denoise (NLM) -> flatten light (high-pass, contrast-normalised) -> edge strength at stroke scale ->
    mask bolts / rim / background -> remove long straight rim runs -> contrast stretch."""
    s = work_h / img.shape[0]
    im = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    lum = _luminance(im)
    h, w = lum.shape
    k = work_h / 360.0  # all scales below were tuned at 360 px height
    den = cv2.fastNlMeansDenoising(np.clip(lum, 0, 255).astype(np.uint8), None, 6, 5, 15).astype(np.float32)
    hp = _highpass(cv2.GaussianBlur(den, (0, 0), max(0.5, 0.8 * k)), max(8.0 * k, 0.15 * h))
    g = cv2.GaussianBlur(hp, (0, 0), max(0.6, 1.0 * k))
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    mask = _plate_mask(h, w)
    ref = float(np.percentile(mag[mask > 0.5], 90)) + 1e-6
    line = np.clip(mag / ref, 0.0, 1.3) / 1.3 * mask
    # rim and frame: long straight runs of line (a glyph contour is never as long as the plate is wide / tall)
    strong = (line > 0.30).astype(np.uint8)
    longh = cv2.morphologyEx(strong, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, int(0.30 * w)), 1)))
    longv = cv2.morphologyEx(strong, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(9, int(0.42 * h)))))
    rim = cv2.dilate(longh | longv, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    line = line * (1.0 - cv2.GaussianBlur(rim.astype(np.float32), (0, 0), 1.2))
    # drop the faint texture (grime, paint grain) and stretch what is left; keeps real glyph contours strong
    return np.clip((line - 0.16) / 0.55, 0.0, 1.0) ** 0.85


def relief_view(img: np.ndarray, work_h: int = 160) -> np.ndarray:
    """Reader input for embossed plates: the signed shading of the relief, uint8, flat plate = 128.

    Light and shadow are kept as they are (lit ridge bright, shadowed face dark) after the lighting, colour and paint
    tone are removed (high-pass, contrast-normalised), so no information is thrown away and nothing is differentiated
    twice (an edge map of a shaded image rings). A model trained on synthetic embossed plates lit from random
    directions sees real ones in the same form. Uncropped (the layout is kept), `work_h` pixels high."""
    s = work_h / img.shape[0]
    im = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    lum = _luminance(im)
    h = lum.shape[0]
    k = work_h / 360.0
    hp = _highpass(cv2.GaussianBlur(lum, (0, 0), max(0.5, 0.8 * k)), max(8.0 * k, 0.15 * h))
    return np.clip(128.0 + 42.0 * np.clip(hp, -3.0, 3.0), 0, 255).astype(np.uint8)


def relief_page(img: np.ndarray, out_width: int = 960) -> np.ndarray:
    """Plain page (uint8, white paper) that draws the plate's raised characters as dark outlines.

    The light and shadow edges of the embossing are turned into line strength: the page is white where the plate is
    flat and dark where the relief changes, so every raised glyph shows as a crisp contour that can be read by eye.
    The line map (see `relief_lines`) is cropped to the text, scaled to the page width and slightly firmed up."""
    line = relief_lines(img, 360)
    h, w = line.shape
    page = 255.0 - 255.0 * line
    # crop to the text with a margin
    ink = line > 0.35
    if ink.sum() < 30:
        crop = page
    else:
        ys, xs = np.where(ink)
        y0, y1, x0, x1 = np.percentile(ys, 0.5), np.percentile(ys, 99.5), np.percentile(xs, 0.2), np.percentile(xs, 99.8)
        my, mx = 0.10 * (y1 - y0) + 6, 0.07 * (x1 - x0) + 8
        crop = page[int(max(0, y0 - my)): int(min(h, y1 + my)), int(max(0, x0 - mx)): int(min(w, x1 + mx))]
    k = out_width / crop.shape[1]
    out = cv2.resize(crop, (out_width, max(16, int(crop.shape[0] * k))), interpolation=cv2.INTER_CUBIC)
    out = cv2.erode(np.clip(out, 0, 255).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # firmer lines
    out[out > 244] = 255
    return out
