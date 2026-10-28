"""Turn a restored plate crop into clean, easily readable plain text (dark on white).

Unlike the fixed tone curve used for paper documents, every threshold here is derived from the
plate's own measured text/body tones, so faint or blurred strokes stay visible instead of being
whitened away. Coloured plates are handled by choosing the channel that best separates text
from body, and light-on-dark plates are inverted automatically.
"""

import cv2
import numpy as np


def _otsu_separation(ch: np.ndarray) -> float:
    thr, _ = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo, hi = ch[ch <= thr], ch[ch > thr]
    if lo.size < 8 or hi.size < 8:
        return 0.0
    w0, w1 = lo.size / ch.size, hi.size / ch.size
    return float(w0 * w1 * (float(hi.mean()) - float(lo.mean())) ** 2)


def _dark_minority(ch: np.ndarray) -> bool:
    """True if the darker Otsu class is the smaller one (dark characters on a lighter body)."""
    thr, _ = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((ch <= thr).mean()) < 0.5


def best_channel(img: np.ndarray) -> np.ndarray:
    """Greyscale projection (gray, B, G, R or PCA) with the strongest text/body separation.

    Only projections in which the characters are the darker minority qualify, so a coloured body
    (teal, green) can never be turned into a "dark body with light text" by a channel choice."""
    if img.ndim == 2:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cands = [gray, *cv2.split(img)]
    flat = img.reshape(-1, 3).astype(np.float32)
    flat -= flat.mean(axis=0)
    if flat.shape[0] > 10:
        _, _, vt = np.linalg.svd(flat[:: max(1, flat.shape[0] // 4000)], full_matrices=False)
        pc = (img.astype(np.float32) - img.reshape(-1, 3).mean(axis=0)) @ vt[0]
        pc = cv2.normalize(pc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        for c in (pc, 255 - pc):
            cands.append(c)
    scores = [_otsu_separation(c) if (i == 0 or _dark_minority(c)) else 0.0 for i, c in enumerate(cands)]
    best = int(np.argmax(scores))
    # prefer plain gray unless another projection is clearly better (>25 %)
    return cands[best] if scores[best] > 1.25 * scores[0] else gray


def _text_is_light(ch: np.ndarray) -> bool:
    """True only for a genuinely dark plate body carrying clearly lighter characters.

    Plates overwhelmingly use dark characters on a lighter - often coloured - body (white, yellow,
    teal, green), and stroke-shape cues are fooled by coloured bodies and dark surroundings. So dark
    text is the default and the polarity flips only when the central region is mostly very dark
    with a much brighter minority."""
    h, w = ch.shape[:2]
    roi = ch[int(0.20 * h): max(int(0.80 * h), int(0.20 * h) + 4), int(0.10 * w): max(int(0.90 * w), int(0.10 * w) + 4)]
    thr, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark, light = roi[roi <= thr], roi[roi > thr]
    if dark.size < 8 or light.size < 8:
        return False
    dark_majority = dark.size > light.size
    return bool(dark_majority and float(dark.mean()) < 85 and float(light.mean()) - float(dark.mean()) > 60)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def make_readable(img: np.ndarray, *, binary: bool = False, min_speckle: int = 4) -> np.ndarray:
    """Return a uint8 grey image with dark text on a white body (or a hard 0/255 binary)."""
    ch = best_channel(img)
    h, w = ch.shape[:2]
    ch = cv2.GaussianBlur(ch, (0, 0), 0.5)

    # polarity: characters are thin strokes, the plate body is broad. Whichever tone class has
    # the smaller stroke width (distance-to-boundary) is the text; make it dark.
    if _text_is_light(ch):
        ch = 255 - ch

    # flatten uneven illumination with a closing wider than a stroke
    k = max(7, int(0.45 * h) | 1)
    bg = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), max(2.0, k / 4.0))
    flat = np.clip(ch.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0) * 255.0, 0, 255)
    flat_u8 = flat.astype(np.uint8)

    # data-driven tone mapping
    thr2, _ = cv2.threshold(flat_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    body = flat[flat > thr2]
    text = flat[flat <= thr2]
    if body.size < 8 or text.size < 8 or float(np.median(body) - np.median(text)) < 12:
        out = cv2.normalize(flat_u8, None, 0, 255, cv2.NORM_MINMAX)
    else:
        body_med, text_med = float(np.median(body)), float(np.median(text))
        lo = text_med + 0.10 * (thr2 - text_med)
        hi = thr2 + 0.55 * (body_med - thr2)
        out = (255.0 * _smoothstep((flat - lo) / max(hi - lo, 1.0))).astype(np.uint8)

    if binary:
        _, bw = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)
        inv = cv2.bitwise_not(bw)
        n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        keep = np.zeros_like(inv)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= min_speckle:
                keep[lab == i] = 255
        out = cv2.bitwise_not(keep)

    b = max(1, min(3, int(min(h, w) * 0.02)))
    out[:b, :] = out[-b:, :] = 255
    out[:, :b] = out[:, -b:] = 255
    return out
