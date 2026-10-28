"""Normalise a rectified plate crop into the plain input the digit reader sees.

Level the plate, remove the slant, strip the frame, find the number row, crop it tightly and turn it into
black ink on white (so plate colour, lighting and shadows disappear). The same function prepares training
and inference images, so the network only has to learn digit shapes, not pose, scale or colour.
"""

import cv2
import numpy as np

from opencv.geometry import deshear, deskew
from opencv.layout import remove_frame_lines
from opencv.readable import _smoothstep, _text_is_light, best_channel

IN_H, IN_W = 48, 128
MIN_WORK_H = 220


def _otsu_ink(img: np.ndarray) -> np.ndarray:
    """0..1 ink strength from a flattened grey image and Otsu body/text levels (1 = ink)."""
    ch = cv2.GaussianBlur(best_channel(img), (0, 0), 0.8)
    if _text_is_light(ch):
        ch = 255 - ch
    h = ch.shape[0]
    k = max(9, int(0.30 * h) | 1)
    bg = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), max(2.0, k / 4.0))
    flat = np.clip(ch.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0) * 255.0, 0, 255)
    thr, _ = cv2.threshold(flat.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    body, text = flat[flat > thr], flat[flat <= thr]
    if body.size < 8 or text.size < 8 or float(np.median(body)) - float(np.median(text)) < 12:
        return np.zeros_like(flat)
    bm, tm = float(np.median(body)), float(np.median(text))
    lo, hi = tm + 0.10 * (thr - tm), thr + 0.45 * (bm - thr)
    return 1.0 - _smoothstep((flat - lo) / max(hi - lo, 1.0))


def _stroke_mask(img: np.ndarray) -> np.ndarray:
    """Where thin dark strokes exist (black-hat), slightly grown. Large dark areas give no response."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
    k = max(9, int(0.11 * gray.shape[0]) | 1)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    top = float(np.percentile(bh, 99.0))
    if top < 14.0:
        return np.zeros(gray.shape, np.uint8)
    return cv2.dilate((bh > 0.22 * top).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


def _flat_ink(img: np.ndarray) -> np.ndarray:
    """Print only: Otsu ink restricted to places that have a thin-stroke response (drops dark surroundings)."""
    return _otsu_ink(img) * _stroke_mask(img)


def _character_rows(binary: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    """Group the ink components into text rows by character height; returns (rows, component stats).

    Huge blobs (dark background, plate edge) are not characters and are discarded first. A row is a
    cluster of components with similar vertical centres; the number row is the lowest row that holds
    several character-sized components."""
    h, w = binary.shape
    n, lab, st, cen = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = []
    for i in range(1, n):
        x, y, cw, ch, area = st[i]
        if area < 0.0004 * h * w or ch < 0.05 * h:
            continue  # speckle
        if ch > 0.55 * h or cw > 0.92 * w or area > 0.10 * h * w:
            continue  # background blob / frame fragment, not a character (touching digits may be wide)
        fill = area / float(max(cw * ch, 1))
        if fill < 0.16 or (cw > 4.5 * ch and ch < 0.2 * h) or (ch > 4.5 * cw and cw < 0.05 * w):
            continue  # hair-thin diagonal bar (plate edge, bodywork), not a character
        keep.append(i)
    if len(keep) < 3:
        return [], st
    hs = np.array([st[i, cv2.CC_STAT_HEIGHT] for i in keep])
    char_h = float(np.median(hs))
    order = sorted(keep, key=lambda i: cen[i][1])
    rows: list[list[int]] = [[order[0]]]
    for i in order[1:]:
        if cen[i][1] - float(np.mean([cen[j][1] for j in rows[-1]])) > 0.7 * char_h:
            rows.append([i])
        else:
            rows[-1].append(i)
    return [r for r in rows if len(r) >= 1], st


def normalise_plate(crop: np.ndarray) -> np.ndarray:
    """Level + de-slant a plate crop (BGR/grey) and upscale it to a working size."""
    img = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    if img.shape[0] < MIN_WORK_H:
        s = MIN_WORK_H / img.shape[0]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return deshear(deskew(img, max_angle=20.0))


def row_roi(crop: np.ndarray, pad: float = 0.10, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """uint8 tight crop of the number row at working resolution: black ink on white, frame removed.

    `pad` (fraction of the row height added around the row) and `dx`/`dy` (shifts) give the slightly
    different views used for voting."""
    img = normalise_plate(crop)
    ink = _flat_ink(img)
    binary = (ink > 0.40).astype(np.uint8)
    h, w = binary.shape
    inset = np.zeros_like(binary)
    iy, ix = int(0.03 * h), int(0.03 * w)
    inset[iy: h - iy, ix: w - ix] = binary[iy: h - iy, ix: w - ix]
    no_frame = remove_frame_lines(inset)
    rows, st = _character_rows(no_frame)
    if not rows:
        y0, y1 = int(0.4 * h), h
        x0, x1 = 0, w
    else:
        # the number row is the lowest row that is several characters wide: pieces may touch (bold fonts), so
        # count width in character units rather than counting separate components
        def chars_wide(r: list[int]) -> float:
            rh = float(np.median([st[i, cv2.CC_STAT_HEIGHT] for i in r]))
            return float(sum(st[i, cv2.CC_STAT_WIDTH] for i in r)) / max(0.6 * rh, 1.0)

        good = [r for r in rows if chars_wide(r) >= 4.0]
        row = good[-1] if good else max(rows, key=chars_wide)
        y0 = int(min(st[i, cv2.CC_STAT_TOP] for i in row))
        y1 = int(max(st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] for i in row))
        x0 = int(min(st[i, cv2.CC_STAT_LEFT] for i in row))
        x1 = int(max(st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH] for i in row))
    rh, rw = y1 - y0, x1 - x0
    p = pad * rh
    ya, yb = int(y0 - p + dy * rh), int(y1 + p + dy * rh)
    xa, xb = int(x0 - p + dx * rh), int(x1 + p + dx * rh)
    page = (1.0 - ink * (1 - cv2.dilate(remove_frame_lines(binary) ^ binary, np.ones((3, 3), np.uint8)))) * 255.0
    page = np.clip(page, 0, 255).astype(np.uint8)
    pad_img = cv2.copyMakeBorder(page, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)
    roi = pad_img[max(0, ya + 60): max(ya + 62, yb + 60), max(0, xa + 60): max(xa + 62, xb + 60)]
    if roi.size == 0:
        roi = page
    return roi


def digit_row_image(crop: np.ndarray, pad: float = 0.10, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """uint8 IN_H x IN_W (squashed) image of the number row, as the neural reader consumed it."""
    return cv2.resize(row_roi(crop, pad, dx, dy), (IN_W, IN_H), interpolation=cv2.INTER_AREA)
