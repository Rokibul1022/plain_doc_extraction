"""Text layout of a plate: remove the frame, find the text rows, and list each row's characters.

Shared by the plain-document renderer and the plate reader. Works on a binary ink mask
(1 = ink) from a rectified, levelled plate crop.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Glyph:
    """One connected piece of ink (a character, part of one, or a mark such as a hyphen)."""

    label: int
    x: int
    y: int
    w: int
    h: int
    area: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@dataclass
class Row:
    glyphs: list[Glyph] = field(default_factory=list)
    top: int = 0
    bottom: int = 0
    char_h: float = 0.0

    @property
    def left(self) -> int:
        return min(g.x for g in self.glyphs)

    @property
    def right(self) -> int:
        return max(g.x + g.w for g in self.glyphs)


@dataclass
class Layout:
    rows: list[Row]
    labels: np.ndarray  # connected-component label image of the cleaned ink
    char_h: float

    def mask(self) -> np.ndarray:
        """Binary mask of every glyph that belongs to a text row."""
        keep = np.zeros(self.labels.shape, np.uint8)
        for r in self.rows:
            for g in r.glyphs:
                keep[self.labels == g.label] = 1
        return keep


def remove_frame_lines(binary: np.ndarray) -> np.ndarray:
    """Delete long horizontal/vertical runs (plate frame, rules). Characters have none."""
    h, w = binary.shape
    # Frame lines run most of the plate's width/height; the long vertical strokes of bold digits are only
    # about a third of the height, so the thresholds must stay well above that.
    lh, lv = max(15, int(0.50 * w)), max(15, int(0.58 * h))
    hor = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (lh, 1)))
    ver = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, lv)))
    lines = cv2.dilate(cv2.bitwise_or(hor, ver), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.bitwise_and(binary, cv2.bitwise_not(lines))


def _split_rows(glyphs: list[Glyph], char_h: float) -> list[list[Glyph]]:
    """Group glyphs into text rows by the vertical gaps between their centres."""
    ordered = sorted(glyphs, key=lambda g: g.cy)
    groups: list[list[Glyph]] = [[ordered[0]]]
    for g in ordered[1:]:
        # compare with the running mean centre of the current row, not just the last glyph
        mean_cy = float(np.mean([x.cy for x in groups[-1]]))
        if g.cy - mean_cy > 0.62 * char_h:
            groups.append([g])
        else:
            groups[-1].append(g)
    return groups


def analyze(binary: np.ndarray, max_rows: int = 3) -> Layout:
    """Find the text rows of a plate. Frame fragments and bolts end up outside every row."""
    h, w = binary.shape
    cleaned = remove_frame_lines(binary)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    img_area = float(h * w)
    glyphs = [Glyph(i, *map(int, st[i])) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] / img_area >= 0.00005]
    if not glyphs:
        return Layout([], lab, 0.0)

    areas = np.array([g.area for g in glyphs])
    big = [g for g in glyphs if g.area >= np.percentile(areas, 55)]
    char_h = float(np.median([g.h for g in big]))

    def sliver(g: Glyph, ref: float) -> bool:
        return (g.h < 0.22 * ref and g.w > 0.8 * ref) or (g.w < 0.20 * ref and g.h > 1.15 * ref)

    # bolt holes: compact round blobs in the top corners
    cand = []
    for g in glyphs:
        fill = g.area / float(max(g.w * g.h, 1))
        corner = (g.cx < 0.20 * w or g.cx > 0.80 * w) and g.cy < 0.26 * h
        if corner and 0.55 < fill < 0.95 and 0.6 < g.w / float(max(g.h, 1)) < 1.7 and g.area / img_area < 0.006:
            continue
        if sliver(g, char_h) or g.h > 2.4 * char_h or g.w > 7.0 * char_h:
            continue
        cand.append(g)
    if not cand:
        return Layout([], lab, char_h)

    groups = _split_rows(cand, char_h)
    # a real row holds several glyphs with real ink; stray groups are frame/background debris
    scored = sorted(groups, key=lambda grp: -sum(g.area for g in grp))[:max_rows]
    rows: list[Row] = []
    for grp in scored:
        if len(grp) < 2 and sum(g.area for g in grp) < 0.5 * np.median(areas):
            continue
        row_h = float(np.median([g.h for g in grp if g.area >= np.percentile([x.area for x in grp], 40)] or [char_h]))
        keep = [g for g in grp if not sliver(g, row_h) and g.h <= 2.2 * row_h]
        if not keep:
            continue
        top = min(g.y for g in keep)
        bottom = max(g.y + g.h for g in keep)
        rows.append(Row(sorted(keep, key=lambda g: g.x), top, bottom, row_h))
    rows.sort(key=lambda r: r.top)
    return Layout(rows, lab, char_h)


def frame_crop_box(binary: np.ndarray, band: float = 0.20, thresh: float = 0.55) -> tuple[int, int, int, int]:
    """Locate the plate frame lines near each edge and return the box just inside them.

    A frame line is a column (or row) of which more than `thresh` is ink - far more than any
    character stroke. The innermost such line within the outer `band` of each side marks the frame."""
    h, w = binary.shape
    cols = binary.mean(axis=0)
    rows = binary.mean(axis=1)

    def inner_edge(profile: np.ndarray, length: int, from_end: bool) -> int:
        n = max(2, int(band * length))
        seg = profile[-n:][::-1] if from_end else profile[:n]
        hits = np.where(seg >= thresh)[0]
        if len(hits) == 0:
            return 0
        # step over the (possibly thick) line: stop at the last hit of the first contiguous cluster
        first = hits[0]
        end = first
        for k in hits[1:]:
            if k - end <= 2:
                end = k
            else:
                break
        return int(end + 1)

    x0 = inner_edge(cols, w, False)
    x1 = w - inner_edge(cols, w, True)
    y0 = inner_edge(rows, h, False)
    y1 = h - inner_edge(rows, h, True)
    if x1 - x0 < 0.4 * w or y1 - y0 < 0.4 * h:
        return 0, 0, w, h
    return x0, y0, x1, y1


def remove_bolts(binary: np.ndarray) -> np.ndarray:
    """Erase small round blobs in the top corners (screw holes), leaving everything else."""
    h, w = binary.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = binary.copy()
    for i in range(1, n):
        x, y, cw, ch, area = st[i]
        fill = area / float(max(cw * ch, 1))
        cx, cy = x + cw / 2.0, y + ch / 2.0
        if (cx < 0.22 * w or cx > 0.78 * w) and cy < 0.28 * h and 0.5 < fill < 0.97                 and 0.6 < cw / float(max(ch, 1)) < 1.7 and area / float(h * w) < 0.008:
            out[lab == i] = 0
    return out


def clean_ink(binary: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Plate ink without frame and bolts. Returns (mask the size of `binary`, inner box)."""
    x0, y0, x1, y1 = frame_crop_box(binary)
    inner = np.zeros_like(binary)
    inner[y0:y1, x0:x1] = binary[y0:y1, x0:x1]
    inner = remove_frame_lines(inner)
    inner = remove_bolts(inner)
    # drop slivers and speckle relative to the plate's own character size
    n, lab, st, _ = cv2.connectedComponentsWithStats(inner, connectivity=8)
    if n <= 1:
        return inner, (x0, y0, x1, y1)
    img_area = float(binary.shape[0] * binary.shape[1])
    comps = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] / img_area >= 0.00005]
    if not comps:
        return np.zeros_like(binary), (x0, y0, x1, y1)
    areas = np.array([st[i, cv2.CC_STAT_AREA] for i in comps])
    char_h = float(np.median([st[i, cv2.CC_STAT_HEIGHT] for i in comps if st[i, cv2.CC_STAT_AREA] >= np.percentile(areas, 55)]))
    keep = np.zeros_like(inner)
    for i in comps:
        cw, ch = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if (ch < 0.2 * char_h and cw > 0.9 * char_h) or (cw < 0.18 * char_h and ch > 1.2 * char_h):
            continue
        keep[lab == i] = 1
    return keep, (x0, y0, x1, y1)
