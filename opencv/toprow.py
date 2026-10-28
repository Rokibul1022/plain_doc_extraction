"""Top-row (city / series letter) clean-up for the plain page - OpenCV + NumPy only.

The first row of a Bangladeshi plate is small, thin Bangla script with conjuncts (e.g. the "tro" of Metro).
The generic stroke clean-up that suits bold digits closes its counters and fattens its strokes, so the words
turn into dark blobs. This module re-works that row alone:

  1. measure the real stroke width of the row (distance-transform ridge) and the row height;
  2. thin a bold row back to a print weight (a fraction of the row height) by eroding the mask, while a
     morphological skeleton is kept so thin strokes never break;
  3. re-open counters and gaps that the blur filled, using the grey image as a guide: pixels inside the mask
     that are clearly lighter than the stroke core are paper, not ink;
  4. drop specks and lightly round the outline.
"""

import cv2
import numpy as np

TARGET_FRAC = 0.095  # print stroke width as a share of the row height
RATIO = 1.15  # rows thicker than RATIO x target are thinned
MODES = ("sauvola", "otsu")  # candidates for the re-extraction; MODE-names: re-extraction: "sauvola", "ridge" or "otsu" (NLM denoise + per-row Otsu + hysteresis)
DENOISE_H = 12
OTSU_STRONG, OTSU_WEAK = 0.10, 0.08
RIDGE_LO, RIDGE_HI = 0.30, 0.55
BRIDGE = True  # join stroke ends / dots across gaps where the grey image shows ink
MONO = False  # redraw the row as uniform-width connected strokes (skeleton + spur pruning)
MONO_WIDTH = 1.3  # drawn stroke width as a multiple of the measured median stroke width
REBIN = True  # re-threshold the row from the grey image (own deblur, CLAHE, Sauvola)
FILL_MAX = 0.31  # rows covering more of their box than this are blobby and are thinned
FILL_GOAL = 0.27  # coverage the thinning aims for


def _skeleton(mask: np.ndarray) -> np.ndarray:
    """Morphological (Lantuejoul) skeleton: centre lines of every stroke, thin-stroke safe."""
    el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = mask.copy()
    sk = np.zeros_like(mask)
    for _ in range(64):
        if not img.any():
            break
        er = cv2.erode(img, el)
        op = cv2.dilate(er, el)
        sk |= (img & (1 - op)).astype(np.uint8)
        img = er
    return sk


def stroke_width(mask: np.ndarray) -> float:
    """Typical stroke width in pixels (twice the median distance-transform ridge value)."""
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    ridge = dt[(dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-6) & (mask > 0)]
    return 2.0 * float(np.median(ridge)) if len(ridge) else 0.0


def _drop_blobs(m: np.ndarray, target: float) -> np.ndarray:
    """Remove solid, compact lumps (rivets, bolt heads, shadow corners): after thinning no real glyph piece
    is a filled blob much thicker than the print stroke."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:
        return m
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    out = m.copy()
    for i in range(1, n):
        x, y, w, h, a = st[i]
        solid = a / float(w * h) > 0.45 and max(w, h) / float(min(w, h)) < 2.5
        thick = 2.0 * float(dt[lab == i].max()) > 2.2 * target
        cx, cy = x + w / 2.0, y + h / 2.0
        gy, gx = np.where(m > 0)
        X0, X1, Y0, Y1 = gx.min(), gx.max() + 1, gy.min(), gy.max() + 1
        edge = (cx - X0) < 0.15 * (X1 - X0) or (X1 - cx) < 0.15 * (X1 - X0) or (cy - Y0) < 0.22 * (Y1 - Y0) or (Y1 - cy) < 0.22 * (Y1 - Y0)
        if solid and thick and edge:
            out[lab == i] = 0
    return out


def _strip_frame(m: np.ndarray, target: float) -> np.ndarray:
    """Remove what is not text but survives upstream: long horizontal rules along the top/bottom of the row,
    long vertical bars at its left/right ends, and thick lumps (rivets, shadow corners) glued to them."""
    ys, xs = np.where(m > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    H, W = y1 - y0, x1 - x0
    kill = np.zeros_like(m)
    L = max(9, int(0.70 * W))
    hl = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    edge = np.zeros_like(m)
    edge[y0: y0 + int(0.2 * H) + 1] = 1
    edge[y1 - int(0.2 * H) - 1: y1] = 1
    kill |= cv2.dilate(hl, np.ones((3, 1), np.uint8)) * edge
    Lv = max(9, int(0.8 * H))
    vl = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, Lv)))
    side = np.zeros_like(m)
    side[:, x0: x0 + int(0.07 * W) + 1] = 1
    side[:, x1 - int(0.07 * W) - 1: x1] = 1
    kill |= cv2.dilate(vl, np.ones((1, 3), np.uint8)) * side
    d = int(max(3, 2.2 * target)) | 1
    lumps = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(lumps, connectivity=8)
    for i in range(1, n):  # only compact solid lumps, never a long bold glyph stem
        w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        cx, cy = st[i, cv2.CC_STAT_LEFT] + w / 2.0 - x0, st[i, cv2.CC_STAT_TOP] + h / 2.0 - y0
        at_edge = cx < 0.15 * W or cx > 0.85 * W or cy < 0.22 * H or cy > 0.78 * H  # rivets sit at the border, letters do not
        if at_edge and max(w, h) < 0.45 * H and st[i, cv2.CC_STAT_AREA] / float(w * h) > 0.45:
            kill |= cv2.dilate((lab == i).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    out = (m & (1 - kill)).astype(np.uint8)
    return out if out.sum() > 0.5 * m.sum() else m


def _pieces(m: np.ndarray) -> int:
    """Number of ink pieces that are more than specks: a measure of how broken the strokes are."""
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    ys, xs = np.where(m > 0)
    if len(ys) == 0:
        return 0
    floor = 0.0015 * float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    return int((st[1:, cv2.CC_STAT_AREA] >= floor).sum())


def _fill(m: np.ndarray) -> float:
    ys, xs = np.where(m > 0)
    if len(ys) == 0:
        return 0.0
    return float(m.sum()) / float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))


def _match_width(m: np.ndarray, want: float) -> np.ndarray:
    """Pull a re-thresholded row back to the stroke width it had, so the page weight does not change: erode by
    the excess, then add the skeleton back so thin strokes stay connected."""
    have = stroke_width(m)
    if want <= 0:
        return m
    if have < 0.85 * want:  # too thin: grow to the old weight, then drop one-pixel spurs
        r = int(max(1, round((want - have) / 2.0)))
        g = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        return cv2.morphologyEx(g, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if have <= 1.1 * want:
        return m
    r = int(max(1, round((have - want) / 2.0)))
    er = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return np.maximum(er, cv2.dilate(_skeleton(m), np.ones((3, 3), np.uint8)) * m).astype(np.uint8)


def rebinarize(mask: np.ndarray, flat: np.ndarray, mode: str = "sauvola") -> np.ndarray:
    """Re-extract the row's ink from the grey image with settings made for small, thin script.

    The page-wide mask is tuned for the big digits. For this row alone: estimate its own blur and undo it
    (Wiener), boost local contrast (CLAHE), then threshold with a stroke-scale Sauvola window and hysteresis.
    The old mask only supplies the where (dilated); the new threshold decides the what, so noise outside the
    old ink never comes back and the strokes inside it get sharper, cleaner edges."""
    from opencv import cleanpage as CP

    m = (mask > 0).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if len(ys) < 60:
        return m
    h = int(ys.max() - ys.min() + 1)
    g = flat.astype(np.float32)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = int(0.15 * h)
    a, b, c, d = max(0, y0 - pad), min(g.shape[0], y1 + pad), max(0, x0 - pad), min(g.shape[1], x1 + pad)
    roi = np.clip(g[a:b, c:d], 0, 255)
    sig = CP.estimate_blur_sigma(roi)
    if sig >= 0.9:
        roi = CP.wiener_deblur(roi, min(sig * 0.9, 3.0), k_reg=0.010)
    u8 = np.clip(roi, 0, 255).astype(np.uint8)
    u8 = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(1, max(1, int(np.ceil((b - a) / 48))))).apply(u8)
    f = cv2.GaussianBlur(u8.astype(np.float32), (0, 0), 0.7)
    if mode == "otsu":
        sw = float(np.clip(stroke_width(m), 2.5, 0.12 * h))
        where_roi = cv2.dilate(m[a:b, c:d], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.30 * h) | 1,) * 2))
        den = cv2.fastNlMeansDenoising(np.clip(roi, 0, 255).astype(np.uint8), None, DENOISE_H, 5, 17).astype(np.float32)
        den = cv2.GaussianBlur(den, (0, 0), max(0.6, 0.30 * sw))
        vals = den[where_roi > 0]
        t, _ = cv2.threshold(vals.astype(np.uint8).reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        paper = float(np.percentile(vals, 90))
        rng = max(paper - t, 10.0)
        strong = den < t - OTSU_STRONG * rng
        weak = den < t + OTSU_WEAK * rng
        n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
        ids = np.unique(lab[strong])
        new = np.isin(lab, ids[ids > 0]).astype(np.uint8)
    elif mode == "ridge":
        from opencv import ridge

        sw = float(np.clip(stroke_width(m), 2.5, 0.12 * h))
        new = ridge.ridge_ink(f, sw, RIDGE_LO, RIDGE_HI)
    else:
        win = max(15, int(0.55 * h)) | 1
        ink = cv2.GaussianBlur(CP.sauvola_ink(f, win, k=0.18), (0, 0), 0.5)
        n, lab = cv2.connectedComponents((ink > 0.36).astype(np.uint8), connectivity=8)
        ids = np.unique(lab[ink > 0.62])
        new = np.isin(lab, ids[ids > 0]).astype(np.uint8)
    reach = 0.30 if mode == "otsu" else 0.10  # the denoised Otsu pass may recover strokes the page mask lost
    where = cv2.dilate(m[a:b, c:d], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(reach * h) | 1,) * 2))
    out = np.zeros_like(m)
    out[a:b, c:d] = new & where
    if out.sum() <= 0.35 * m.sum():
        return m
    nys, nxs = np.where(out > 0)
    fill = float(out.sum()) / float((nys.max() - nys.min() + 1) * (nxs.max() - nxs.min() + 1))
    return m if fill > 0.38 else out  # an over-filled result is a blob: keep the old mask and let thin_row work on it


def _zhang_suen(img: np.ndarray) -> np.ndarray:
    """Classic Zhang-Suen thinning, vectorised: one-pixel-wide centre lines that keep the stroke topology."""
    im = np.pad((img > 0).astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p2, p3, p4 = im[:-2, 1:-1], im[:-2, 2:], im[1:-1, 2:]
            p5, p6, p7 = im[2:, 2:], im[2:, 1:-1], im[2:, :-2]
            p8, p9 = im[1:-1, :-2], im[:-2, :-2]
            c = im[1:-1, 1:-1]
            nb = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            a = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            kill = (c == 1) & (nb >= 2) & (nb <= 6) & (a == 1) & cond
            if kill.any():
                im[1:-1, 1:-1][kill] = 0
                changed = True
        if not changed:
            break
    return im[1:-1, 1:-1]


def _endpoints(sk: np.ndarray) -> np.ndarray:
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
    return ((sk > 0) & (nb <= 1)).astype(np.uint8)


def _prune(sk: np.ndarray, n: int) -> np.ndarray:
    """Remove spurs shorter than n pixels: peel end points n times, then grow the surviving ends back n times
    along the original skeleton so genuine stroke ends keep their length."""
    cur = sk.copy()
    for _ in range(n):
        cur = cur & (1 - _endpoints(cur))
    grow = _endpoints(cur)
    k = np.ones((3, 3), np.uint8)
    for _ in range(n):
        grow = cv2.dilate(grow, k) & sk
    return (cur | grow).astype(np.uint8)


def mono_redraw(m: np.ndarray, width: float) -> np.ndarray:
    """Redraw the row as clean connected strokes of one width: smooth, thin to centre lines, prune spurs, redraw."""
    ys, xs = np.where(m > 0)
    if len(ys) < 60:
        return m
    h = int(ys.max() - ys.min() + 1)
    sm = (cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(0.8, 0.012 * h)) > 0.5).astype(np.uint8)
    sk = _zhang_suen(sm)
    if sk.sum() < 20:
        return m
    sk = _prune(sk, max(3, int(0.07 * h)))
    d = int(max(3, round(width))) | 1
    out = cv2.dilate(sk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    return out if out.sum() > 0.4 * m.sum() else m


def bridge_gaps(m: np.ndarray, flat: np.ndarray | None, width: float, reach_frac: float = 0.14) -> np.ndarray:
    """Fill dots and hairline breaks without redrawing anything: every stroke end (and every stray dot) is joined to
    the nearest ink of another piece when the grey image really is dark along the line between them."""
    ys, xs = np.where(m > 0)
    if len(ys) < 60 or flat is None or flat.shape != m.shape:
        return m
    h = int(ys.max() - ys.min() + 1)
    reach = max(4.0, reach_frac * h)
    g = cv2.GaussianBlur(flat.astype(np.float32), (0, 0), 0.8)
    ink_lvl, paper = float(np.percentile(g[m > 0], 40)), float(np.percentile(g[m > 0], 95))
    if paper - ink_lvl < 10:
        return m
    out = m.copy()
    thick = int(max(2, round(0.8 * width)))
    for _ in range(2):
        n, lab, st, cen = cv2.connectedComponentsWithStats(out, connectivity=8)
        if n <= 2:
            break
        pys, pxs = np.where(out > 0)
        plab = lab[pys, pxs]
        sk = _zhang_suen(out)
        ends = _endpoints(sk)
        ey, ex = np.where(ends > 0)
        pts = list(zip(ey, ex))
        areas = st[:, cv2.CC_STAT_AREA]
        small = float(np.median(areas[1:])) * 0.5
        for i in range(1, n):  # a tiny dot is an end point of a stroke too
            if areas[i] <= small:
                pts.append((int(round(cen[i][1])), int(round(cen[i][0]))))
        added = 0
        for (y, x) in pts:
            c = lab[y, x]
            if c == 0:
                continue
            other = plab != c
            if not other.any():
                continue
            d2 = (pys[other] - y) ** 2 + (pxs[other] - x) ** 2
            j = int(np.argmin(d2))
            if d2[j] > reach ** 2 or d2[j] < 4:
                continue
            ty, tx = int(pys[other][j]), int(pxs[other][j])
            k = max(2, int(np.sqrt(d2[j])))
            ly = np.linspace(y, ty, k).round().astype(int)
            lx = np.linspace(x, tx, k).round().astype(int)
            if (paper - float(g[ly, lx].mean())) / (paper - ink_lvl) < 0.55:  # the picture must show ink there
                continue
            cv2.line(out, (int(x), int(y)), (tx, ty), 1, thick, cv2.LINE_8)
            added += 1
        if not added:
            break
    return out


def thin_row(mask: np.ndarray, flat: np.ndarray | None = None, target_frac: float = TARGET_FRAC) -> np.ndarray:
    """Return the row mask at print weight with its counters open. `flat` is the illumination-flattened grey
    (dark ink on light paper) in the same coordinates as `mask`, used to re-open filled counters."""
    m = (mask > 0).astype(np.uint8)
    if m.sum() < 30:
        return m
    stroke_w_ref = stroke_width(m)
    if flat is not None and flat.shape == m.shape and REBIN:
        old_w = stroke_width(m)
        best, best_pieces = m, _pieces(m)
        for mode in MODES:  # keep the candidate with the least broken strokes that is not a blob
            cand = _match_width(rebinarize(m, flat, mode), old_w)
            pc = _pieces(cand)
            if _fill(cand) <= 0.30 and pc <= 0.9 * best_pieces:
                best, best_pieces = cand, pc
        m = best
    ys, xs = np.where(m > 0)
    h = int(ys.max() - ys.min() + 1)
    w0 = stroke_width(m)
    target = max(2.0, target_frac * h)
    box = float(h * (xs.max() - xs.min() + 1))
    fill0 = float(m.sum()) / box  # share of the row's box covered by ink: print Bangla sits near 0.20-0.28
    if w0 <= RATIO * target and fill0 <= FILL_MAX:  # already print weight: never cut it further, only bridge gaps
        gap = max(3, int(0.03 * h)) | 1
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap)))
        m = _strip_frame(m, target)
        if BRIDGE:
            m = bridge_gaps(m, flat, stroke_w_ref)
        return mono_redraw(m, max(3.5, MONO_WIDTH * stroke_w_ref)) if MONO else m
    # bold or blobby row: erode until it reaches print weight and print coverage, but never below 0.8 x target
    r_cap = int(max(1, min(7, round((w0 - 0.8 * target) / 2.0))))
    if fill0 > FILL_MAX:  # blobby (counters clogged by blur): keep going until the coverage is a print level
        r_cap = 8  # safe: the centre lines are put back below, so no stroke can vanish
    out, r = m, 0
    for r in range(1, r_cap + 1):
        out = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        if float(out.sum()) / box <= FILL_GOAL and (w0 - 2 * r <= target or fill0 > FILL_MAX):
            break
    # skeleton pixels keep thin or short strokes alive; a 3x3 grow keeps them connected after smoothing
    keep_lines = cv2.dilate(_zhang_suen(m), np.ones((3, 3), np.uint8)) * m  # centre lines: no stroke may vanish
    out = np.maximum(out, keep_lines)
    if flat is not None and flat.shape == m.shape:
        g = cv2.GaussianBlur(flat.astype(np.float32), (0, 0), 0.8)
        inside = g[out > 0]
        if inside.size > 50:
            core, paper = float(np.percentile(inside, 15)), float(np.percentile(g[m > 0], 90))
            if paper - core > 12:
                lighter = g > core + 0.62 * (paper - core)  # clearly paper-like, although inside the old mask
                out = np.maximum(out & (~lighter).astype(np.uint8), keep_lines)
    if BRIDGE:  # join dots to their strokes before the speck filter below can delete them
        out = bridge_gaps(out, flat, stroke_w_ref)
    # keep every stroke connected to its neighbours' scale: remove specks far below a glyph piece
    n, lab, st, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    if n > 1:
        areas = st[1:, cv2.CC_STAT_AREA]
        floor = max(8.0, 0.06 * float(np.percentile(areas, 80)))
        out = np.isin(lab, 1 + np.where(areas >= floor)[0]).astype(np.uint8)
    out = _strip_frame(_drop_blobs(out, target), target)
    if MONO:
        out = mono_redraw(out, max(3.5, MONO_WIDTH * stroke_w_ref))
    f = cv2.GaussianBlur(out.astype(np.float32), (0, 0), 0.0035 * h + 0.5)
    return (f > 0.45).astype(np.uint8)
