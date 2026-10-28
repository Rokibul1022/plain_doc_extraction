"""Edge-verified plate quad - OpenCV + NumPy only.

The colour-cluster hull finds roughly where the plate is, but its blob can merge with similarly coloured background
(car body, padding) and the hull then spills past the plate. This module checks a quad against the picture:

  edge_support(quad)  : for each side, how strongly the colour just inside differs from just outside, along the
                        whole side (a real plate border is a long, consistent colour step; background is not);
  candidates(img)     : many quads from colour clusters (several k, all big blobs, hull / approx / min-area rect);
  refine(img, quad)   : coordinate ascent on the four corners to maximise edge support, with a bounded move and
                        an area prior so the quad cannot shrink onto the letters.
"""

import cv2
import numpy as np

from opencv.geometry import order_points

BODY_W = 0.5
LEAK_W = 0.9
_BODY: np.ndarray | None = None  # plate body colour for the image being fitted (set by best_quad)
EXTEND = 0.25  # corners may lie this far (share of the image size) beyond the frame: plates cut off by the picture
WORK = 320  # longest side the search runs at


def _prep(img: np.ndarray):
    s = WORK / max(img.shape[:2])
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    return small, lab, s


def _valid(q: np.ndarray, shape) -> bool:
    if not cv2.isContourConvex(q.reshape(-1, 1, 2).astype(np.float32)):
        return False
    area = cv2.contourArea(q.astype(np.float32))
    if area < 0.12 * shape[0] * shape[1]:
        return False
    w = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    h = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    return 1.1 <= w / max(h, 1e-3) <= 6.0


def _sample(lab: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h, w = lab.shape[:2]
    x = np.clip(pts[:, 0], 0, w - 1.001)
    y = np.clip(pts[:, 1], 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return (lab[y0, x0] * (1 - fx) * (1 - fy) + lab[y0, x0 + 1] * fx * (1 - fy)
            + lab[y0 + 1, x0] * (1 - fx) * fy + lab[y0 + 1, x0 + 1] * fx * fy)


def side_support(lab: np.ndarray, p: np.ndarray, q: np.ndarray, centre: np.ndarray, off: float,
                 body: np.ndarray | None = None) -> tuple[float, float]:
    """(mean colour step, share of the side that has a clear step) across side p->q, outside vs inside."""
    h, w = lab.shape[:2]
    n = int(max(8, np.linalg.norm(q - p) / 2))
    t = np.linspace(0.04, 0.96, n)[:, None]
    pts = p[None] * (1 - t) + q[None] * t
    d = q - p
    nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    if np.dot(nrm, pts.mean(0) - centre) < 0:
        nrm = -nrm  # point outward
    out_pts, in_pts = pts + nrm * off, pts - nrm * off
    on_frame = (out_pts[:, 0] < 0) | (out_pts[:, 0] > w - 1) | (out_pts[:, 1] < 0) | (out_pts[:, 1] > h - 1)
    if on_frame.mean() > 0.6:  # the side lies on the frame: good only if what is just inside still looks like plate body
        if body is None:
            return 12.0, 0.5
        dist = np.linalg.norm(_sample(lab, in_pts) - body, axis=1)
        ok = float((dist < 22.0).mean())
        return 4.0 + 24.0 * ok, 0.1 + 0.75 * ok
    a, b = _sample(lab, out_pts), _sample(lab, in_pts)
    step = np.linalg.norm(a - b, axis=1)
    step[on_frame] = 28.0
    return float(step.mean()), float((step > 10.0).mean())


def _body_colour(lab: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Median Lab colour of the quad's middle (the plate body; letters are a minority)."""
    c = q.mean(0)
    inner = (c + (q - c) * 0.6).astype(np.int32)
    m = np.zeros(lab.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, inner, 1)
    px = lab[m > 0]
    return np.median(px, axis=0) if len(px) else np.array([128.0, 128.0, 128.0])


def body_share(lab: np.ndarray, q: np.ndarray, body: np.ndarray) -> float:
    """Share of the quad's in-frame pixels that look like plate body; background spill lowers it."""
    m = np.zeros(lab.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, np.round(q).astype(np.int32), 1)
    px = lab[m > 0]
    if len(px) == 0:
        return 0.0
    return float((np.linalg.norm(px - body, axis=1) < 26.0).mean())


def side_leak(lab: np.ndarray, p: np.ndarray, q: np.ndarray, centre: np.ndarray, off: float, body: np.ndarray) -> float:
    """Share of the band just OUTSIDE this side that still looks like plate body: high means the side cuts through
    the plate (e.g. across the digits) instead of lying on its border."""
    h, w = lab.shape[:2]
    n = int(max(8, np.linalg.norm(q - p) / 2))
    t = np.linspace(0.08, 0.92, n)[:, None]
    pts = p[None] * (1 - t) + q[None] * t
    d = q - p
    nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    if np.dot(nrm, pts.mean(0) - centre) < 0:
        nrm = -nrm
    hits, tot = 0, 0
    for k in (2.0, 3.5, 5.0):
        o = pts + nrm * off * k
        inside = (o[:, 0] >= 0) & (o[:, 0] <= w - 1) & (o[:, 1] >= 0) & (o[:, 1] <= h - 1)
        if inside.any():
            dist = np.linalg.norm(_sample(lab, o[inside]) - body, axis=1)
            hits += int((dist < 24.0).sum())
            tot += int(inside.sum())
    return hits / tot if tot else 0.0


def edge_support(lab: np.ndarray, q: np.ndarray, off: float) -> float:
    c = q.mean(0)
    body = _BODY if _BODY is not None else _body_colour(lab, q)
    tot = 0.0
    for i in range(4):
        m, f = side_support(lab, q[i], q[(i + 1) % 4], c, off, body)
        tot += min(m, 40.0) / 40.0 * 0.5 + f * 0.5 - LEAK_W * side_leak(lab, q[i], q[(i + 1) % 4], c, off, body)
    return tot / 4.0 + BODY_W * body_share(lab, q, body)


def candidates(img: np.ndarray):
    small, lab, s = _prep(img)
    sh, sw = small.shape[:2]
    feats = lab.reshape(-1, 3)
    quads = []
    for k in (2, 3, 4, 5):
        cv2.setRNGSeed(1234)
        _, labels, _ = cv2.kmeans(feats, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0),
                                  2, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(sh, sw)
        for c in range(k):
            m = cv2.morphologyEx((labels == c).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            n, li, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i in range(1, n):
                if not 0.15 <= st[i, cv2.CC_STAT_AREA] / float(sh * sw) <= 0.98:
                    continue
                cnts, _ = cv2.findContours((li == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
                peri = cv2.arcLength(hull, True)
                shapes = [cv2.boxPoints(cv2.minAreaRect(hull))]
                for eps in (0.02, 0.04, 0.07):
                    ap = cv2.approxPolyDP(hull, eps * peri, True)
                    if len(ap) == 4:
                        shapes.append(ap.reshape(4, 2))
                        break
                for sq in shapes:
                    sq = order_points(sq.astype(np.float32))
                    if _valid(sq, (sh, sw)):
                        quads.append(sq)
    # the whole frame is always a candidate (tight crops where the plate fills the picture)
    quads.append(order_points(np.float32([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]])))
    return quads, lab, s


def seeded_candidates(lab: np.ndarray) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Grow the plate body from the middle of the crop: body colour = majority colour of the central patch, region =
    connected pixels of that colour (letter holes filled), quad = fitted to the region. Unlike free colour
    clustering it cannot wander into background of a different colour that merely shares a cluster."""
    H, W = lab.shape[:2]
    cy0, cy1, cx0, cx1 = int(0.35 * H), int(0.65 * H), int(0.30 * W), int(0.70 * W)
    patch = lab[cy0:cy1, cx0:cx1].reshape(-1, 3)
    if len(patch) < 30:
        return [], None
    L = patch[:, 0].astype(np.uint8).reshape(-1, 1)
    thr, _ = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hi = patch[:, 0] > thr
    core = lab[int(0.42 * H): int(0.58 * H), int(0.42 * W): int(0.58 * W)].reshape(-1, 3)
    core_hi = float((core[:, 0] > thr).mean())  # the plate sits at the centre of these crops: its class owns the core
    pick = hi if core_hi >= 0.5 else ~hi
    if pick.sum() < 20:
        pick = hi if hi.sum() >= (~hi).sum() else ~hi
    body = np.median(patch[pick], axis=0)
    dist = np.linalg.norm(lab - body, axis=2)
    out = []
    for T in (16.0, 24.0, 34.0):
        m = (dist < T).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, int(0.05 * W)) | 1,) * 2))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, li, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            continue
        # component holding most of the central patch
        cnt = np.bincount(li[cy0:cy1, cx0:cx1].ravel(), minlength=n)
        cnt[0] = 0
        i = int(np.argmax(cnt))
        if cnt[i] < 0.15 * (cy1 - cy0) * (cx1 - cx0):
            continue
        cnts, _ = cv2.findContours((li == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        peri = cv2.arcLength(hull, True)
        shapes = [cv2.boxPoints(cv2.minAreaRect(hull))]
        for eps in (0.02, 0.04, 0.07):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                shapes.append(ap.reshape(4, 2))
                break
        for sq in shapes:
            sq = order_points(sq.astype(np.float32))
            if _valid(sq, (H, W)):
                out.append(sq)
    return out, body


def text_candidates(small: np.ndarray) -> list[np.ndarray]:
    """Quads anchored on the characters: the tilted box of the biggest cluster of thin strokes (dark-on-light and
    light-on-dark), grown to plate margins. Characters are the most reliable thing on a plate, whatever its colour."""
    sh, sw = small.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (0, 0), 0.8)
    k = max(9, int(0.11 * sh) | 1)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = []
    for resp in (cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, el), cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, el)):
        resp = resp.astype(np.float32)
        top = float(np.percentile(resp, 99.0))
        if top < 14.0:
            continue
        m = (resp > 0.25 * top).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, int(0.07 * sw)) | 1, max(3, int(0.05 * sh)) | 1)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            continue
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        if st[i, cv2.CC_STAT_AREA] < 0.06 * sh * sw:
            continue
        ys, xs = np.where(lab == i)
        (cx, cy), (w, h), ang = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        th = np.deg2rad(ang)
        if w < h:
            w, h, th = h, w, th + np.pi / 2
        for gw, gh in ((1.22, 1.45), (1.12, 1.30)):
            out.append(order_points(_rect_pts(cx, cy, w * gw, h * gh, th)))
    return out


def refine(lab: np.ndarray, quad: np.ndarray, off: float, move_frac: float = 0.22) -> np.ndarray:
    """Move each corner in a small window to raise the edge support of its two sides."""
    q = quad.copy()
    base_area = cv2.contourArea(q.astype(np.float32))
    span = np.sqrt(base_area)
    lim = move_frac * span
    q0 = quad.copy()

    def score(qq):
        if not cv2.isContourConvex(qq.reshape(-1, 1, 2).astype(np.float32)):
            return -1.0
        a = cv2.contourArea(qq.astype(np.float32)) / base_area
        # area prior: do not collapse onto interior letter edges, do not balloon either
        return edge_support(lab, qq, off) + 0.10 * min(a, 1.0) - 0.20 * max(a - 1.25, 0.0)

    best = score(q)
    for step in (0.06 * span, 0.03 * span, 0.015 * span):
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                    c = q.copy()
                    c[i] = q[i] + np.array([dx, dy]) * step
                    c[i] = np.clip(c[i], -EXTEND * np.array([lab.shape[1], lab.shape[0]]), (1 + EXTEND) * np.array([lab.shape[1], lab.shape[0]]))
                    if np.linalg.norm(c[i] - q0[i]) > lim:
                        continue
                    sc = score(c)
                    if sc > best + 1e-4:
                        q, best, improved = c, sc, True
    return q


def _rect_pts(cx, cy, w, h, th) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    d = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    r = np.array([[c, -s], [s, c]])
    return (d @ r.T + [cx, cy]).astype(np.float32)


def fit_rect(lab: np.ndarray, quad: np.ndarray, off: float) -> np.ndarray:
    """Best rotated rectangle near `quad` (5 parameters: centre, size, angle), by coordinate ascent on edge support.
    Few parameters keep it from drifting into odd shapes; it may extend past the frame for cut-off plates."""
    (cx, cy), (w, h), ang = cv2.minAreaRect(quad.astype(np.float32))
    th = np.deg2rad(ang)
    if w < h:
        w, h, th = h, w, th + np.pi / 2
    H, W = lab.shape[:2]
    area0 = w * h

    def sc(p):
        cx, cy, w, h, th = p
        if w < 8 or h < 6 or not 1.2 <= w / h <= 6.0 or abs(th) > 0.7:
            return -9.0
        if abs(cx) > W * 1.2 or abs(cy) > H * 1.2:
            return -9.0
        q = _rect_pts(cx, cy, w, h, th)
        if (q < -EXTEND * np.array([W, H])).any() or (q > (1 + EXTEND) * np.array([W, H])).any():
            return -9.0
        a = w * h / area0
        return edge_support(lab, q, off) + 0.10 * min(a, 1.0) - 0.20 * max(a - 1.9, 0.0)

    th = (th + np.pi / 2) % np.pi - np.pi / 2
    p = np.array([cx, cy, w, h, th], np.float64)
    best = sc(p)
    span = np.sqrt(area0)
    for k in (0.08, 0.04, 0.02, 0.01):
        steps = np.array([k * span, k * span, k * span, k * span, k * 0.8])
        improved = True
        while improved:
            improved = False
            for j in range(5):
                for sgn in (1, -1):
                    c = p.copy()
                    c[j] += sgn * steps[j]
                    v = sc(c)
                    if v > best + 1e-4:
                        p, best, improved = c, v, True
    return _rect_pts(*p)


def _line_from(p, q):
    """(angle of the unit normal, offset) of the infinite line through p, q."""
    d = q - p
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    return float(np.arctan2(n[1], n[0])), float(n @ p)


def _intersect(l1, l2):
    a1, r1 = l1
    a2, r2 = l2
    A = np.array([[np.cos(a1), np.sin(a1)], [np.cos(a2), np.sin(a2)]])
    if abs(np.linalg.det(A)) < 1e-3:
        return None
    return np.linalg.solve(A, [r1, r2])


def _quad_from_lines(L):
    pts = [_intersect(L[0], L[3]), _intersect(L[0], L[1]), _intersect(L[2], L[1]), _intersect(L[2], L[3])]
    if any(p is None for p in pts):
        return None
    return np.array(pts, np.float32)  # TL, TR, BR, BL for lines top, right, bottom, left


def _angle_gap(a, b):
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return abs(d)


def fit_lines(lab: np.ndarray, quad: np.ndarray, off: float, move: float = 0.55) -> np.ndarray:
    """Optimise the four border lines one at a time (offset and angle), keeping the shape plate-like:
    opposite sides near parallel, adjacent sides near perpendicular. Each line is judged along its whole length."""
    q0 = order_points(quad.astype(np.float32))
    H, W = lab.shape[:2]
    span = np.sqrt(cv2.contourArea(q0))
    L = [_line_from(q0[i], q0[(i + 1) % 4]) for i in range(4)]  # top, right, bottom, left (outward normals not needed)
    L0 = list(L)
    base = cv2.contourArea(q0)
    lo, hi = -EXTEND * np.array([W, H]), (1 + EXTEND) * np.array([W, H])

    def ok_shape(Ls):
        if _angle_gap(Ls[0][0], Ls[2][0] + np.pi) > 0.30 and _angle_gap(Ls[0][0], Ls[2][0]) > 0.30:
            return False
        if _angle_gap(Ls[1][0], Ls[3][0] + np.pi) > 0.30 and _angle_gap(Ls[1][0], Ls[3][0]) > 0.30:
            return False
        for i in range(4):
            g = _angle_gap(Ls[i][0], Ls[(i + 1) % 4][0]) % np.pi
            if abs(g - np.pi / 2) > 0.55:
                return False
        return True

    def sc(Ls):
        if not ok_shape(Ls):
            return -9.0
        qq = _quad_from_lines(Ls)
        if qq is None or (qq < lo).any() or (qq > hi).any():
            return -9.0
        qq = order_points(qq)
        if not cv2.isContourConvex(qq.reshape(-1, 1, 2)):
            return -9.0
        a = cv2.contourArea(qq) / base
        if a < 0.5:
            return -9.0
        return edge_support(lab, qq, off) + 0.10 * min(a, 1.0) - 0.20 * max(a - 1.9, 0.0)

    best = sc(L)
    for ks in (0.12, 0.06, 0.03, 0.015, 0.008):
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for dr, da in ((ks * span, 0), (-ks * span, 0), (0, ks * 0.9), (0, -ks * 0.9)):
                    c = list(L)
                    c[i] = (L[i][0] + da, L[i][1] + dr)
                    if abs(c[i][1] - L0[i][1]) > move * span or _angle_gap(c[i][0], L0[i][0]) > 0.3:
                        continue
                    v = sc(c)
                    if v > best + 1e-4:
                        L, best, improved = c, v, True
    out = _quad_from_lines(L)
    return order_points(out) if out is not None else q0


def _global_body(small: np.ndarray, lab: np.ndarray, tq: list[np.ndarray]) -> np.ndarray | None:
    """Plate body colour from the character region: median colour of its pixels that are not strokes."""
    if not tq:
        return None
    q = tq[0] if len(tq) < 2 else tq[1]
    m = np.zeros(lab.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, np.round(q).astype(np.int32), 1)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    px = lab[m > 0]
    g = gray[m > 0].astype(np.float32)
    if len(px) < 50:
        return None
    # the body is the larger of the two grey populations (strokes are the minority)
    thr, _ = cv2.threshold(g.astype(np.uint8).reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hi = g > thr
    pick = hi if hi.sum() >= (~hi).sum() else ~hi
    return np.median(px[pick], axis=0)


def best_quad(img: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Best edge-verified quad (in input-image coordinates) and its support score."""
    global _BODY
    cands, lab, s = candidates(img)
    if not cands:
        return None
    small = cv2.resize(img, (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    seeded, sbody = seeded_candidates(lab)
    _BODY = sbody
    good = [q for q in seeded if body_share(lab, q, sbody) >= 0.5 and cv2.contourArea(q) >= 0.12 * lab.shape[0] * lab.shape[1]]
    cands = good if good else cands + seeded  # the segmented plate body is the most reliable anchor: trust it first
    tight = bool(good)
    off = max(1.5, 0.012 * max(lab.shape[:2]))
    scored = []
    for q in cands:
        a = cv2.contourArea(q.astype(np.float32)) / float(lab.shape[0] * lab.shape[1])
        scored.append((edge_support(lab, q, off) + 0.25 * a, q))
    scored.sort(key=lambda t: -t[0])
    top = []
    for sc, q in scored[:6]:  # refine the best few, keep the winner after refinement
        r = fit_lines(lab, q, off, 0.12 if tight else 0.55)
        a = cv2.contourArea(r.astype(np.float32)) / float(lab.shape[0] * lab.shape[1])
        top.append((edge_support(lab, r, off) + 0.25 * a, r))
    sc, q = max(top, key=lambda t: t[0])
    _BODY = None
    return (q / s).astype(np.float32), float(sc)
