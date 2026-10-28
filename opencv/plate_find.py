"""Classical plate finding and straightening - OpenCV + NumPy only.

    quad = crop_mode_quad(img)              # colour-cluster hull of a tight plate crop
    corners = subpixel_corners(img, quad)   # snap corners to the picture
    plate = rectify(img, corners)           # perspective-correct to the plate's true aspect
    img, (dx, dy) = trim_letterbox(img)     # cut black padding bars first
"""

import cv2
import numpy as np

from opencv.detect import _find_plate_quad, _valid_quad
from opencv.geometry import order_points

COMMON_ASPECTS = (1.3, 2.0, 2.5, 3.2, 4.0, 4.7, 5.2)


def _iou_box(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def refine_quad(img: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, bool]:
    """Find the plate's tilted corners inside a detector box. Returns (quad, refined?)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = int(0.18 * bw) + 2, int(0.30 * bh) + 2
    rx1, ry1, rx2, ry2 = max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)
    roi = img[ry1:ry2, rx1:rx2]
    rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    if roi.size == 0:
        return rect, False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    s = 1.0
    if gray.shape[1] < 240:
        s = 240.0 / gray.shape[1]
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    quad, score = _find_plate_quad(gray)
    if quad is None:
        return rect, False
    quad = quad / s + np.array([rx1, ry1], np.float32)
    quad = order_points(quad.astype(np.float32))
    # accept only a quad that agrees with the detector box
    qx1, qy1 = quad.min(axis=0)
    qx2, qy2 = quad.max(axis=0)
    if _iou_box((qx1, qy1, qx2, qy2), box) < 0.5 or cv2.contourArea(quad) < 0.45 * bw * bh or not _valid_quad(quad):
        return rect, False
    return quad, True


def subpixel_corners(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Snap each corner to the strongest nearby sub-pixel corner (bounded shift)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    pts = quad.reshape(-1, 1, 2).astype(np.float32).copy()
    win = max(2, int(0.04 * np.linalg.norm(quad[1] - quad[0])))
    if (pts[:, 0, 0] < win + 2).any() or (pts[:, 0, 0] > w - win - 3).any() or \
       (pts[:, 0, 1] < win + 2).any() or (pts[:, 0, 1] > h - win - 3).any():
        return quad
    refined = pts.copy()
    try:
        cv2.cornerSubPix(gray, refined, (win, win), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.03))
    except cv2.error:
        return quad
    shift = np.linalg.norm(refined - pts, axis=2).ravel()
    keep = np.where(shift[:, None] <= max(2.0, 0.5 * win), refined[:, 0, :], pts[:, 0, :])
    return keep.astype(np.float32)


def true_aspect(quad: np.ndarray, shape: tuple[int, int]) -> float:
    """Plate width/height ratio from the quad's homography (assumes a ~50-degree FOV camera),
    snapped to a common plate aspect when close. Falls back to the measured edge ratio."""
    q = order_points(quad)
    w_edge = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    h_edge = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    measured = w_edge / max(h_edge, 1e-3)
    ih, iw = shape[:2]
    f = 1.1 * max(iw, ih)
    k = np.array([[f, 0, iw / 2.0], [0, f, ih / 2.0], [0, 0, 1.0]], np.float64)
    try:
        hmat = cv2.getPerspectiveTransform(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32), q)
        m = np.linalg.inv(k) @ hmat
        a1, a2 = np.linalg.norm(m[:, 0]), np.linalg.norm(m[:, 1])
        est = a1 / max(a2, 1e-6)
    except (np.linalg.LinAlgError, cv2.error):
        est = measured
    if not (1.0 <= est <= 6.0) or abs(np.log(est / measured)) > 0.45:
        est = measured
    nearest = min(COMMON_ASPECTS, key=lambda a: abs(np.log(a / est)))
    return float(nearest if abs(np.log(nearest / est)) < 0.10 else est)


def rectify(img: np.ndarray, quad: np.ndarray, *, pad: float = 0.05, max_side: int | None = None) -> np.ndarray:
    """Perspective-correct the plate to its true aspect with a small margin around it."""
    q = order_points(quad.astype(np.float32))
    aspect = true_aspect(q, img.shape)
    wpx = max(16.0, 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])))
    hpx = max(8.0, wpx / aspect)
    out_w, out_h = int(round(wpx)), int(round(hpx))
    mx, my = int(round(out_w * pad)), int(round(out_h * pad * 1.6))
    dst = np.float32([[mx, my], [mx + out_w - 1, my], [mx + out_w - 1, my + out_h - 1], [mx, my + out_h - 1]])
    m = cv2.getPerspectiveTransform(q, dst)
    out = cv2.warpPerspective(img, m, (out_w + 2 * mx, out_h + 2 * my), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    if max_side and max(out.shape[:2]) > max_side:
        s = max_side / max(out.shape[:2])
        out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return out


def crop_mode_quad(img: np.ndarray) -> np.ndarray | None:
    """For a tight plate crop (the plate fills the frame): find the tilted plate body inside it.

    Colour-clusters the crop and takes the cluster whose largest blob is big and compact - the
    plate body, whatever its colour (white, yellow, teal) - then fits a quad to its convex hull.
    Characters and border are other clusters but sit inside the hull, so they do not matter."""
    h, w = img.shape[:2]
    s = 160.0 / max(h, w) if max(h, w) > 160 else 1.0
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    sh, sw = small.shape[:2]
    if min(sh, sw) < 16:
        return None
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (3, 3), 0), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    best = None
    for k in (3, 4):
        cv2.setRNGSeed(1234)  # k-means seeds from OpenCV's RNG: fix it so the same image always gives the same corners
        _, labels, _ = cv2.kmeans(lab, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0),
                                  2, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(sh, sw)
        for c in range(k):
            m = (labels == c).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            n, lab_img, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i in range(1, n):
                area = st[i, cv2.CC_STAT_AREA] / float(sh * sw)
                if not 0.22 <= area <= 0.97:
                    continue
                cnts, _ = cv2.findContours((lab_img == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
                peri = cv2.arcLength(hull, True)
                quad = None
                for eps in (0.02, 0.035, 0.05, 0.08):
                    ap = cv2.approxPolyDP(hull, eps * peri, True)
                    if len(ap) == 4:
                        quad = ap.reshape(4, 2).astype(np.float32)
                        break
                if quad is None:
                    quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
                quad = order_points(quad)
                if not _valid_quad(quad):
                    continue
                qa = cv2.contourArea(quad)
                fill = cv2.contourArea(hull) / max(qa, 1.0)
                cx, cy = cen[i]
                centred = 1.0 - min(1.0, np.hypot(cx / sw - 0.5, cy / sh - 0.5))
                score = min(area / max(qa / (sh * sw), 1e-3), 1.0) * (1.0 if 0.85 <= fill <= 1.12 else 0.6) * (0.6 + 0.4 * centred) * np.sqrt(qa / (sh * sw))
                if best is None or score > best[0]:
                    best = (score, quad)
    if best is None:
        return None
    return (best[1] / s).astype(np.float32)


def trim_letterbox(img: np.ndarray, thresh: int = 6) -> tuple[np.ndarray, tuple[int, int]]:
    """Cut pure-black padding bars (from letterboxed/resized crops); returns (image, (dx, dy))."""
    g = img if img.ndim == 2 else img.max(axis=2)
    cols = np.where((g > thresh).sum(axis=0) > max(2, 0.03 * g.shape[0]))[0]
    rows = np.where((g > thresh).sum(axis=1) > max(2, 0.03 * g.shape[1]))[0]
    if len(cols) == 0 or len(rows) == 0:
        return img, (0, 0)
    x0, x1, y0, y1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
    if (x1 - x0) * (y1 - y0) > 0.97 * g.shape[0] * g.shape[1] or (x1 - x0) < 24 or (y1 - y0) < 12:
        return img, (0, 0)
    return img[y0:y1, x0:x1], (int(x0), int(y0))
