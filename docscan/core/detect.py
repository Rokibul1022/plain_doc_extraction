"""Document, card, plate, and text boundary detection using OpenCV multi-tier heuristics."""

from typing import Literal
import cv2
import numpy as np
from docscan.core.geometry import order_points

DetectTarget = Literal["auto", "plate", "document"]


def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Compute Canny edges with automatic thresholds based on image median."""
    median_val = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median_val))
    upper = int(min(255, (1.0 + sigma) * median_val))
    if lower == upper:
        lower, upper = 40, 160
    return cv2.Canny(gray, lower, upper)


def _measure_boundary_support(
    quad: np.ndarray, dilated_edges: np.ndarray, img_shape: tuple[int, int]
) -> float:
    """Calculate fraction of boundary points coinciding with detected edge lines."""
    h, w = img_shape[:2]
    hits, total = 0, 0
    for i in range(4):
        p1 = quad[i]
        p2 = quad[(i + 1) % 4]
        for t in np.linspace(0.05, 0.95, 25):
            pt = p1 + t * (p2 - p1)
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < w and 0 <= y < h:
                total += 1
                if dilated_edges[y, x] > 0:
                    hits += 1
    return float(hits / max(1, total))


# Smallest region (fraction of frame) accepted from generic detectors; a lone glyph or
# speckle blob falls below this and is rejected in favour of the full frame.
MIN_GENERIC_AREA = 0.04
# Plates can legitimately be tiny in a wide shot, but never smaller than this.
MIN_PLATE_AREA = 0.003


def _quad_area_ratio(quad: np.ndarray, shape: tuple[int, int]) -> float:
    h, w = shape[:2]
    return float(cv2.contourArea(quad.astype(np.float32))) / float(max(1, h * w))


def _body_full(roi: np.ndarray, thr: float, dark_frac: float) -> np.ndarray:
    """Mask of the majority-tone (plate body) pixels in a ROI, given its Otsu split."""
    return (roi < thr) if dark_frac >= 0.5 else (roi >= thr)


def _rectify_with_margin(
    gray: np.ndarray, quad: np.ndarray, mx_frac: float = 0.15, my_frac: float = 0.35,
    max_side: int = 320,
) -> tuple[np.ndarray, tuple[int, int, int, int], float, float] | None:
    """Warp a quad upright (with a margin of surroundings) so it can be analysed in its own
    frame, whatever its rotation or perspective. Returns (canvas, (mx, my, bw, bh), w0, h0)
    where w0/h0 are the quad's true pixel size, or None if it is too small to judge."""
    q = order_points(quad.astype(np.float32))
    w0 = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    h0 = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    if w0 < 24 or h0 < 10:
        return None
    s = min(1.0, max_side / max(w0, h0))
    bw, bh = max(8, int(round(w0 * s))), max(8, int(round(h0 * s)))
    mx, my = int(round(mx_frac * bw)), int(round(my_frac * bh))
    dst = np.float32([[mx, my], [mx + bw - 1, my], [mx + bw - 1, my + bh - 1], [mx, my + bh - 1]])
    m = cv2.getPerspectiveTransform(q, dst)
    canvas = cv2.warpPerspective(gray, m, (bw + 2 * mx, bh + 2 * my), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    return canvas, (mx, my, bw, bh), float(w0), float(h0)


def _plate_likeness(gray: np.ndarray, quad: np.ndarray) -> float:
    """Score 0..1 for how much a quad looks like a plate/label: a clean rectangle that
    stands apart from its surroundings and holds stroke-like text with real structure.
    Measured in the quad's own rectified frame, so rotation and tilt do not matter."""
    quad = quad.astype(np.float32)
    rect = cv2.minAreaRect(quad)
    rw, rh = rect[1]
    if rw < 1 or rh < 1:
        return 0.0
    aspect = max(rw, rh) / min(rw, rh)
    if not (1.0 <= aspect <= 6.5):
        return 0.0
    rectangularity = min(1.0, cv2.contourArea(quad) / (rw * rh))

    rect_info = _rectify_with_margin(gray, quad)
    if rect_info is None:
        return 0.0
    canvas, (mx, my, bw, bh), _, _ = rect_info
    shrink = 2
    roi = canvas[my + shrink:my + bh - shrink, mx + shrink:mx + bw - shrink]
    if roi.size < 60:
        return 0.0
    vals = roi.reshape(-1)

    # Text: a distinct minority of pixels (dark on light or light on dark) against the body.
    thr, _ = cv2.threshold(vals.reshape(-1, 1).astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = vals < thr
    dark_frac = float(np.mean(dark))
    if not (0.07 <= min(dark_frac, 1.0 - dark_frac) <= 0.5):
        return 0.0
    contrast = abs(float(vals[~dark].mean()) - float(vals[dark].mean()))
    if contrast < 40.0:
        return 0.0

    # A plate has a clean, uniform body (the majority tone) - a grille or foliage does not.
    body = ~dark if dark_frac < 0.5 else dark
    body_std = float(vals[body].std())
    uniformity = float(np.clip(1.0 - (body_std - 12.0) / 40.0, 0.0, 1.0))
    if uniformity < 0.2:
        return 0.0

    # ...and several separate, tall characters, not one lone glyph or blob.
    text = (~_body_full(roi, thr, dark_frac)).astype(np.uint8) * 255
    n_cc, _, stats, _ = cv2.connectedComponentsWithStats(text, connectivity=8)
    min_area = max(6, 0.002 * roi.size)
    min_h = 0.22 * roi.shape[0]
    heights = [
        int(stats[i, cv2.CC_STAT_HEIGHT])
        for i in range(1, n_cc)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
        and stats[i, cv2.CC_STAT_HEIGHT] >= min_h
        and stats[i, cv2.CC_STAT_HEIGHT] <= 4.0 * max(1, stats[i, cv2.CC_STAT_WIDTH])
    ]
    if len(heights) < 3:
        return 0.0
    # Characters on one plate are alike in height; a car front or grille is a jumble of sizes.
    med = float(np.median(heights))
    consistent = sum(1 for hgt in heights if abs(hgt - med) <= 0.35 * med)
    if consistent < 3 or consistent < 0.6 * len(heights) or len(heights) > 40:
        return 0.0
    # Plates are overwhelmingly light-bodied with dark text; dark vents/grilles are not.
    tone_fit = 1.0 if float(vals[body].mean()) >= 110.0 else 0.75

    # Stroke structure: horizontal gradient energy, far higher than on a smooth patch.
    gx = np.abs(cv2.Sobel(roi, cv2.CV_16S, 1, 0, ksize=3)).astype(np.float32)
    structure = min(1.0, float(gx.mean()) / 40.0)

    # The region should differ in tone from the ring of surroundings.
    ring_mask = np.ones(canvas.shape, bool)
    ring_mask[my:my + bh, mx:mx + bw] = False
    ring = canvas[ring_mask]
    separation = 0.5
    if ring.size > 30:
        separation = min(1.0, abs(float(np.median(vals)) - float(np.median(ring))) / 50.0)

    aspect_fit = 1.0 if 1.1 <= aspect <= 5.5 else 0.7
    score = (
        0.25 * rectangularity
        + 0.20 * structure
        + 0.20 * separation
        + 0.15 * min(1.0, contrast / 120.0)
        + 0.20 * uniformity
    ) * aspect_fit * tone_fit
    return float(np.clip(score, 0.0, 1.0))


def _text_spills_out(gray: np.ndarray, quad: np.ndarray) -> bool:
    """True if the same plate body and stroke-like text continue just outside the quad,
    i.e. the quad is only a sub-crop of the plate (such as a single row of characters).
    Checked in the quad's rectified frame, so a rotated plate is not mistaken for spill."""
    rect_info = _rectify_with_margin(gray, quad)
    if rect_info is None:
        return False
    canvas, (mx, my, bw, bh), _, _ = rect_info
    inside = canvas[my:my + bh, mx:mx + bw]
    body_tone = float(np.median(inside))
    grad = np.abs(cv2.Sobel(canvas, cv2.CV_16S, 1, 0, ksize=3)).astype(np.float32)
    inner_energy = float(grad[my:my + bh, mx:mx + bw].mean())
    bands = (
        (0, 0, canvas.shape[1], my),                               # above (full width)
        (0, my + bh, canvas.shape[1], canvas.shape[0]),            # below
        (0, my, mx, my + bh),                                      # left
        (mx + bw, my, canvas.shape[1], my + bh),                   # right
    )
    for x0, y0, x1, y1 in bands:
        if (x1 - x0) * (y1 - y0) < 150:
            continue
        band = canvas[y0:y1, x0:x1]
        if abs(float(np.median(band)) - body_tone) > 30.0:
            continue
        if float(grad[y0:y1, x0:x1].mean()) >= 0.6 * inner_energy:
            return True
    return False


def _plate_prior(quad: np.ndarray, shape: tuple[int, int]) -> float:
    """Multiplier (<=1) for how plausible a plate-sized region is, independent of its look."""
    h, w = shape[:2]
    ratio = _quad_area_ratio(quad, shape)
    # A few-pixel box is likelier to be a lamp, sticker or lone character than a plate.
    prior = 0.85 + 0.15 * min(1.0, ratio / 0.05) ** 0.5
    # Very large regions are bodywork, windows or the whole photo.
    prior *= 1.0 if ratio <= 0.6 else max(0.6, 1.0 - (ratio - 0.6))
    # Small regions pinned to the border are background clutter; a tight plate photo
    # legitimately fills the frame, so large ones are exempt.
    touches = (quad[:, 0].min() <= 2 or quad[:, 1].min() <= 2
               or quad[:, 0].max() >= w - 3 or quad[:, 1].max() >= h - 3)
    if touches and ratio < 0.3:
        prior *= 0.6
    return float(prior)


def _find_plate_quad(gray: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Localise a licence plate / label, including small ones in wide shots.

    Builds several foreground masks (global/adaptive/locally-bright/edges), keeps compact
    rectangular blobs of plausible size and aspect, then ranks them by `_plate_likeness`."""
    h, w = gray.shape[:2]
    total = float(h * w)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    masks = []
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(otsu)
    masks.append(cv2.bitwise_not(otsu))
    for blk in (31, 61):
        masks.append(cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, blk, -8))
    tiny = cv2.resize(gray, (max(8, w // 8), max(8, h // 8)), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(tiny, (0, 0), max(h, w) / 96.0), (w, h),
                    interpolation=cv2.INTER_LINEAR)
    masks.append(((blur.astype(np.int16) - bg.astype(np.int16)) > 18).astype(np.uint8) * 255)
    masks.append(cv2.dilate(cv2.Canny(blur, 60, 160), np.ones((3, 3), np.uint8)))

    scored: list[tuple[float, np.ndarray]] = []
    for m in masks:
        for kw, kh in ((15, 5), (7, 3), (25, 9)):
            cl = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)))
            cl = cv2.morphologyEx(cl, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            # RETR_LIST: a plate sits inside the bumper/grille blob, so nested contours matter.
            cnts, _ = cv2.findContours(cl, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:40]:
                area = cv2.contourArea(c)
                ratio = area / total
                if ratio < MIN_PLATE_AREA or ratio > 0.9:
                    continue
                rect = cv2.minAreaRect(c)
                rw, rh = rect[1]
                if min(rw, rh) < 10:
                    continue
                aspect = max(rw, rh) / min(rw, rh)
                if not (1.0 <= aspect <= 6.5) or area / (rw * rh) < 0.72:
                    continue
                approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    quad = approx.reshape(4, 2).astype(np.float32)
                else:
                    quad = cv2.boxPoints(rect).astype(np.float32)
                quad = order_points(quad)
                score = _plate_likeness(gray, quad)
                if score <= 0:
                    continue
                score *= _plate_prior(quad, gray.shape)
                if score >= 0.55:
                    if _text_spills_out(gray, quad):
                        score *= 0.6
                    if score >= 0.55:
                        scored.append((score, quad))
    if not scored:
        return None, 0.0

    # Prefer the innermost strong candidate: a plate sits inside the car front, bumper or photo
    # that also scores well. Sub-crops of a plate (one text row) are already penalised by the
    # spill check, so they do not win this.
    top = max(sc for sc, _ in scored)
    strong = [(sc, q) for sc, q in scored if sc >= 0.9 * top]
    best = min(strong, key=lambda sq: cv2.contourArea(sq[1]))
    return best[1], float(best[0])


def _boundary_confidence(gray: np.ndarray, quad: np.ndarray) -> float:
    """Evidence-based confidence: share of the quad boundary lying on real image edges."""
    edges = cv2.dilate(auto_canny(gray), np.ones((3, 3), np.uint8))
    return _measure_boundary_support(quad, edges, gray.shape)


def _find_best_document_quad(
    gray: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray | None, float]:
    """Search for a large convex paper document quadrilateral covering > 16% of the frame."""
    h, w = gray.shape[:2]
    total_area = float(h * w)
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    closed = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )
    cnts, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, float, np.ndarray]] = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:15]:
        area = cv2.contourArea(c)
        if area < 0.16 * total_area or area > 0.98 * total_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = order_points(approx.reshape(4, 2).astype(np.float32))
            support = _measure_boundary_support(quad, dilated_edges, (h, w))
            area_ratio = area / total_area
            score = support * 0.6 + area_ratio * 0.4
            candidates.append((support, score, quad))

    if not candidates:
        return None, 0.0

    candidates.sort(key=lambda item: (item[0] >= 0.35, item[1]), reverse=True)
    best_support, best_score, best_quad = candidates[0]
    if best_support >= 0.24:
        return best_quad, float(best_score)
    return None, 0.0


def _find_main_context_quad(
    gray: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    """Detect the primary rectangular document, card, plate, or label context with concentrated text."""
    h, w = gray.shape[:2]
    total_area = float(h * w)

    search_cnts: list[np.ndarray] = []
    # Test multiple thresholding levels (both light and inverted masks)
    for thresh in (0, 140, 180, 200):
        flag = cv2.THRESH_BINARY + cv2.THRESH_OTSU if thresh == 0 else cv2.THRESH_BINARY
        _, m = cv2.threshold(gray, thresh, 255, flag)
        cl = cv2.morphologyEx(
            m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        )
        c_list, _ = cv2.findContours(cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        search_cnts.extend(c_list)

        _, minv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cl_inv = cv2.morphologyEx(
            minv, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        )
        c_list_inv, _ = cv2.findContours(cl_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        search_cnts.extend(c_list_inv)

    candidates: list[tuple[float, np.ndarray]] = []
    for c in search_cnts:
        area = cv2.contourArea(c)
        if area < 0.005 * total_area or area > 0.85 * total_area:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), angle = rect
        cand_w, cand_h = max(rw, rh), min(rw, rh)
        if cand_w < 14 or cand_h < 7:
            continue
        aspect = cand_w / max(1.0, cand_h)
        if not (1.0 <= aspect <= 6.5):
            continue

        # Try approxPolyDP first to extract genuine 4-corner perspective quadrilateral
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.025 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = order_points(approx.reshape(4, 2).astype(np.float32))
        else:
            box = cv2.boxPoints(rect)
            quad = order_points(box.astype(np.float32))

        min_x, max_x = np.min(quad[:, 0]), np.max(quad[:, 0])
        min_y, max_y = np.min(quad[:, 1]), np.max(quad[:, 1])
        rx, ry = max(0, int(round(min_x))), max(0, int(round(min_y)))
        rw_b, rh_b = min(w - rx, int(round(max_x - min_x))), min(h - ry, int(round(max_y - min_y)))
        if rw_b < 14 or rh_b < 7:
            continue

        roi = gray[ry : ry + rh_b, rx : rx + rw_b]
        if roi.size == 0:
            continue
        sobel_roi = cv2.convertScaleAbs(cv2.Sobel(roi, cv2.CV_16S, 1, 0, ksize=3))
        text_energy = float(np.mean(sobel_roi))
        if text_energy < 12.0:
            continue

        # Penalize if it touches outer image borders (background / vehicle body / wall)
        border_penalty = 1.0
        if rx <= 3 or (rx + rw_b) >= w - 4:
            border_penalty *= 0.35
        if ry <= 3 or (ry + rh_b) >= h - 4:
            border_penalty *= 0.35

        aspect_bonus = 1.35 if (1.3 <= aspect <= 4.5) else 1.0
        dist_from_center = np.hypot(cx - w / 2, cy - h / 2) / (np.hypot(w / 2, h / 2))
        center_factor = max(0.4, 1.0 - 0.5 * dist_from_center)
        area_factor = min(1.0, (area / total_area) / 0.12)

        score = (
            text_energy
            * (0.5 + 0.5 * area_factor)
            * border_penalty
            * aspect_bonus
            * center_factor
        )
        candidates.append((score, quad))

    if not candidates:
        return None, 0.0

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], 0.88


def _find_contrast_segmented_quad(
    gray: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    """Fallback: detect high-contrast document body via threshold segmentation."""
    h, w = gray.shape[:2]
    total_area = float(h * w)

    for thresh in (0, 190, 80):
        flag = (
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
            if thresh == 0
            else cv2.THRESH_BINARY
        )
        _, mask = cv2.threshold(gray, thresh, 255, flag)
        seg_closed = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
        )
        cnts, _ = cv2.findContours(
            seg_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(c)
            if area < 0.18 * total_area or area > 0.98 * total_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.025 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = order_points(approx.reshape(4, 2).astype(np.float32))
                return quad, 0.65

    return None, 0.0


def _find_text_content_quad(
    gray: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    """Fallback: detect tight bounding box enclosing concentrated text or content."""
    h, w = gray.shape[:2]
    total_area = float(h * w)
    sobelx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3))
    _, tbin = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tclose = cv2.morphologyEx(
        tbin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    )
    cnts, _ = cv2.findContours(tclose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_boxes = []
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw > 15 and bh > 8 and (bw * bh) < 0.85 * total_area:
            valid_boxes.append((bx, by, bx + bw, by + bh))

    if not valid_boxes:
        return None, 0.0

    min_x = min(b[0] for b in valid_boxes)
    min_y = min(b[1] for b in valid_boxes)
    max_x = max(b[2] for b in valid_boxes)
    max_y = max(b[3] for b in valid_boxes)

    # Add small padding margin around text
    pad_x = int(0.04 * (max_x - min_x))
    pad_y = int(0.04 * (max_y - min_y))
    min_x, max_x = max(0, min_x - pad_x), min(w - 1, max_x + pad_x)
    min_y, max_y = max(0, min_y - pad_y), min(h - 1, max_y + pad_y)

    quad = np.array(
        [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]],
        dtype=np.float32,
    )
    return order_points(quad), 0.55


def _valid_quad(quad: np.ndarray) -> bool:
    """A usable boundary is convex, non-degenerate and roughly rectangular in perspective:
    interior angles 50-130 degrees and opposite sides within a 3x length ratio. Rejects the
    bow-ties and slivers that a bad corner ordering can produce."""
    q = order_points(quad.astype(np.float32))
    if not cv2.isContourConvex(q.reshape(-1, 1, 2)) or cv2.contourArea(q) < 100:
        return False
    for i in range(4):
        a, b, c = q[i - 1], q[i], q[(i + 1) % 4]
        v1, v2 = a - b, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 4 or n2 < 4:
            return False
        ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))
        if not (50.0 <= ang <= 130.0):
            return False
    top, bottom = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])
    left, right = np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])
    return max(top, bottom) <= 3.0 * min(top, bottom) and max(left, right) <= 3.0 * min(left, right)


def _paper_score(gray: np.ndarray, quad: np.ndarray) -> float:
    """0..1: does this quad look like a page of text - a bright, fairly uniform body with many
    small ink marks, standing apart from its surroundings?"""
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8))
    vals = gray[inner > 0]
    if vals.size < 400:
        return 0.0

    body_tone = float(np.percentile(vals, 70))
    body = vals[vals >= body_tone - 30.0]
    uniformity = float(np.clip(1.0 - (float(body.std()) - 8.0) / 40.0, 0.0, 1.0))

    ink = ((gray < body_tone - 45.0) & (inner > 0)).astype(np.uint8)
    ink_frac = float(ink.sum()) / float(vals.size)
    n_cc, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    small = sum(1 for i in range(1, n_cc) if 4 <= stats[i, cv2.CC_STAT_AREA] <= 0.02 * vals.size)
    # A page carries many small ink marks; a plate or a photo of one carries only a handful.
    if small < 12:
        return 0.0
    text_rich = float(np.clip(small / 25.0, 0.0, 1.0))
    ink_ok = 1.0 if 0.002 <= ink_frac <= 0.35 else 0.0

    pad = max(6, int(0.05 * min(h, w)))
    ring = cv2.dilate(mask, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
    ring[mask > 0] = 0
    ring_vals = gray[ring > 0]
    separation = 0.5
    if ring_vals.size > 100:
        separation = float(np.clip(abs(float(np.median(vals)) - float(np.median(ring_vals))) / 40.0, 0.0, 1.0))

    return float(ink_ok * (0.35 * text_rich + 0.30 * uniformity + 0.35 * separation))


def _fit_quad(hull: np.ndarray) -> np.ndarray:
    """Fit a 4-corner quad to a convex hull: polygon approximation, else min-area rectangle."""
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.045, 0.065, 0.09):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            return ap.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)


def _find_paper_quad(gray: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Segment a page/card that is brighter or darker than its surroundings, however the
    background looks (wood, gradient, clutter), and return its validated 4-corner quad."""
    h, w = gray.shape[:2]
    total = float(h * w)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Large-sigma blur is slow at full size; compute it on a 1/8 copy and scale back up.
    tiny = cv2.resize(gray, (max(8, w // 8), max(8, h // 8)), interpolation=cv2.INTER_AREA)
    local = cv2.resize(cv2.GaussianBlur(tiny, (0, 0), max(h, w) / 48.0), (w, h),
                       interpolation=cv2.INTER_LINEAR)
    masks = [
        otsu,
        cv2.bitwise_not(otsu),
        ((blur.astype(np.int16) - local.astype(np.int16)) > 6).astype(np.uint8) * 255,
    ]
    dilated_edges = cv2.dilate(auto_canny(gray), np.ones((3, 3), np.uint8))

    best: tuple[float, np.ndarray] | None = None
    for m in masks:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:4]:
            ratio = cv2.contourArea(c) / total
            if ratio < 0.12 or ratio > 0.985:
                continue
            hull = cv2.convexHull(c)
            quad = _fit_quad(hull)
            if not _valid_quad(quad):
                continue
            quad = order_points(quad)
            fill = cv2.contourArea(hull) / max(1.0, cv2.contourArea(quad))
            if not (0.8 <= fill <= 1.12):
                continue
            support = _measure_boundary_support(quad, dilated_edges, (h, w))
            paper = _paper_score(gray, quad)
            score = 0.55 * paper + 0.30 * support + 0.15 * min(1.0, ratio / 0.5)
            if paper >= 0.55 and (best is None or score > best[0]):
                best = (score, quad)
    if best is None or best[0] < 0.55:
        return None, 0.0
    return best[1], float(best[0])


def detect_document(
    image: np.ndarray,
    det_h: int = 600,
    target: DetectTarget = "auto",
) -> tuple[np.ndarray, float, bool]:
    """Detect boundaries of the primary document or text context in the image.

    Args:
        image: Original full-resolution BGR image.
        det_h: Standardized height for detection copy.
        target: Target hint ('auto', 'document', or 'plate').

    Returns:
        Tuple of (corners [TL, TR, BR, BL], confidence [0.0..1.0], detected: bool).
    """
    orig_h, orig_w = image.shape[:2]

    if orig_h > det_h:
        scale = float(orig_h) / float(det_h)
        det_w = max(10, int(round(orig_w / scale)))
        small = cv2.resize(image, (det_w, det_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        small = image.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small.copy()
    if float(np.std(gray)) < 5.0:
        full_frame = np.array(
            [
                [0.0, 0.0],
                [float(orig_w - 1), 0.0],
                [float(orig_w - 1), float(orig_h - 1)],
                [0.0, float(orig_h - 1)],
            ],
            dtype=np.float32,
        )
        return order_points(full_frame), 0.0, False

    def finish(quad: np.ndarray, conf: float) -> tuple[np.ndarray, float, bool]:
        if not _valid_quad(quad) and _quad_area_ratio(quad, gray.shape) >= MIN_GENERIC_AREA:
            quad = cv2.boxPoints(cv2.minAreaRect(quad.astype(np.float32))).astype(np.float32)
        return order_points(quad * scale), round(float(np.clip(conf, 0.0, 1.0)), 3), True

    def sane(quad: np.ndarray | None) -> bool:
        return quad is not None and _quad_area_ratio(quad, gray.shape) >= MIN_GENERIC_AREA

    def plate_pick() -> tuple[np.ndarray, float] | None:
        """Best plate-like region from the plate localiser or the text-context finder."""
        quad, score = _find_plate_quad(gray)
        if quad is None:
            # Fallback: the text-context finder, held to the same size/border priors.
            cq, _ = _find_main_context_quad(gray)
            if cq is not None and _quad_area_ratio(cq, gray.shape) >= MIN_PLATE_AREA:
                cs = _plate_likeness(gray, cq) * _plate_prior(cq, gray.shape)
                if cs >= 0.55:
                    quad, score = cq, cs
        if quad is None:
            return None
        return quad, 0.5 * score + 0.5 * _boundary_confidence(gray, quad)

    # Targeted detection branch: plate / text card prioritised
    if target == "plate":
        pick = plate_pick()
        if pick is not None:
            return finish(*pick)

    # 1. Edge-based paper document quad (e.g. invoice, receipt, notes, certificate)
    edges = auto_canny(gray)
    doc_quad, doc_score = _find_best_document_quad(gray, edges)
    if doc_quad is not None and doc_score >= 0.35:
        # A big rectangle is only a page if it carries page-like text. A photo frame or white
        # mount around a plate does not - there the plate inside is the real target.
        if target == "plate" or _paper_score(gray, doc_quad) >= 0.4:
            return finish(doc_quad, min(1.0, max(0.5, doc_score)))
        inner_plate, inner_score = _find_plate_quad(gray)
        if inner_plate is not None and inner_score >= 0.7:
            return finish(inner_plate, 0.5 * inner_score + 0.5 * _boundary_confidence(gray, inner_plate))
        fallback_doc = (doc_quad, doc_score)
    else:
        fallback_doc = None

    # 1b. Page/card segmented from any background (dark wood, gradients, clutter)
    if target != "plate":
        paper_quad, paper_score = _find_paper_quad(gray)
        if paper_quad is not None:
            # A strong plate inside a region that is only weakly page-like (a gradient, a
            # bumper, a photo mount) is the real text area, not the region around it.
            inner_plate, inner_score = _find_plate_quad(gray)
            if (
                inner_plate is not None
                and inner_score >= 0.8
                and _quad_area_ratio(inner_plate, gray.shape) < 0.5 * _quad_area_ratio(paper_quad, gray.shape)
                and _paper_score(gray, paper_quad) < 0.75
            ):
                return finish(inner_plate, 0.5 * inner_score + 0.5 * _boundary_confidence(gray, inner_plate))
            return finish(paper_quad, min(1.0, 0.35 + 0.65 * paper_score))

    # 2. Plate / label, including small ones in wide shots (e.g. a car photo)
    if target != "plate":
        pick = plate_pick()
        if pick is not None:
            return finish(*pick)

    # 3. Main context: card / label with concentrated text
    context_quad, _ = _find_main_context_quad(gray)
    if sane(context_quad):
        return finish(context_quad, 0.6 * _boundary_confidence(gray, context_quad) + 0.1)

    # 4. Contrast segmentation fallback
    seg_quad, _ = _find_contrast_segmented_quad(gray)
    if sane(seg_quad):
        return finish(seg_quad, 0.5 * _boundary_confidence(gray, seg_quad) + 0.1)

    # 5. Text content bounding quad fallback
    txt_quad, _ = _find_text_content_quad(gray)
    if sane(txt_quad):
        return finish(txt_quad, 0.4 * _boundary_confidence(gray, txt_quad) + 0.1)

    # 5b. A big non-page rectangle found earlier beats giving up
    if fallback_doc is not None and _valid_quad(fallback_doc[0]):
        return finish(fallback_doc[0], min(1.0, max(0.4, fallback_doc[1])) * 0.8)

    # 6. Fallback to full frame
    full_frame = np.array(
        [
            [0.0, 0.0],
            [float(orig_w - 1), 0.0],
            [float(orig_w - 1), float(orig_h - 1)],
            [0.0, float(orig_h - 1)],
        ],
        dtype=np.float32,
    )
    return order_points(full_frame), 0.0, False
