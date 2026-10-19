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

    # Targeted detection branch: plate / text card prioritized
    if target == "plate":
        ctx_quad, ctx_score = _find_main_context_quad(gray)
        if ctx_quad is not None:
            return order_points(ctx_quad * scale), round(ctx_score, 3), True

    # 1. Edge-based paper document quad (e.g. invoice, receipt, notes, certificate)
    edges = auto_canny(gray)
    doc_quad, doc_score = _find_best_document_quad(gray, edges)
    if doc_quad is not None and doc_score >= 0.35:
        scaled_quad = order_points(doc_quad * scale)
        return scaled_quad, round(min(1.0, max(0.5, doc_score)), 3), True

    # 2. Main context: Card / Document / Plate / Label with concentrated text
    context_quad, context_score = _find_main_context_quad(gray)
    if context_quad is not None:
        scaled_quad = order_points(context_quad * scale)
        return scaled_quad, round(context_score, 3), True

    # 3. Contrast segmentation fallback
    seg_quad, seg_score = _find_contrast_segmented_quad(gray)
    if seg_quad is not None:
        scaled_quad = order_points(seg_quad * scale)
        return scaled_quad, round(seg_score, 3), True

    # 4. Text content bounding quad fallback
    txt_quad, txt_score = _find_text_content_quad(gray)
    if txt_quad is not None:
        scaled_quad = order_points(txt_quad * scale)
        return scaled_quad, round(txt_score, 3), True

    # 5. Fallback to full frame
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
