"""Geometric transformations: corner ordering, perspective warping, rotation, deskew."""

import cv2
import numpy as np


def order_points(pts: np.ndarray | list) -> np.ndarray:
    """Order four 2D points in clockwise order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.

    Args:
        pts: Array-like of shape (4, 2).

    Returns:
        NumPy array of shape (4, 2) with dtype float32 [TL, TR, BR, BL].
    """
    pts_arr = np.asarray(pts, dtype=np.float32).reshape((4, 2))
    ordered = np.zeros((4, 2), dtype=np.float32)

    # Sum of coordinates: x + y
    # Top-left has smallest sum, bottom-right has largest sum
    s = pts_arr.sum(axis=1)
    ordered[0] = pts_arr[np.argmin(s)]
    ordered[2] = pts_arr[np.argmax(s)]

    # Difference of coordinates: x - y
    # Top-right has largest difference (large x, small y)
    # Bottom-left has smallest difference (small x, large y)
    diff = pts_arr[:, 0] - pts_arr[:, 1]
    ordered[1] = pts_arr[np.argmax(diff)]
    ordered[3] = pts_arr[np.argmin(diff)]

    return ordered


def scale_corners(corners: np.ndarray | list, factor: float) -> np.ndarray:
    """Scale corner coordinates by a multiplication factor."""
    pts = np.asarray(corners, dtype=np.float32).reshape((4, 2))
    return pts * float(factor)


def warp(image: np.ndarray, corners: np.ndarray | list) -> np.ndarray:
    """Perspective-warp the document quadrilateral into a top-down rectified image.

    Args:
        image: Source image (BGR or grayscale uint8).
        corners: 4 corner points in source image coordinate space.

    Returns:
        Warped rectangular image.
    """
    ordered = order_points(corners)
    tl, tr, br, bl = ordered

    # Compute width of new image (maximum of top and bottom edge distances)
    width_bottom = np.linalg.norm(br - bl)
    width_top = np.linalg.norm(tr - tl)
    max_width = max(int(round(width_bottom)), int(round(width_top)))

    # Compute height of new image (maximum of right and left edge distances)
    height_right = np.linalg.norm(tr - br)
    height_left = np.linalg.norm(tl - bl)
    max_height = max(int(round(height_right)), int(round(height_left)))

    # Safeguard minimum dimensions
    out_w = max(10, max_width)
    out_h = max(10, max_height)

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(
        image, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def rotate90(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image by 0, 90, 180, or 270 degrees clockwise.

    Args:
        image: Source image.
        angle: Rotation angle in degrees (clockwise).

    Returns:
        Rotated image.
    """
    normalized = angle % 360
    if normalized == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if normalized == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if normalized == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image.copy()


def rotate_corners(
    corners: np.ndarray | list, angle: int, width: int, height: int
) -> np.ndarray:
    """Rotate corner coordinates when the underlying image has been rotated."""
    pts = order_points(corners)
    normalized = angle % 360
    if normalized == 0:
        return pts

    rotated = np.zeros_like(pts)
    for i, (x, y) in enumerate(pts):
        if normalized == 90:
            rotated[i] = [height - 1 - y, x]
        elif normalized == 180:
            rotated[i] = [width - 1 - x, height - 1 - y]
        elif normalized == 270:
            rotated[i] = [y, width - 1 - x]

    return order_points(rotated)


def _text_mask(gray: np.ndarray, max_dim: int = 500) -> np.ndarray:
    """Downscaled binary mask of the text strokes (the minority tone on the page)."""
    h, w = gray.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.count_nonzero(mask) > mask.size / 2:
        mask = cv2.bitwise_not(mask)
    return mask


def _row_sharpness(mask: np.ndarray, angle: float) -> float:
    """How sharply text lines stack into rows once the mask is rotated by `angle` degrees."""
    h, w = mask.shape[:2]
    rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(mask, rot, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    profile = rotated.sum(axis=1, dtype=np.float64) / 255.0
    return float(np.sum(np.diff(profile) ** 2))


def estimate_text_skew(image: np.ndarray, max_angle: float = 20.0) -> float:
    """Estimate the rotation (degrees, cv2 convention: positive = counter-clockwise) that
    makes the text lines horizontal. Returns 0.0 when there is no clear text-line signal.

    Uses a projection-profile search: the correct angle is where the row histogram of text
    pixels is sharpest. This follows the text itself rather than borders or clutter."""
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = _text_mask(gray)
    if np.count_nonzero(mask) < 50:
        return 0.0

    base = _row_sharpness(mask, 0.0)
    coarse = np.arange(-max_angle, max_angle + 0.01, 1.0)
    scores = [_row_sharpness(mask, a) for a in coarse]
    best = float(coarse[int(np.argmax(scores))])
    fine = np.arange(best - 1.0, best + 1.01, 0.1)
    fine_scores = [_row_sharpness(mask, a) for a in fine]
    best = float(fine[int(np.argmax(fine_scores))])
    best_score = max(fine_scores)

    # Only trust a clear improvement over the unrotated page.
    if base <= 0 or best_score < 1.05 * base:
        return 0.0
    return best


def _hough_skew(gray: np.ndarray, max_angle: float) -> float:
    """Fallback skew estimate from near-horizontal Hough line segments."""
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 40, 140, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=max(25, min(75, int(w * 0.15))),
        minLineLength=max(15, w // 8), maxLineGap=15,
    )
    if lines is None or len(lines) == 0:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        deg = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if abs(deg) <= max_angle:
            angles.append(deg)
        elif abs(abs(deg) - 180.0) <= max_angle:
            angles.append(deg - 180.0 if deg > 0 else deg + 180.0)
    return float(np.median(angles)) if angles else 0.0


def deskew(image: np.ndarray, max_angle: float = 20.0) -> np.ndarray:
    """Level the text on a rectified document so it reads straight, like a flatbed scan.

    The angle comes from the text lines themselves (projection profile), falling back to
    Hough lines. The canvas is enlarged to hold the rotated page so no corner is cut off;
    the new margins are filled white.

    Args:
        image: Warped BGR or grayscale image.
        max_angle: Maximum rotation in degrees to consider as skew.

    Returns:
        Deskewed image.
    """
    if image.size == 0:
        return image

    angle = estimate_text_skew(image, max_angle)
    if angle == 0.0:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        angle = _hough_skew(gray, min(max_angle, 12.0))
    if abs(angle) < 0.3:
        return image

    h, w = image.shape[:2]
    rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    cos, sin = abs(rot[0, 0]), abs(rot[0, 1])
    new_w = int(np.ceil(h * sin + w * cos))
    new_h = int(np.ceil(h * cos + w * sin))
    rot[0, 2] += new_w / 2.0 - w / 2.0
    rot[1, 2] += new_h / 2.0 - h / 2.0
    return cv2.warpAffine(
        image,
        rot,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if image.ndim == 3 else 255,
    )


def _col_sharpness(mask: np.ndarray, shear: float) -> float:
    """How sharply vertical strokes stack into columns after shearing x' = x + shear*(y-cy)."""
    h, w = mask.shape[:2]
    pad = int(np.ceil(abs(shear) * h / 2.0))
    m = np.float32([[1.0, shear, -shear * h / 2.0 + pad], [0.0, 1.0, 0.0]])
    sheared = cv2.warpAffine(mask, m, (w + 2 * pad, h), flags=cv2.INTER_LINEAR, borderValue=0)
    profile = sheared.sum(axis=0, dtype=np.float64) / 255.0
    return float(np.sum(np.diff(profile) ** 2))


def estimate_text_shear(image: np.ndarray, max_angle: float = 15.0) -> float:
    """Estimate the horizontal shear factor that makes character strokes vertical.

    Returns 0.0 unless vertical-stroke alignment improves clearly, so upright print,
    natural handwriting slant and weak signals are left untouched."""
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = _text_mask(gray)
    if np.count_nonzero(mask) < 50:
        return 0.0

    base = _col_sharpness(mask, 0.0)
    angles = np.arange(-max_angle, max_angle + 0.01, 1.0)
    scores = [_col_sharpness(mask, float(np.tan(np.radians(a)))) for a in angles]
    best = float(angles[int(np.argmax(scores))])
    fine = np.arange(best - 1.0, best + 1.01, 0.2)
    fine_scores = [_col_sharpness(mask, float(np.tan(np.radians(a)))) for a in fine]
    best = float(fine[int(np.argmax(fine_scores))])

    if base <= 0 or max(fine_scores) < 1.15 * base or abs(best) < 1.0:
        return 0.0
    return float(np.tan(np.radians(best)))


def deshear(image: np.ndarray) -> np.ndarray:
    """Straighten slanted/keystoned characters so strokes stand vertical (after deskew)."""
    shear = estimate_text_shear(image)
    if shear == 0.0:
        return image
    h, w = image.shape[:2]
    pad = int(np.ceil(abs(shear) * h / 2.0))
    m = np.float32([[1.0, shear, -shear * h / 2.0 + pad], [0.0, 1.0, 0.0]])
    return cv2.warpAffine(
        image, m, (w + 2 * pad, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if image.ndim == 3 else 255,
    )
