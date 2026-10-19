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


def deskew(image: np.ndarray, max_angle: float = 12.0) -> np.ndarray:
    """Refine small residual tilt/skew on a rectified text document using Hough lines.

    Args:
        image: Warped BGR or grayscale image.
        max_angle: Maximum rotation in degrees to consider as skew.

    Returns:
        Deskewed image.
    """
    if image.size == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    # Edge detection
    edges = cv2.Canny(gray, 40, 140, apertureSize=3)
    thresh = max(25, min(75, int(w * 0.15)))
    min_len = max(15, w // 8)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=thresh, minLineLength=min_len, maxLineGap=15
    )

    if lines is None or len(lines) == 0:
        return image

    lines = lines.reshape(-1, 4)
    angles = []
    for x1, y1, x2, y2 in lines:
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = np.degrees(angle_rad)
        # We are looking for nearly horizontal lines
        if abs(angle_deg) <= max_angle:
            angles.append(angle_deg)
        elif abs(abs(angle_deg) - 180.0) <= max_angle:
            norm_deg = angle_deg - 180.0 if angle_deg > 0 else angle_deg + 180.0
            angles.append(norm_deg)

    if not angles:
        return image

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.2:
        return image

    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(
        image,
        rot_mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if image.ndim == 3 else 255,
    )
