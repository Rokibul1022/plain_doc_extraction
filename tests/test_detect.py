"""Unit tests for document boundary detection."""

import cv2
import numpy as np
import pytest
from docscan.core.detect import detect_document
from docscan.core.geometry import order_points


def test_detect_synthetic_document(synthetic_sample):
    img, gt_corners = synthetic_sample
    h, w = img.shape[:2]
    diag = np.sqrt(h**2 + w**2)

    corners, confidence, detected = detect_document(img, det_h=600)

    assert detected is True
    assert confidence > 0.3
    assert corners.shape == (4, 2)

    # Compute maximum corner distance between detected and ground truth
    ordered_gt = order_points(gt_corners)
    ordered_detected = order_points(corners)
    dists = np.linalg.norm(ordered_gt - ordered_detected, axis=1)
    max_err = np.max(dists)

    # Error must be below 2.5% of diagonal
    assert max_err < 0.025 * diag, f"Max corner error {max_err:.1f}px exceeds 2.5% diag ({0.025*diag:.1f}px)"


def test_detect_fallback_on_blank():
    # Plain uniform image where no edges exist
    blank = np.full((500, 500, 3), 128, dtype=np.uint8)
    corners, confidence, detected = detect_document(blank, det_h=300)

    # Must fail gracefully without exception
    assert corners.shape == (4, 2)
    # When no document can be found, detected should be False
    assert detected is False
    assert confidence == 0.0


def test_detect_fallback_on_extreme_cases():
    black = np.zeros((400, 400, 3), dtype=np.uint8)
    corners, _, _ = detect_document(black)
    assert corners.shape == (4, 2)

    white = np.full((400, 400, 3), 255, dtype=np.uint8)
    corners, _, _ = detect_document(white)
    assert corners.shape == (4, 2)


def test_detect_sample_documents_and_text():
    from pathlib import Path
    samples_dir = Path(__file__).parent.parent / "samples"
    for doc_file in ["invoice.jpg", "receipt.jpg", "images (1).jpeg"]:
        p = samples_dir / doc_file
        if not p.is_file():
            continue
        img = cv2.imread(str(p))
        corners, conf, det = detect_document(img)
        assert det is True
        assert conf > 0.3
        assert corners.shape == (4, 2)



def _plate_in_scene() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Dark 'car' scene with a small white plate carrying several characters."""

    rng = np.random.default_rng(3)
    img = np.clip(rng.normal(55, 18, (480, 640, 3)), 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    x0, y0, x1, y1 = 250, 300, 390, 345
    cv2.rectangle(img, (x0, y0), (x1, y1), (235, 235, 235), -1)
    cv2.putText(img, "BT 4311", (x0 + 8, y1 - 9), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3)
    return img, (x0, y0, x1, y1)


def test_small_plate_in_wide_shot_is_localised():
    img, (x0, y0, x1, y1) = _plate_in_scene()
    corners, confidence, detected = detect_document(img, det_h=600, target="auto")

    assert detected is True
    cx, cy = corners.mean(axis=0)
    assert x0 <= cx <= x1 and y0 <= cy <= y1
    # Region must be plate-sized, not the whole scene or a single glyph.
    area = cv2.contourArea(corners.astype(np.float32))
    plate_area = (x1 - x0) * (y1 - y0)
    assert 0.5 * plate_area <= area <= 3.0 * plate_area


def test_tiny_blob_falls_back_to_full_frame():

    img = np.full((500, 400, 3), 200, np.uint8)
    cv2.circle(img, (200, 250), 18, (20, 20, 20), 4)  # lone glyph-like ring
    corners, confidence, detected = detect_document(img, det_h=600, target="auto")

    area = cv2.contourArea(corners.astype(np.float32))
    assert area >= 0.9 * 400 * 500 or detected is False


def _iou(a: np.ndarray, b: np.ndarray, shape: tuple[int, int]) -> float:
    m1 = np.zeros(shape, np.uint8)
    m2 = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m1, np.round(a).astype(np.int32), 1)
    cv2.fillConvexPoly(m2, np.round(b).astype(np.int32), 1)
    return float((m1 & m2).sum()) / max(1.0, float((m1 | m2).sum()))


def test_page_on_dark_background_beats_stamp_inside_it(synthetic_sample):
    """A page photographed on a dark, rotated background must be found whole - not one of the
    stamps/boxes printed on it."""
    img, gt_corners = synthetic_sample
    h, w = img.shape[:2]
    rng = np.random.default_rng(1)
    big_w, big_h = 1000, 760
    bg = np.clip(rng.normal(60, 14, (big_h, big_w, 3)), 0, 255).astype(np.uint8)
    scale = 0.55
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 8.0, scale)
    m[0, 2] += big_w / 2 - w / 2
    m[1, 2] += big_h / 2 - h / 2
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), m, (big_w, big_h))
    warped = cv2.warpAffine(img, m, (big_w, big_h), flags=cv2.INTER_AREA)
    scene = np.where(mask[..., None] > 0, warped, bg)
    gt = order_points((np.c_[gt_corners, np.ones(4)] @ m.T).astype(np.float32))

    corners, _, detected = detect_document(scene, det_h=600, target="auto")

    assert detected is True
    assert _iou(order_points(corners), gt, (big_h, big_w)) > 0.8


def test_detected_quad_is_never_degenerate(synthetic_sample):
    from docscan.core.detect import _valid_quad

    img, _ = synthetic_sample
    corners, _, detected = detect_document(img, det_h=600, target="auto")
    assert detected is True
    assert _valid_quad(corners)
    assert not _valid_quad(np.array([[0, 0], [100, 0], [100, 100], [98, 99]], np.float32))


def test_rotated_plate_is_found_by_its_own_geometry():
    img, (x0, y0, x1, y1) = _plate_in_scene()
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D(((x0 + x1) / 2, (y0 + y1) / 2), 14.0, 1.0)
    rotated = cv2.warpAffine(img, m, (w, h), borderValue=(55, 55, 55))
    corners, _, detected = detect_document(rotated, det_h=600, target="auto")

    assert detected is True
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    assert abs(corners.mean(axis=0)[0] - cx) < 40 and abs(corners.mean(axis=0)[1] - cy) < 40
    area = cv2.contourArea(corners.astype(np.float32))
    assert 0.5 * (x1 - x0) * (y1 - y0) <= area <= 3.0 * (x1 - x0) * (y1 - y0)
