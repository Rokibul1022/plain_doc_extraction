"""Unit tests for document boundary detection."""

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
    import cv2
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

