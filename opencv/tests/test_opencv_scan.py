"""Self-contained tests for the OpenCV-only plate scan: run with  python -m pytest opencv/tests"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from opencv import scan  # noqa: E402


def synthetic_plate(tilt: float = 6.0) -> np.ndarray:
    """A green plate with dark digits on a grey background, slightly rotated."""
    img = np.full((240, 360, 3), 90, np.uint8)
    cv2.rectangle(img, (40, 50), (320, 190), (120, 200, 90), -1)
    cv2.putText(img, "11-0905", (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 40, 20), 5, cv2.LINE_AA)
    cv2.putText(img, "METRO", (110, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 40, 20), 2, cv2.LINE_AA)
    m = cv2.getRotationMatrix2D((180, 120), tilt, 1.0)
    return cv2.warpAffine(img, m, (360, 240), borderValue=(90, 90, 90))


def test_scan_returns_four_corners_and_a_plain_page():
    r = scan(synthetic_plate())
    assert r.corners.shape == (4, 2)
    assert r.document.ndim == 2 and r.document.dtype == np.uint8
    assert (r.document == 255).mean() > 0.5  # white paper
    assert (r.document < 80).mean() > 0.01  # dark characters


def test_scan_is_deterministic():
    img = synthetic_plate()
    a, b = scan(img), scan(img)
    assert np.allclose(a.corners, b.corners)
    assert np.array_equal(a.document, b.document)


def test_relief_mode_runs_on_a_plate_without_ink():
    img = np.full((200, 320, 3), (150, 190, 120), np.uint8)
    cv2.putText(img, "11-1990", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (165, 205, 135), 6, cv2.LINE_AA)  # same colour: relief only
    r = scan(cv2.GaussianBlur(img, (0, 0), 1.0), relief="force")
    assert r.document.ndim == 2 and r.document.shape[1] == 960


def test_concurrent_scans_in_threads_match_sequential_scans():
    """The scan keeps no shared per-image state: a thread pool (as in the web service) must give the same corners."""
    from concurrent.futures import ThreadPoolExecutor

    from opencv import find_corners

    imgs = [synthetic_plate(t) for t in (-8, -3, 0, 4, 9, 12)] + [cv2.resize(synthetic_plate(5), (300, 200))]
    sequential = [find_corners(i)[0] for i in imgs]
    with ThreadPoolExecutor(4) as pool:
        parallel = list(pool.map(lambda i: find_corners(i)[0], imgs * 3))
    for k, corners in enumerate(parallel):
        assert np.array_equal(corners, sequential[k % len(imgs)])
