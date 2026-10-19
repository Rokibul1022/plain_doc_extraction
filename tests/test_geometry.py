"""Unit tests for geometric transformations."""

import numpy as np
import pytest
from docscan.core.geometry import deskew, order_points, rotate90, rotate_corners, scale_corners, warp


def test_order_points_standard():
    # Input in random order: BL, TR, TL, BR
    pts = np.array([[20, 180], [190, 15], [10, 10], [185, 195]], dtype=np.float32)
    ordered = order_points(pts)

    assert ordered.shape == (4, 2)
    # TL should be [10, 10]
    np.testing.assert_allclose(ordered[0], [10, 10])
    # TR should be [190, 15]
    np.testing.assert_allclose(ordered[1], [190, 15])
    # BR should be [185, 195]
    np.testing.assert_allclose(ordered[2], [185, 195])
    # BL should be [20, 180]
    np.testing.assert_allclose(ordered[3], [20, 180])


def test_scale_corners():
    corners = np.array([[0, 0], [10, 0], [10, 20], [0, 20]], dtype=np.float32)
    scaled = scale_corners(corners, 2.5)
    np.testing.assert_allclose(scaled[1], [25, 0])
    np.testing.assert_allclose(scaled[2], [25, 50])


def test_warp_dimensions():
    # Create simple 400x300 image with known corners
    img = np.ones((400, 300, 3), dtype=np.uint8) * 200
    corners = np.array([[50, 50], [250, 60], [240, 350], [45, 340]], dtype=np.float32)

    warped = warp(img, corners)
    assert warped.ndim == 3
    # Width approx 200, height approx 290-300
    assert 180 <= warped.shape[1] <= 220
    assert 280 <= warped.shape[0] <= 320


def test_rotate90():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[0:10, 0:10] = (255, 0, 0)  # Blue dot at top-left

    rot90 = rotate90(img, 90)
    assert rot90.shape == (200, 100, 3)

    rot180 = rotate90(img, 180)
    assert rot180.shape == (100, 200, 3)

    rot270 = rotate90(img, 270)
    assert rot270.shape == (200, 100, 3)

    rot0 = rotate90(img, 0)
    assert rot0.shape == (100, 200, 3)


def test_rotate_corners():
    w, h = 300, 400
    corners = np.array([[10, 10], [280, 15], [285, 390], [15, 385]], dtype=np.float32)
    rot_c = rotate_corners(corners, 90, w, h)
    assert rot_c.shape == (4, 2)
    # Check that points are ordered properly
    ordered = order_points(rot_c)
    np.testing.assert_allclose(rot_c, ordered)


def test_deskew_no_crash():
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    res = deskew(img, max_angle=5.0)
    assert res.shape == img.shape
