"""Shared fixtures and synthetic test document generators for pytest."""

import cv2
import numpy as np
import pytest
from opencv.geometry import order_points


def create_synthetic_document(
    bg_size: tuple[int, int] = (1200, 1600),
    page_size: tuple[int, int] = (800, 1100),
    add_shadow: bool = True,
    add_noise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic photograph of a paper document on a textured surface.

    Args:
        bg_size: (height, width) of the full canvas.
        page_size: (height, width) of the unwarped document page.
        add_shadow: Whether to apply a non-uniform lighting shadow gradient.
        add_noise: Whether to add sensor noise.

    Returns:
        Tuple of (photograph_bgr, ground_truth_corners_float32).
    """
    np.random.seed(1234)  # deterministic texture/noise, so detector tests are not flaky
    bg_h, bg_w = bg_size
    doc_h, doc_w = page_size

    # 1. Base table background: dark-brown wooden texture
    table = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
    table[:, :] = (35, 45, 60)  # Dark table color (BGR)
    # Add subtle texture
    texture = np.random.randint(-10, 10, (bg_h, bg_w, 3), dtype=np.int16)
    table = np.clip(table.astype(np.int16) + texture, 0, 255).astype(np.uint8)

    # 2. Render document page (clean white paper with black text lines and headers)
    doc_img = np.full((doc_h, doc_w, 3), 245, dtype=np.uint8)
    # Add simulated header
    cv2.putText(
        doc_img,
        "DOCUMENT SCANNER TEST REPORT",
        (50, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.line(doc_img, (50, 105), (doc_w - 50, 105), (50, 50, 50), 2)

    # Add simulated text lines
    for y in range(160, doc_h - 100, 40):
        length = np.random.randint(doc_w // 2, doc_w - 100)
        cv2.line(doc_img, (60, y), (length, y), (40, 40, 40), 3)

    # Add a couple of colored boxes/stamps to test color retention
    cv2.rectangle(doc_img, (doc_w - 220, doc_h - 220), (doc_w - 60, doc_h - 60), (30, 30, 200), 3)
    cv2.putText(
        doc_img, "APPROVED", (doc_w - 200, doc_h - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 200), 2
    )

    # 3. Perspective warp page onto table canvas
    src_pts = np.array(
        [[0, 0], [doc_w - 1, 0], [doc_w - 1, doc_h - 1], [0, doc_h - 1]],
        dtype=np.float32,
    )

    # Target points inside table frame: tilted and perspective-distorted
    dst_pts = np.array(
        [
            [bg_w * 0.18, bg_h * 0.14],  # TL
            [bg_w * 0.82, bg_h * 0.10],  # TR
            [bg_w * 0.88, bg_h * 0.86],  # BR
            [bg_w * 0.12, bg_h * 0.82],  # BL
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_doc = cv2.warpPerspective(doc_img, M, (bg_w, bg_h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(
        np.full((doc_h, doc_w), 255, dtype=np.uint8), M, (bg_w, bg_h), flags=cv2.INTER_NEAREST
    )

    # Composite onto table
    mask_3d = mask[:, :, None] > 0
    composite = np.where(mask_3d, warped_doc, table)

    # 4. Add lighting shadow gradient across the entire frame
    if add_shadow:
        x_grad = np.linspace(0.45, 1.15, bg_w, dtype=np.float32)
        y_grad = np.linspace(0.55, 1.05, bg_h, dtype=np.float32)[:, None]
        lighting = np.clip(x_grad * y_grad, 0.3, 1.2)[:, :, None]
        composite = np.clip(composite.astype(np.float32) * lighting, 0, 255).astype(np.uint8)

    # 5. Add subtle noise
    if add_noise:
        noise = np.random.normal(0, 3, composite.shape).astype(np.float32)
        composite = np.clip(composite.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    gt_corners = order_points(dst_pts)
    return composite, gt_corners


@pytest.fixture
def synthetic_sample() -> tuple[np.ndarray, np.ndarray]:
    """Fixture providing a synthetic photo and its ground truth corners."""
    return create_synthetic_document(bg_size=(1000, 1200), page_size=(700, 900))
