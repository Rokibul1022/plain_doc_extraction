"""Unit tests for enhancement modules: illumination, denoise, sharpen, binarize."""

import cv2
import numpy as np
import pytest
from docscan.core.binarize import binarize, remove_small_speckles
from docscan.core.denoise import denoise
from docscan.core.illumination import apply_clahe, remove_shadows
from docscan.core.modes import classify_document_mode
from docscan.core.sharpen import estimate_blur, sharpen


def test_illumination_reduces_shadow_variance():
    # Create white document with a strong synthetic shadow gradient
    h, w = 300, 400
    base = np.full((h, w), 240, dtype=np.uint8)
    # Apply harsh linear shadow across x
    gradient = np.linspace(0.3, 1.0, w, dtype=np.float32)
    shadowed = np.clip(base.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

    shadowed_var = float(np.var(shadowed))
    corrected = remove_shadows(shadowed, strength=1.0)
    corrected_var = float(np.var(corrected))

    # Variance across the paper region should drop dramatically
    assert corrected_var < shadowed_var * 0.25


def test_clahe_contrast_enhancement():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    enhanced = apply_clahe(img, clip_limit=2.0)
    assert enhanced.shape == img.shape


def test_denoise_methods():
    noisy = (np.random.normal(128, 15, (100, 100, 3))).astype(np.uint8)
    for method in ("median", "bilateral", "nlm"):
        cleaned = denoise(noisy, method=method, strength=0.5)
        assert cleaned.shape == noisy.shape
        assert float(np.var(cleaned)) <= float(np.var(noisy))


def test_sharpen_and_blur_estimation():
    # Blurred image vs sharp edges
    blurry = cv2.GaussianBlur(
        (np.random.randint(0, 255, (100, 100), dtype=np.uint8)), (15, 15), 0
    )
    b_val = estimate_blur(blurry)

    sharp_pattern = np.zeros((100, 100), dtype=np.uint8)
    sharp_pattern[::4, :] = 255
    s_val = estimate_blur(sharp_pattern)

    assert s_val > b_val

    sharpened = sharpen(blurry, amount=1.5, auto_scale=True)
    assert sharpened.shape == blurry.shape


def test_binarize_is_strictly_binary():
    gray = np.random.randint(50, 200, (150, 150), dtype=np.uint8)
    for method in ("adaptive", "sauvola", "otsu"):
        binary = binarize(gray, method=method)
        unique_vals = set(np.unique(binary))
        assert unique_vals.issubset({0, 255})


def test_speckle_removal():
    # White background (255) with a 2x2 dark speckle and a large 20x20 dark text block
    binary = np.full((100, 100), 255, dtype=np.uint8)
    binary[10:12, 10:12] = 0  # 4 pixels speckle
    binary[30:50, 30:50] = 0  # 400 pixels text block

    cleaned = remove_small_speckles(binary, min_area=8)
    # The 4px speckle should be removed (turned to 255)
    assert np.all(cleaned[10:12, 10:12] == 255)
    # The text block should remain 0
    assert np.all(cleaned[30:50, 30:50] == 0)


def test_classify_mode():
    # Colorful image
    color_img = np.zeros((100, 100, 3), dtype=np.uint8)
    color_img[:, :] = (0, 0, 255)  # Bright red
    assert classify_document_mode(color_img) == "color_enhanced"

    # Clean text document
    text_doc = np.full((200, 200, 3), 250, dtype=np.uint8)
    for y in range(20, 180, 20):
        cv2.line(text_doc, (20, y), (180, y), (20, 20, 20), 4)
    mode = classify_document_mode(text_doc)
    assert mode in ("bw", "handwriting")


def test_clean_plate_and_label_text():
    from docscan.core.binarize import clean_plate_and_label_text
    # Plate with black text on white background
    plate = np.full((50, 160, 3), 240, dtype=np.uint8)
    cv2.putText(plate, "ABC 123", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    cleaned = clean_plate_and_label_text(plate)
    assert cleaned.ndim == 2
    unique_vals = set(np.unique(cleaned))
    assert unique_vals.issubset({0, 255})
    # There should be text (0) and background (255)
    assert 0 in unique_vals and 255 in unique_vals
