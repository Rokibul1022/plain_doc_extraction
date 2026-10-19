"""Unit tests for the end-to-end processing pipeline."""

import numpy as np
import pytest
from docscan.core.pipeline import process
from docscan.schemas import ProcessParams


def test_pipeline_modes(synthetic_sample):
    img, corners = synthetic_sample
    modes = ["original", "color_enhanced", "grayscale", "bw", "handwriting", "auto"]

    for mode in modes:
        params = ProcessParams(mode=mode)
        result = process(img, corners, params, preview=True)

        assert result.image is not None
        assert result.image.size > 0
        assert result.mode_used in modes
        assert "total_ms" in result.timings_ms
        assert result.timings_ms["total_ms"] > 0


def test_pipeline_preview_vs_fullres(synthetic_sample):
    img, corners = synthetic_sample
    params = ProcessParams(mode="bw")

    # Preview run
    res_preview = process(img, corners, params, preview=True)
    # Full resolution run
    res_full = process(img, corners, params, preview=False)

    h_prev, w_prev = res_preview.image.shape[:2]
    h_full, w_full = res_full.image.shape[:2]

    # Full res dimensions should be greater than or equal to preview
    assert max(h_full, w_full) >= max(h_prev, w_prev)


def test_pipeline_adjustments(synthetic_sample):
    img, corners = synthetic_sample
    params = ProcessParams(
        mode="color_enhanced",
        brightness=20.0,
        contrast=15.0,
        rotation=90,
        deskew=True,
    )
    result = process(img, corners, params, preview=True)
    assert result.image is not None
    assert result.image.shape[0] > 0
