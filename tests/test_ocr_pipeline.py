"""The OCR reading pipeline returns a checked, displayable report (models are the shipped ONNX files)."""

import cv2
import numpy as np
import pytest

from ocr.pipeline import read_plate, render_report
from ocr.synth_row import embossify, render_plate


def _plate(seed: int, emboss: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p, _, _ = render_plate(rng)
    return cv2.resize(embossify(p, rng, 0.3) if emboss else p, (240, 160))


@pytest.mark.parametrize("emboss", [False, True])
def test_report_is_well_formed(emboss):
    rep = read_plate(_plate(3, emboss))
    assert rep.verdict in ("readable", "uncertain", "unreadable")
    assert len(rep.digits) == 6 and len(rep.digit_conf) == 6
    assert all("০" <= c <= "৯" for c in rep.digits)  # only Bengali digits
    assert rep.mode in ("ink", "relief") and rep.page.ndim == 2


def test_every_digit_position_gets_a_digit_and_no_question_marks():
    rep = read_plate(_plate(5, True))
    assert all("০" <= c <= "৯" for c in rep.digits) and len(rep.digits) == 6


def test_result_picture_renders():
    pic = render_report(read_plate(_plate(7, False)))
    assert pic.ndim == 3 and pic.shape[1] == 1000 and pic.std() > 10
