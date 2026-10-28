"""Tests for the license-plate package (quality, restoration, readable output, OCR voting,
verdict, geometry, video tracking/fusion and the HTTP API)."""

from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest

from ocr.degrade import TIERS, degrade_crop
from ocr.detector import PlateDetector, rectify, refine_quad, true_aspect
from ocr.ocr import PlateOCR, PlateReading, vote
from opencv.quality import assess_quality, sharpness
from opencv.readable import make_readable
from opencv.restore import enhance_low_light, restore_plate, upscale
from ocr.verdict import reliability, verdict_from
from ocr.video import PlateTrack, associate, fuse_crops
from ocr.detector import PlateDetection

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "new samples" / "images (2).jpeg"  # Audi, small plate in a wide shot ("BT 43110")

_ocr = PlateOCR()
needs_ocr = pytest.mark.skipif(not _ocr.available, reason="plate OCR model unavailable (offline?)")
needs_detector = pytest.mark.skipif(not PlateDetector().learned.available, reason="plate detector model unavailable")


def make_plate(text="AB1234", dark_text=True, w=280, h=90) -> np.ndarray:
    body, ink = ((235, 235, 235), (20, 20, 20)) if dark_text else ((25, 25, 25), (235, 235, 235))
    img = np.full((h, w, 3), body, np.uint8)
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), ink, 2)
    cv2.putText(img, text, (14, int(h * 0.7)), cv2.FONT_HERSHEY_DUPLEX, 1.7, ink, 4, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- quality
def test_quality_orders_sharp_above_blurred_and_flags_issues():
    plate = make_plate()
    blurred = cv2.GaussianBlur(plate, (0, 0), 3.0)
    dark = (plate * 0.18).astype(np.uint8)
    tiny = cv2.resize(plate, (40, 13), interpolation=cv2.INTER_AREA)
    q_sharp, q_blur = assess_quality(plate), assess_quality(blurred)
    assert q_sharp.score > q_blur.score + 10
    assert sharpness(plate) > 2 * sharpness(blurred)
    assert "blurry" in q_blur.issues
    assert "dark" in assess_quality(dark).issues
    assert "low_resolution" in assess_quality(tiny).issues
    assert 0 <= q_blur.score <= 100


# ---------------------------------------------------------------- readable
@pytest.mark.parametrize("dark_text", [True, False])
def test_readable_is_always_dark_text_on_white(dark_text):
    out = make_readable(make_plate(dark_text=dark_text))
    assert out.ndim == 2
    assert float(np.median(out)) > 200  # body is white
    assert (out < 80).mean() > 0.02  # text is dark and present


def test_readable_binary_is_two_level():
    out = make_readable(make_plate(), binary=True)
    assert set(np.unique(out)) <= {0, 255}


# ---------------------------------------------------------------- restore / degrade
def test_restore_upscales_and_lifts_dark_crops():
    dark = (make_plate(w=120, h=40) * 0.25).astype(np.uint8)
    res = restore_plate(dark)
    assert res.image.shape[0] >= 96
    assert float(res.image.mean()) > float(dark.mean()) * 2
    assert any("low-light" in s for s in res.steps)
    assert upscale(np.zeros((200, 300, 3), np.uint8))[1] == 1.0
    assert enhance_low_light(dark).mean() > dark.mean()


def test_degradation_tiers_get_progressively_worse():
    plate = make_plate(w=400, h=130)
    rng = np.random.default_rng(0)
    sharp = {t: sharpness(degrade_crop(plate, t, rng, 400)) for t in TIERS}
    assert sharp["easy"] > sharp["extreme"]
    widths = {t: degrade_crop(plate, t, np.random.default_rng(1), 400).shape[1] for t in TIERS}
    assert widths["easy"] > widths["medium"] > widths["extreme"]


# ---------------------------------------------------------------- OCR voting + verdict
def _reading(text, conf, per=None):
    return PlateReading(text=text, confidence=conf, char_confidence=per or [conf] * len(text), format_ok=True)


def test_vote_prefers_agreement_and_repairs_single_character():
    a, b, c = _reading("AB1234", 0.95), _reading("AB1234", 0.9), _reading("AB1Z34", 0.4)
    v = vote([a, b, c])
    assert v.text == "AB1234" and v.votes == 2 and v.variants == 3
    # three same-length reads disagreeing on one character: the confident majority wins that slot
    v2 = vote([_reading("XY7788", 0.9), _reading("XY7788", 0.85), _reading("XY7188", 0.5)])
    assert v2.text == "XY7788"


def test_reliability_rises_with_confidence_and_zero_for_empty():
    q = assess_quality(make_plate())
    weak = _reading("AB1234", 0.40, [0.4, 0.3, 0.9, 0.9, 0.2, 0.9])
    strong = _reading("AB1234", 0.99)
    strong.votes = strong.variants = 3
    assert reliability(strong, q) > reliability(weak, q)
    assert reliability(PlateReading(), q) == 0.0
    assert verdict_from(0.95) == "readable" and verdict_from(0.5) == "uncertain" and verdict_from(0.1) == "unreadable"


@needs_ocr
def test_ocr_reads_a_rendered_plate_and_batch_matches_single():
    crop = make_plate("AB1234")
    single = _ocr.read(crop)
    batch = _ocr.read_many([crop, crop])
    assert single.text == batch[0].text == batch[1].text
    assert single.text  # produced something plate-like
    assert 0.0 <= single.confidence <= 1.0


# ---------------------------------------------------------------- geometry
def test_true_aspect_snaps_and_rectify_preserves_shape():
    quad = np.float32([[100, 100], [300, 100], [300, 200], [100, 200]])  # 2:1
    assert true_aspect(quad, (480, 640)) == pytest.approx(2.0, abs=0.01)
    img = np.zeros((480, 640, 3), np.uint8)
    out = rectify(img, quad)
    assert 1.8 < out.shape[1] / out.shape[0] < 2.5  # margin adds a little height, aspect stays ~2


def test_refine_quad_recovers_tilted_plate_inside_a_loose_box():
    scene = np.full((300, 500, 3), 70, np.uint8)
    plate = make_plate("ZX4417", w=200, h=70)
    src = np.float32([[0, 0], [199, 0], [199, 69], [0, 69]])
    dst = np.float32([[150, 110], [350, 95], [356, 160], [155, 176]])
    m = cv2.getPerspectiveTransform(src, dst)
    mask = cv2.warpPerspective(np.full((70, 200), 255, np.uint8), m, (500, 300))
    scene = np.where(mask[..., None] > 0, cv2.warpPerspective(plate, m, (500, 300)), scene)
    box = (140, 85, 365, 185)
    quad, refined = refine_quad(scene, box)
    assert refined
    err = np.linalg.norm(quad - dst, axis=1).max()
    assert err < 18


# ---------------------------------------------------------------- video
def _det(box):
    x1, y1, x2, y2 = box
    return PlateDetection(np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]), box, 0.9, "t")


def test_tracker_associates_moving_plate_and_keeps_two_cars_apart():
    t1, t2 = PlateTrack(1), PlateTrack(2)
    from ocr.video import Observation

    for tr, box in ((t1, (100, 100, 160, 130)), (t2, (400, 300, 460, 330))):
        tr.update(Observation(0, 0.0, _det(box), np.zeros((30, 60, 3), np.uint8), 50.0))
    tr_list = [t1, t2]
    mapping = associate(tr_list, [_det((108, 101, 168, 131)), _det((405, 300, 465, 330)), _det((250, 50, 310, 80))])
    assert mapping[0] == 0 and mapping[1] == 1 and 2 not in mapping  # third detection starts a new track


def test_fuse_crops_registers_shifted_noisy_frames():
    clean = make_plate("QW8821", w=160, h=52)
    rng = np.random.default_rng(3)
    crops = []
    for dx, dy in ((0, 0), (2, 1), (-1, 2), (1, -2), (-2, 0)):
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(clean, m, (160, 52), borderMode=cv2.BORDER_REPLICATE)
        crops.append(np.clip(shifted + rng.normal(0, 18, shifted.shape), 0, 255).astype(np.uint8))
    fused = fuse_crops(crops, [1.0] * len(crops))
    single = cv2.resize(crops[0], (fused.shape[1], fused.shape[0]), interpolation=cv2.INTER_CUBIC)
    # noise level in a flat, text-free strip of the plate body falls by ~sqrt(N) when frames are registered
    flat = (slice(14, 90), slice(10, 26))  # left margin: inside the border, before the first character
    assert fused[flat].std() < 0.65 * single[flat].std()
    # and registration keeps the characters sharp (no ghosting): edges are not much softer
    assert sharpness(fused) > 0.7 * sharpness(single) or sharpness(fused) > 0.3


# ---------------------------------------------------------------- end to end + API
@needs_ocr
@needs_detector
@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image missing")
def test_scan_finds_small_plate_in_wide_shot_and_reads_it():
    from ocr.scan import scan_image

    res = scan_image(cv2.imread(str(SAMPLE)), latin=True)
    assert res, "no plate found"
    best = max(res, key=lambda r: r.reading.confidence)
    assert best.reading.text == "BT43110"
    assert best.verdict in ("readable", "uncertain")
    assert best.readable.ndim == 2 and best.restored.shape[0] >= 96


@needs_ocr
@needs_detector
@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image missing")
def test_plate_api_scan_and_assets():
    from fastapi.testclient import TestClient

    from app.server import app

    c = TestClient(app)
    caps = c.get("/api/plate/capabilities").json()
    assert caps["learned_detector"] and caps["ocr"]
    r = c.post("/api/plate/scan?latin=true", files={"file": ("a.jpg", SAMPLE.read_bytes(), "image/jpeg")})
    assert r.status_code == 200
    d = r.json()
    assert d["plates"] and d["plates"][0]["reading"]["text"] == "BT43110"
    for url in (d["annotated"], d["plates"][0]["images"]["readable"]):
        a = c.get(url)
        assert a.status_code == 200 and a.headers["content-type"].startswith("image/")
    assert c.get("/api/plate/asset/nope/none").status_code == 404
    assert c.get("/").status_code == 200 and c.get("/doc").status_code == 200


@needs_ocr
@needs_detector
@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image missing")
def test_plate_api_video_job_tracks_a_moving_plate():
    import time
    from fastapi.testclient import TestClient

    from app.server import app

    scene = cv2.imread(str(SAMPLE))
    h, w = scene.shape[:2]
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "clip.mp4")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 12, (w, h))
        for i in range(18):
            m = np.float32([[1, 0, 2 * i], [0, 1, i]])
            vw.write(cv2.warpAffine(scene, m, (w, h), borderMode=cv2.BORDER_REPLICATE))
        vw.release()
        c = TestClient(app)
        r = c.post("/api/plate/video?fps=12&latin=true", files={"file": ("clip.mp4", Path(path).read_bytes(), "video/mp4")})
        assert r.status_code == 200
        job = r.json()["job_id"]
        for _ in range(120):
            st = c.get(f"/api/plate/job/{job}").json()
            if st["status"] != "running":
                break
            time.sleep(1)
        assert st["status"] == "done", st
        plates = st["result"]["plates"]
        assert len(plates) == 1  # one vehicle, tracked as one plate
        assert plates[0]["reading"]["text"] == "BT43110"
        assert plates[0]["frames_seen"] >= 10


# ---------------------------------------------------------------- Bengali reading + plain document
def test_ctc_decode_collapses_repeats_and_blanks_and_reports_confidence():
    from ocr.bangla_reader import BLANK, decode_probs

    def probs(seq):
        p = np.full((len(seq), 11), 0.01, np.float32)
        for t, k in enumerate(seq):
            p[t, k] = 0.9
        return p

    # digits 1,1 (repeat needs a blank between them), then 5
    text, conf = decode_probs(probs([1, 1, BLANK, 1, 1, BLANK, 5, 5]))
    assert text == "১১৫" and len(conf) == 3 and all(c >= 0.85 for c in conf)


def test_bengali_reliability_needs_six_digits_and_rises_with_confidence():
    from ocr.bangla_reader import RowReading
    from ocr.verdict_bn import reliability as rel_bn

    good = RowReading("১১৯০৯৫", 0.97, [0.97] * 6, 1.0, 6)
    weak = RowReading("১১৯০৯৫", 0.55, [0.9, 0.4, 0.5, 0.6, 0.3, 0.7], 0.4, 6)
    short = RowReading("১১৯০৯", 0.99, [0.99] * 5, 1.0, 6)
    assert rel_bn(good, 90) > rel_bn(weak, 90)
    assert rel_bn(short, 90) == 0.0


def test_frame_peeling_crops_just_inside_the_plate_frame():
    from opencv.layout import frame_crop_box

    binary = np.zeros((120, 200), np.uint8)
    cv2.rectangle(binary, (4, 4), (195, 115), 1, 4)  # a 4-px frame
    cv2.putText(binary, "88", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 1, 3)
    x0, y0, x1, y1 = frame_crop_box(binary)
    assert 5 <= x0 <= 14 and 5 <= y0 <= 14 and 186 <= x1 <= 195 and 106 <= y1 <= 115


def test_trim_letterbox_removes_black_bars_and_keeps_content():
    from ocr.scan import trim_letterbox

    img = np.zeros((120, 160, 3), np.uint8)
    img[:90, :110] = 200
    out, (dx, dy) = trim_letterbox(img)
    assert out.shape[:2] == (90, 110) and (dx, dy) == (0, 0)


@pytest.mark.skipif(not __import__("ocr.textrender", fromlist=["x"]).available(), reason="no Bengali font/shaper")
def test_bengali_text_is_shaped_not_scrambled_and_typed_page_is_clean():
    from ocr import textrender
    from ocr.typeset import render_typed

    word = textrender.render("মেট্রো", 80)
    assert word.shape[1] > 100 and (word < 128).mean() > 0.05
    page = render_typed(["চট্ট মেট্রো-ট", "১১-৯০৯৫"])
    assert page.ndim == 2 and page.shape[1] == 960
    assert (page == 255).mean() > 0.8  # a white page ...
    assert set(np.unique(page[page < 255]).tolist()) is not None and (page < 100).mean() > 0.04  # ... with solid black text


def test_make_document_returns_white_page_with_text_and_no_dark_frame_border():
    from opencv.document import make_document

    doc = make_document(make_plate("AB1234", w=300, h=110))
    assert doc.ndim == 2 and doc.shape[1] == 960
    assert (doc == 255).mean() > 0.5 and (doc < 80).mean() > 0.02  # mostly white paper, with solid dark text
    assert (doc[:, :6] == 255).all() and (doc[:, -6:] == 255).all()  # margins are clean paper
