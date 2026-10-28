"""The web app serves the plate UI and both APIs."""

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.server import app

client = TestClient(app)


def _png() -> bytes:
    img = np.full((120, 240, 3), (120, 200, 90), np.uint8)
    cv2.putText(img, "11-0905", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 40, 20), 4)
    return cv2.imencode(".png", img)[1].tobytes()


def test_ui_and_static_files_are_served():
    r = client.get("/")
    assert r.status_code == 200 and "OpenCV" in r.text and "Document mode" not in r.text
    for url in ("/static/plate.js", "/static/plate.css", "/static/styles.css", "/opencv-static/opencv.js", "/opencv-static/opencv.css"):
        assert client.get(url).status_code == 200, url
    assert client.get("/api/health").json()["status"] == "ok"


def test_opencv_route_accepts_the_relief_option():
    for mode, code in (("auto", 200), ("force", 200), ("bogus", 400)):
        r = client.post("/api/opencv/scan", files={"file": ("p.png", _png(), "image/png")}, data={"relief": mode})
        assert r.status_code == code, mode
    d = client.post("/api/opencv/scan", files={"file": ("p.png", _png(), "image/png")}).json()
    assert d["document"].startswith("data:image/png;base64,") and len(d["corners"]) == 4


def test_ocr_route_is_mounted():
    assert client.get("/api/plate/capabilities").status_code == 200


def test_read_endpoint_runs_the_new_pipeline():
    for mode, code in (("auto", 200), ("relief", 200), ("ink", 200), ("bogus", 400)):
        r = client.post("/api/plate/read", files={"file": ("p.png", _png(), "image/png")}, data={"mode": mode})
        assert r.status_code == code, mode
    forced = {m: client.post("/api/plate/read", files={"file": ("p.png", _png(), "image/png")}, data={"mode": m}).json()["mode"] for m in ("ink", "relief")}
    assert forced == {"ink": "ink", "relief": "relief"}  # the form field really selects the reader (the UI checkbox relies on it)
    d = client.post("/api/plate/read", files={"file": ("p.png", _png(), "image/png")}).json()
    assert d["verdict"] in ("readable", "uncertain", "unreadable") and len(d["digits"]) == 6
    assert len(d["digits"]) == 6 and d["number"] == d["digits"][:2] + "-" + d["digits"][2:]
    assert not any("conf" in k for k in d)  # no confidence numbers are exposed
    assert set(d["images"]) == {"result", "page", "straight", "original"} and d["images"]["result"].startswith("data:image/png")
    assert client.post("/api/plate/read", files={"file": ("p.png", b"", "image/png")}).status_code == 400


def test_image_tab_uses_the_read_endpoint():
    js = client.get("/static/plate.js").text
    assert "/api/plate/read" in js and "embossedRead" in js
    assert 'id="embossedRead"' in client.get("/").text
