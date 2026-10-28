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
