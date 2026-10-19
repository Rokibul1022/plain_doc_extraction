"""Integration tests for FastAPI endpoints."""

import io
import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from docscan.server import app

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "ocr" in data["capabilities"]
    assert "heif" in data["capabilities"]


def test_full_api_workflow(synthetic_sample):
    img, _ = synthetic_sample
    success, buffer = cv2.imencode(".jpg", img)
    assert success
    img_bytes = buffer.tobytes()

    # 1. Upload
    files = {"file": ("document.jpg", img_bytes, "image/jpeg")}
    upload_res = client.post("/api/upload", files=files)
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    image_id = upload_data["image_id"]
    assert image_id
    assert upload_data["width"] > 0
    assert upload_data["height"] > 0

    # 2. Detect
    detect_res = client.post("/api/detect", json={"image_id": image_id})
    assert detect_res.status_code == 200
    detect_data = detect_res.json()
    corners = detect_data["corners"]
    assert len(corners) == 4
    assert detect_data["detected"] is True

    # 3. Preview
    preview_payload = {
        "image_id": image_id,
        "corners": corners,
        "params": {
            "mode": "bw",
            "threshold_method": "adaptive",
            "threshold_block_size": 35,
            "threshold_c": 10.0,
        },
    }
    preview_res = client.post("/api/preview", json=preview_payload)
    assert preview_res.status_code == 200
    assert preview_res.headers["content-type"] == "image/jpeg"
    assert "x-mode-used" in preview_res.headers

    # 4. Export PDF
    export_payload = {
        "pages": [
            {
                "image_id": image_id,
                "corners": corners,
                "params": preview_payload["params"],
            }
        ],
        "format": "pdf",
        "page_size": "a4",
        "jpeg_quality": 85,
        "ocr": False,
    }
    export_res = client.post("/api/export", json=export_payload)
    assert export_res.status_code == 200
    assert export_res.headers["content-type"] == "application/pdf"
    assert export_res.content.startswith(b"%PDF")

    # 5. Export ZIP
    export_payload["format"] = "zip"
    zip_res = client.post("/api/export", json=export_payload)
    assert zip_res.status_code == 200
    assert zip_res.headers["content-type"] == "application/zip"
    assert zip_res.content.startswith(b"PK")

    # 6. Delete image
    del_res = client.delete(f"/api/image/{image_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # Subsequent access should 404
    detect_404 = client.post("/api/detect", json={"image_id": image_id})
    assert detect_404.status_code == 404


def test_upload_invalid_file():
    # Empty or corrupt image data
    files = {"file": ("corrupt.jpg", b"not an image", "image/jpeg")}
    res = client.post("/api/upload", files=files)
    assert res.status_code in (400, 500)
