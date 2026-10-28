"""HTTP API for the OpenCV-only plate scan (tab "OpenCV" in the main UI)."""

import asyncio
import base64
import os
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from opencv.imgio import encode_image, load_image_from_bytes
from opencv.scan import scan

router = APIRouter(prefix="/api/opencv", tags=["opencv"])

MAX_UPLOAD_BYTES = int(os.environ.get("OPENCV_MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_IMAGE_DIM = int(os.environ.get("OPENCV_MAX_IMAGE_DIM", "4000"))
# a scan is CPU-bound: run it off the event loop and let at most this many run at once in one process
_SLOTS = asyncio.Semaphore(int(os.environ.get("OPENCV_MAX_CONCURRENT", "2")))


def _data_uri(img: np.ndarray, png: bool = False) -> str:
    raw = encode_image(img, "png" if png else "jpeg", 92)
    return f"data:image/{'png' if png else 'jpeg'};base64," + base64.b64encode(raw).decode()


def _draw_corners(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    shown = img.copy()
    w = img.shape[1]
    cv2.polylines(shown, [corners.astype(np.int32)], True, (0, 220, 0), max(2, w // 200), cv2.LINE_AA)
    for x, y in corners:
        cv2.circle(shown, (int(x), int(y)), max(3, w // 60), (0, 120, 255), -1, cv2.LINE_AA)
    return shown


@router.post("/scan")
async def opencv_scan(file: UploadFile = File(...), relief: str = Form("auto")) -> dict:
    """Scan one image: corners overlay, straightened plate, restored plate and the final plain page."""
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Image too large")
    try:
        img = await asyncio.to_thread(load_image_from_bytes, data, MAX_IMAGE_DIM)
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Cannot read image: {exc}") from exc
    if relief not in ("auto", "force", "off"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "relief must be auto, force or off")
    t0 = time.time()
    async with _SLOTS:
        r = await asyncio.to_thread(scan, img, relief)
    return {
        "width": img.shape[1],
        "height": img.shape[0],
        "method": r.method,
        "relief": relief,
        "corners": [[round(float(x), 1), round(float(y), 1)] for x, y in r.corners],
        "original": _data_uri(_draw_corners(img, r.corners)),
        "rectified": _data_uri(r.rectified, png=True),
        "restored": _data_uri(r.restored, png=True),
        "document": _data_uri(r.document, png=True),
        "elapsed_s": round(time.time() - t0, 2),
    }
