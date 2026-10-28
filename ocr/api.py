"""HTTP API for plate scanning: images, videos (background jobs) and result assets."""

import asyncio
import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status

from opencv.imgio import encode_image, load_image_from_bytes
from ocr.scan import PlateResult, get_detector, get_ocr, scan_image
from ocr.video import PlateTrackResult, scan_video

logger = logging.getLogger("ocr.api")
router = APIRouter(prefix="/api/plate", tags=["plate"])

MAX_VIDEO_BYTES = 300 * 1024 * 1024
MAX_UPLOAD_BYTES = int(os.environ.get("OCR_MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_IMAGE_DIM = int(os.environ.get("OCR_MAX_IMAGE_DIM", "4000"))
ASSET_TTL = 3600
MAX_SCANS = 60
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".ogv", ".m4v", ".mpg", ".mpeg")


class AssetStore:
    """Thread-safe in-memory store of encoded result images, evicted by age and count."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, tuple[float, dict[str, tuple[bytes, str]]]] = {}

    def put(self, scan_id: str, assets: dict[str, tuple[bytes, str]]) -> None:
        with self._lock:
            now = time.time()
            self._data = {k: v for k, v in self._data.items() if now - v[0] < ASSET_TTL}
            while len(self._data) >= MAX_SCANS:
                self._data.pop(min(self._data, key=lambda k: self._data[k][0]))
            self._data[scan_id] = (now, assets)

    def get(self, scan_id: str, name: str) -> tuple[bytes, str] | None:
        with self._lock:
            entry = self._data.get(scan_id)
            return entry[1].get(name) if entry else None


assets = AssetStore()
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _enc(img: np.ndarray, png: bool = False) -> tuple[bytes, str]:
    if png:
        return encode_image(img, "png"), "image/png"
    return encode_image(img, "jpeg", 92), "image/jpeg"


def _annotate(img: np.ndarray, results: list[PlateResult]) -> np.ndarray:
    vis = img.copy()
    thick = max(2, max(vis.shape[:2]) // 400)
    for i, r in enumerate(results, 1):
        colour = {"readable": (80, 200, 90), "uncertain": (0, 190, 255), "unreadable": (70, 70, 240)}[r.verdict]
        cv2.polylines(vis, [r.detection.quad.astype(np.int32)], True, colour, thick, cv2.LINE_AA)
        x, y = int(r.detection.quad[:, 0].min()), int(r.detection.quad[:, 1].min())
        label = f"{i}: {r.reading.text or '?'}"
        scale = max(0.5, max(vis.shape[:2]) / 1600)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(vis, (x, max(0, y - th - 8)), (x + tw + 6, y), colour, -1)
        cv2.putText(vis, label, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return vis


def _plate_payload(scan_id: str, key: str, r_dict: dict, imgs: dict[str, np.ndarray],
                   store: dict[str, tuple[bytes, str]]) -> dict:
    urls = {}
    for name, im in imgs.items():
        if im is None:
            continue
        asset = f"{key}_{name}"
        store[asset] = _enc(im, png=True)
        urls[name] = f"/api/plate/asset/{scan_id}/{asset}"
    return {**r_dict, "images": urls}


def _image_response(img: np.ndarray, results: list[PlateResult], elapsed: float) -> dict:
    scan_id = uuid.uuid4().hex
    store: dict[str, tuple[bytes, str]] = {"annotated": _enc(_annotate(img, results))}
    plates = []
    for i, r in enumerate(results, 1):
        plates.append(_plate_payload(scan_id, f"p{i}", {"index": i, **r.to_dict()},
                                     {"document": r.document, "typed": r.typed, "scan": r.scan_document, "crop": r.crop, "restored": r.restored,
                                      "readable": r.readable}, store))
    assets.put(scan_id, store)
    return {
        "scan_id": scan_id,
        "width": img.shape[1],
        "height": img.shape[0],
        "annotated": f"/api/plate/asset/{scan_id}/annotated",
        "plates": plates,
        "elapsed_s": round(elapsed, 2),
    }


def _track_payload(scan_id: str, t: PlateTrackResult, store: dict[str, tuple[bytes, str]]) -> dict:
    return _plate_payload(
        scan_id, f"t{t.track_id}", t.to_dict(),
        {"document": t.document, "typed": t.typed, "scan": t.scan_document, "crop": t.best_crop, "fused": t.fused,
         "restored": t.restored, "readable": t.readable}, store,
    )


@router.get("/capabilities")
async def capabilities():
    det = await asyncio.to_thread(lambda: get_detector().learned.available)
    ocr = await asyncio.to_thread(lambda: get_ocr().available)
    return {"learned_detector": det, "ocr": ocr, "max_video_mb": MAX_VIDEO_BYTES // (1024 * 1024)}


@router.post("/scan")
async def scan(file: UploadFile = File(...), latin: bool = False):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Image too large.")
    try:
        img = await asyncio.to_thread(load_image_from_bytes, data, MAX_IMAGE_DIM)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    t0 = time.perf_counter()
    results = await asyncio.to_thread(lambda: scan_image(img, latin=latin))
    return await asyncio.to_thread(_image_response, img, results, time.perf_counter() - t0)


def _run_video_job(job_id: str, path: str, target_fps: float, max_seconds: float | None, latin: bool = False) -> None:
    def progress(p: float) -> None:
        with _jobs_lock:
            _jobs[job_id]["progress"] = round(p, 3)

    try:
        res = scan_video(path, target_fps=target_fps, max_seconds=max_seconds, progress=progress, latin=latin)
        scan_id = uuid.uuid4().hex
        store: dict[str, tuple[bytes, str]] = {}
        plates = [_track_payload(scan_id, t, store) for t in res.tracks]
        assets.put(scan_id, store)
        out = {"scan_id": scan_id, **{k: v for k, v in res.to_dict().items() if k != "plates"}, "plates": plates}
        with _jobs_lock:
            _jobs[job_id].update(status="done", progress=1.0, result=out)
    except Exception as exc:  # surfaced to the client instead of dying silently in the thread
        logger.exception("Video job failed")
        with _jobs_lock:
            _jobs[job_id].update(status="error", error=str(exc))
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@router.post("/video")
async def scan_video_upload(file: UploadFile = File(...), fps: float = 6.0, max_seconds: float | None = None,
                            latin: bool = False):
    name = (file.filename or "upload").lower()
    if not name.endswith(VIDEO_EXT):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unsupported video type; use one of {', '.join(VIDEO_EXT)}")
    suffix = Path(name).suffix
    fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="platevid_")
    size = 0
    with os.fdopen(fd, "wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_VIDEO_BYTES:
                fh.close()
                os.remove(tmp)
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Video too large.")
            fh.write(chunk)
    cap = cv2.VideoCapture(tmp)
    ok = cap.isOpened()
    cap.release()
    if not ok:
        os.remove(tmp)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Could not decode this video.")
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "progress": 0.0, "created": time.time()}
    threading.Thread(target=_run_video_job, args=(job_id, tmp, max(0.5, min(fps, 30.0)), max_seconds, latin),
                     daemon=True).start()
    return {"job_id": job_id}


@router.get("/job/{job_id}")
async def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown job.")
        return dict(job)


@router.get("/asset/{scan_id}/{name}")
async def asset(scan_id: str, name: str):
    found = assets.get(scan_id, name)
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Asset not found or expired.")
    data, mime = found
    return Response(content=data, media_type=mime)
