"""HTTP route for the merged reader (used by the "OCR Merge" tab): same pipeline and JSON as /api/plate/read, one model.

    POST /api/merge/read          form: file, mode = auto | ink | relief
    GET  /api/merge/capabilities  {"available", "model_mb", "device", "views", "retry_views", "priors"}
"""

import asyncio
import os
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from ocr.api import _READ_SLOTS, MAX_IMAGE_DIM, MAX_UPLOAD_BYTES, read_payload
from ocr_merge.pipeline import RETRY_DEFAULT, read_plate, reader
from ocr_merge.views import VIEW_SETS
from opencv.imgio import load_image_from_bytes

router = APIRouter(prefix="/api/merge", tags=["merge"])


@router.get("/capabilities")
async def capabilities():
    r = reader()
    available = await asyncio.to_thread(lambda: r.available)  # also loads the model, so the first read is not slower
    return {"available": available, "model_mb": round(r.path.stat().st_size / 2 ** 20, 1) if r.path.exists() else None,
            "device": r.device, "views": len(r.views), "retry_views": len(VIEW_SETS[os.environ.get("MERGE_RETRY_VIEWS", RETRY_DEFAULT)])
            if os.environ.get("MERGE_RETRY_VIEWS", RETRY_DEFAULT) in VIEW_SETS else 0, "priors": r.priors is not None}


@router.post("/read")
async def read(file: UploadFile = File(...), mode: str = Form("auto")):
    if mode not in ("auto", "ink", "relief"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "mode must be auto, ink or relief")
    if not reader().path.exists():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "ocr_merge/plate_net.onnx is missing")
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Image too large.")
    try:
        img = await asyncio.to_thread(load_image_from_bytes, data, MAX_IMAGE_DIM)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    t0 = time.perf_counter()
    async with _READ_SLOTS:
        rep = await asyncio.to_thread(read_plate, img, None if mode == "auto" else mode)
    return read_payload(img, rep, t0, model="merged", device=reader().device, views=rep.views, retried=rep.retried)
