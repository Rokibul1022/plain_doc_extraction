"""Stand-alone service for the OCR route: YOLO plate detector + plate reading (+ video).

    uvicorn ocr.server:app --host 0.0.0.0 --port 8001 --workers 2

Endpoints:  POST /api/plate/scan   POST /api/plate/video   GET /api/plate/capabilities   GET /health

CPU settings (environment):
  ORT_INTRA_THREADS / ORT_INTER_THREADS  ONNX Runtime threads per model session (0 = ONNX default = all cores)
  OPENCV_THREADS                         OpenCV threads (default 1)
  OCR_MAX_UPLOAD_MB                      upload limit (default 25)
Models are loaded lazily on the first request; the Docker image pre-downloads them so nothing is fetched at runtime.
"""

import os

import cv2
from fastapi import FastAPI

from ocr.api import router

cv2.setNumThreads(int(os.environ.get("OPENCV_THREADS", "1")))
cv2.ocl.setUseOpenCL(False)

app = FastAPI(title="ocr + yolo plate reader", docs_url="/docs")
app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ocr"}
