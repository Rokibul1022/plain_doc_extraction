"""Stand-alone service for the OpenCV-only plate scan (no OCR, no models).

    uvicorn opencv.server:app --host 0.0.0.0 --port 8000 --workers 4

Endpoints:  POST /api/opencv/scan (file, relief=auto|force|off)   GET /health

CPU settings (environment):
  OPENCV_THREADS         threads OpenCV may use inside one scan (default 1: parallelism comes from --workers)
  OPENCV_MAX_CONCURRENT  scans running at once inside one worker process (default 2)
  OPENCV_MAX_UPLOAD_MB   upload limit (default 25)
"""

import os

import cv2
from fastapi import FastAPI

from opencv.api import router

cv2.setNumThreads(int(os.environ.get("OPENCV_THREADS", "1")))
cv2.ocl.setUseOpenCL(False)  # CPU only: no OpenCL probing / driver lookups

app = FastAPI(title="opencv plate scan", docs_url="/docs")
app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "opencv", "threads": cv2.getNumThreads()}
