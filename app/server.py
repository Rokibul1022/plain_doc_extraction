"""FastAPI web app: serves the plate UI and mounts the OCR route (/api/plate/*), the merged-reader route (/api/merge/*)
and the OpenCV route (/api/opencv/*).

    python run.py                                        # opens http://127.0.0.1:8000
    uvicorn app.server:app --host 0.0.0.0 --port 8000
"""

import logging
import os
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ocr.api import router as ocr_router
from ocr_merge.api import router as merge_router
from opencv.api import router as opencv_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app.server")

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
NO_CACHE = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}


def create_app() -> FastAPI:
    cv2.setNumThreads(int(os.environ.get("OPENCV_THREADS", "1")))
    app = FastAPI(title="Plate scanner", version="2.0.0",
                  description="Plate UI: OpenCV-only scan (opencv/) and YOLO + OCR (ocr/)")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def no_cache_for_frontend(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith(("/static", "/opencv-static")) or request.url.path == "/":
            response.headers.update(NO_CACHE)
        return response

    @app.get("/", include_in_schema=False)
    async def index():
        page = STATIC / "plate.html"
        if not page.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "plate.html not found")
        return FileResponse(page, headers=NO_CACHE)

    app.include_router(ocr_router)
    app.include_router(merge_router)
    app.include_router(opencv_router)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.mount("/opencv-static", StaticFiles(directory=ROOT / "opencv" / "static"), name="opencv-static")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=204)

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": "2.0.0"}

    return app


app = create_app()
