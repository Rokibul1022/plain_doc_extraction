"""FastAPI application factory, routes, static asset mounting, and exception handlers."""

import asyncio
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator, Literal
from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np

from docscan.config import settings
from docscan.core.detect import detect_document
from docscan.core.io_utils import encode_image
from docscan.core.pipeline import process
from docscan.export.images import create_zip_archive, export_single_image
from docscan.export.ocr import is_heif_available, is_ocr_available
from docscan.export.pdf import images_to_pdf_bytes
from docscan.schemas import (
    DetectResponse,
    ExportRequest,
    HealthResponse,
    ProcessParams,
    UploadResponse,
)
from docscan.storage import session_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("docscan.server")


class DetectRequest(BaseModel):
    image_id: str
    target: Literal["auto", "plate", "document"] = "auto"


class PreviewRequest(BaseModel):
    image_id: str
    corners: list[list[float]]
    params: ProcessParams


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage background session cleanup task and handle clean shutdown."""
    logger.info("DocScan server initialized.")
    cleanup_task = asyncio.create_task(_periodic_cleanup())
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    session_store.cleanup_all()
    logger.info("DocScan server stopped; temporary session files cleared.")


async def _periodic_cleanup() -> None:
    """Periodically purge expired image sessions."""
    while True:
        try:
            await asyncio.sleep(settings.session_cleanup_interval)
            evicted = session_store.cleanup_expired()
            if evicted > 0:
                logger.info(f"Evicted {evicted} expired session images.")
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error(f"Error in background cleanup task: {exc}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="DocScan API",
        version="1.0.0",
        description="Classical CV CamScanner-style document processing backend",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Ensure static directory exists
    settings.static_dir.mkdir(parents=True, exist_ok=True)

    # Middleware to prevent caching of frontend HTML/JS/CSS assets during development
    @app.middleware("http")
    async def add_no_cache_headers(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_file = settings.static_dir / "index.html"
        if not index_file.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="index.html not found in static directory.",
            )
        response = FileResponse(index_file)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=204)

    @app.get("/api/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="ok",
            version="1.0.0",
            capabilities={
                "ocr": is_ocr_available(),
                "heif": is_heif_available(),
            },
        )

    @app.post("/api/upload", response_model=UploadResponse)
    async def upload_image(file: UploadFile = File(...)):
        # Validate MIME type if available
        if file.content_type and file.content_type.lower() not in settings.allowed_mime_types:
            logger.warning(f"Uploaded file MIME type {file.content_type} outside allowed list")

        contents = await file.read()
        if len(contents) > settings.max_upload_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded file exceeds maximum limit of {settings.max_upload_size_bytes // (1024*1024)} MB.",
            )

        try:
            meta = await asyncio.to_thread(
                session_store.save_image, contents, file.filename or "upload.jpg"
            )
        except ValueError as val_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(val_err),
            )
        except Exception as exc:
            logger.error(f"Failed to process upload: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Image decoding failed: {exc}",
            )

        return UploadResponse(
            image_id=meta["image_id"],
            width=meta["width"],
            height=meta["height"],
            preview_url=f"/api/raw/{meta['image_id']}",
        )

    @app.get("/api/raw/{image_id}", include_in_schema=False)
    async def get_raw_image(image_id: str):
        img = await asyncio.to_thread(session_store.get_image, image_id)
        if img is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image {image_id} not found.",
            )
        encoded = await asyncio.to_thread(encode_image, img, "jpeg", 85)
        return Response(content=encoded, media_type="image/jpeg")

    @app.post("/api/detect", response_model=DetectResponse)
    async def detect_corners(req: DetectRequest):
        img = await asyncio.to_thread(session_store.get_image, req.image_id)
        if img is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image session {req.image_id} not found.",
            )

        corners, confidence, detected = await asyncio.to_thread(
            detect_document, img, settings.detection_height, req.target
        )
        return DetectResponse(
            corners=corners.tolist(),
            confidence=round(confidence, 3),
            detected=detected,
        )

    @app.post("/api/preview")
    async def generate_preview(req: PreviewRequest):
        img = await asyncio.to_thread(session_store.get_image, req.image_id)
        if img is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image session {req.image_id} not found.",
            )

        result = await asyncio.to_thread(
            process, img, req.corners, req.params, preview=True
        )

        # Encode preview as JPEG
        preview_bytes = await asyncio.to_thread(encode_image, result.image, "jpeg", 85)
        headers = {
            "X-Mode-Used": result.mode_used,
            "X-Total-Ms": str(result.timings_ms.get("total_ms", 0)),
        }
        return Response(content=preview_bytes, media_type="image/jpeg", headers=headers)

    @app.post("/api/export")
    async def export_document(req: ExportRequest):
        if not req.pages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one page is required for export.",
            )

        processed_pages: list[tuple[str, np.ndarray]] = []
        for idx, page_item in enumerate(req.pages):
            img = await asyncio.to_thread(session_store.get_image, page_item.image_id)
            if img is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Image session {page_item.image_id} not found for page {idx + 1}.",
                )
            # Run at full resolution
            res = await asyncio.to_thread(
                process, img, page_item.corners, page_item.params, preview=False
            )
            processed_pages.append((f"page_{idx + 1:03d}", res.image))

        # 1. PDF Export
        if req.format == "pdf":
            try:
                images_list = [img for _, img in processed_pages]
                pdf_bytes = await asyncio.to_thread(
                    images_to_pdf_bytes,
                    images_list,
                    page_size=req.page_size,
                    jpeg_quality=req.jpeg_quality,
                    ocr=req.ocr,
                )
                return Response(
                    content=pdf_bytes,
                    media_type="application/pdf",
                    headers={
                        "Content-Disposition": 'attachment; filename="scanned_document.pdf"'
                    },
                )
            except Exception as exc:
                logger.error(f"PDF export failure: {exc}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to generate PDF: {exc}",
                )

        # 2. ZIP Export
        if req.format == "zip":
            files_to_zip: list[tuple[str, bytes]] = []
            for name, img in processed_pages:
                b, _, ext = export_single_image(img, "png", req.jpeg_quality)
                files_to_zip.append((f"{name}{ext}", b))
            zip_bytes = await asyncio.to_thread(create_zip_archive, files_to_zip)
            return Response(
                content=zip_bytes,
                media_type="application/zip",
                headers={
                    "Content-Disposition": 'attachment; filename="scanned_pages.zip"'
                },
            )

        # 3. Single / Multi image format (PNG or JPEG)
        fmt = "jpeg" if req.format == "jpeg" else "png"
        if len(processed_pages) == 1:
            name, img = processed_pages[0]
            b, mime, ext = export_single_image(img, fmt, req.jpeg_quality)
            return Response(
                content=b,
                media_type=mime,
                headers={"Content-Disposition": f'attachment; filename="{name}{ext}"'},
            )

        # Multi-page image export -> archive in ZIP
        files_to_zip = []
        for name, img in processed_pages:
            b, _, ext = export_single_image(img, fmt, req.jpeg_quality)
            files_to_zip.append((f"{name}{ext}", b))
        zip_bytes = await asyncio.to_thread(create_zip_archive, files_to_zip)
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="scanned_{fmt}.zip"'},
        )

    @app.delete("/api/image/{image_id}")
    async def delete_image(image_id: str):
        success = session_store.delete_image(image_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image session {image_id} not found.",
            )
        return {"status": "deleted", "image_id": image_id}

    # Mount static assets directory
    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")

    return app


app = create_app()
