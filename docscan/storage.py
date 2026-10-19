"""Session storage for uploaded images and temporary files with TTL eviction."""

import logging
from pathlib import Path
import shutil
import threading
import time
import uuid
import cv2
import numpy as np

from docscan.config import settings
from docscan.core.io_utils import load_image_from_bytes

logger = logging.getLogger("docscan.storage")


class SessionStore:
    """Thread-safe disk-backed session manager for uploaded images."""

    def __init__(self, temp_dir: Path | None = None, ttl_seconds: int = 3600):
        self.temp_dir = temp_dir or settings.temp_dir
        self.ttl_seconds = ttl_seconds
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._metadata: dict[str, dict] = {}

    def save_image(self, data: bytes, filename: str = "upload.jpg") -> dict:
        """Decode image bytes, persist BGR array to disk, and index metadata."""
        image = load_image_from_bytes(data, max_dim=settings.max_image_dim)
        h, w = image.shape[:2]

        total_pixels = h * w
        if total_pixels > settings.max_pixels:
            raise ValueError(
                f"Image resolution ({w}x{h} = {total_pixels} px) exceeds maximum allowed limit of {settings.max_pixels} pixels."
            )

        image_id = uuid.uuid4().hex
        file_path = self.temp_dir / f"{image_id}.png"
        cv2.imwrite(str(file_path), image)

        now = time.time()
        meta = {
            "image_id": image_id,
            "filename": filename,
            "path": str(file_path),
            "width": w,
            "height": h,
            "created_at": now,
            "last_accessed": now,
        }

        with self._lock:
            self._metadata[image_id] = meta

        return meta

    def get_image(self, image_id: str) -> np.ndarray | None:
        """Retrieve original image from disk by image_id."""
        with self._lock:
            meta = self._metadata.get(image_id)
            if not meta:
                return None
            meta["last_accessed"] = time.time()
            path = Path(meta["path"])

        if not path.is_file():
            return None

        # cv2.imread loads as BGR
        return cv2.imread(str(path))

    def get_metadata(self, image_id: str) -> dict | None:
        """Retrieve session metadata for an image_id."""
        with self._lock:
            return self._metadata.get(image_id)

    def delete_image(self, image_id: str) -> bool:
        """Delete an image session and remove its file on disk."""
        with self._lock:
            meta = self._metadata.pop(image_id, None)

        if meta:
            path = Path(meta["path"])
            try:
                if path.exists():
                    path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {path}: {e}")
            return True
        return False

    def cleanup_expired(self) -> int:
        """Remove session images whose last access time exceeds the TTL."""
        now = time.time()
        expired_ids = []
        with self._lock:
            for img_id, meta in list(self._metadata.items()):
                if now - meta.get("last_accessed", meta["created_at"]) > self.ttl_seconds:
                    expired_ids.append(img_id)

        for img_id in expired_ids:
            self.delete_image(img_id)

        return len(expired_ids)

    def cleanup_all(self) -> None:
        """Wipe all session files and empty the temporary directory."""
        with self._lock:
            self._metadata.clear()
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                self.temp_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Error during shutdown cleanup: {e}")


session_store = SessionStore()
