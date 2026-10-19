"""Export individual images as PNG/JPEG and package multi-page documents as ZIP."""

import io
from typing import Literal
import zipfile
import numpy as np

from docscan.core.io_utils import encode_image


def export_single_image(
    image: np.ndarray,
    fmt: Literal["png", "jpeg"] = "png",
    quality: int = 85,
) -> tuple[bytes, str, str]:
    """Encode an image and return (bytes, mime_type, file_extension)."""
    clean_fmt = "jpeg" if fmt.lower() in ("jpg", "jpeg") else "png"
    mime = "image/jpeg" if clean_fmt == "jpeg" else "image/png"
    ext = ".jpg" if clean_fmt == "jpeg" else ".png"

    img_bytes = encode_image(image, fmt=clean_fmt, quality=quality)
    return img_bytes, mime, ext


def create_zip_archive(items: list[tuple[str, bytes]]) -> bytes:
    """Bundle a list of named byte payloads into an in-memory ZIP file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, data in items:
            zf.writestr(filename, data)
    return buf.getvalue()
