"""PDF document generation supporting single and multi-page documents, standard page sizes, and OCR."""

import io
import logging
from typing import Literal
import numpy as np
from PIL import Image

from docscan.core.io_utils import encode_image
from docscan.export.ocr import images_to_searchable_pdf, is_ocr_available

logger = logging.getLogger("docscan.export.pdf")

try:
    import img2pdf
    HAS_IMG2PDF = True
    A4_POINTS = (img2pdf.in_to_pt(8.27), img2pdf.in_to_pt(11.69))
    LETTER_POINTS = (img2pdf.in_to_pt(8.5), img2pdf.in_to_pt(11.0))
except ImportError:
    img2pdf = None
    HAS_IMG2PDF = False
    # Standard dimensions in points (72 DPI)
    A4_POINTS = (595.28, 841.89)
    LETTER_POINTS = (612.0, 792.0)

PageSizeOption = Literal["a4", "letter", "fit"]


def _pillow_pdf_fallback(images: list[np.ndarray]) -> bytes:
    """Fallback generator using Pillow when img2pdf is unavailable or fails."""
    pil_images = []
    for img in images:
        if img.ndim == 2:
            pil_images.append(Image.fromarray(img, mode="L"))
        else:
            pil_images.append(Image.fromarray(img[:, :, ::-1], mode="RGB"))

    buf = io.BytesIO()
    first, rest = pil_images[0], pil_images[1:]
    first.save(buf, format="PDF", save_all=True, append_images=rest)
    return buf.getvalue()


def images_to_pdf_bytes(
    images: list[np.ndarray],
    page_size: PageSizeOption = "a4",
    jpeg_quality: int = 85,
    ocr: bool = False,
) -> bytes:
    """Generate a single or multi-page PDF document from an array of images.

    Args:
        images: List of uint8 images (1-channel or BGR).
        page_size: 'a4', 'letter', or 'fit'.
        jpeg_quality: Quality for color images (85 default).
        ocr: If True and OCR is available, generates a searchable text PDF.

    Returns:
        Raw PDF file bytes.
    """
    if not images:
        raise ValueError("Cannot generate PDF from an empty list of images.")

    if ocr and is_ocr_available():
        return images_to_searchable_pdf(images)

    if not HAS_IMG2PDF:
        return _pillow_pdf_fallback(images)

    # Encode images to memory bytes suitable for img2pdf
    encoded_pages: list[bytes] = []
    for img in images:
        if img.ndim == 2:
            # Grayscale or binary: encode as PNG for sharp, artifact-free text
            page_bytes = encode_image(img, fmt="png")
        else:
            page_bytes = encode_image(img, fmt="jpeg", quality=jpeg_quality)
        encoded_pages.append(page_bytes)

    # Configure page layout
    layout_fun = None
    if page_size == "a4":
        layout_fun = img2pdf.get_layout_fun(
            A4_POINTS, fit=img2pdf.FitMode.into, auto_orient=True
        )
    elif page_size == "letter":
        layout_fun = img2pdf.get_layout_fun(
            LETTER_POINTS, fit=img2pdf.FitMode.into, auto_orient=True
        )

    try:
        if layout_fun is not None:
            return img2pdf.convert(encoded_pages, layout_fun=layout_fun)
        return img2pdf.convert(encoded_pages)
    except Exception as exc:
        logger.warning(f"img2pdf conversion failed, falling back to Pillow: {exc}")
        return _pillow_pdf_fallback(images)
