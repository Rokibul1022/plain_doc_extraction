"""Optional OCR engine integration using Tesseract for searchable PDF export."""

import io
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger("docscan.ocr")


def is_ocr_available() -> bool:
    """Check if pytesseract and the underlying Tesseract binary are present and functional."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def is_heif_available() -> bool:
    """Check if pillow-heif is installed for HEIC photo decoding."""
    try:
        import pillow_heif  # noqa: F401
        return True
    except Exception:
        return False


def images_to_searchable_pdf(images: list[np.ndarray]) -> bytes:
    """Convert an ordered list of images to a searchable PDF using Tesseract OCR.

    Raises:
        RuntimeError: If Tesseract OCR is not available.
    """
    if not is_ocr_available():
        raise RuntimeError(
            "Tesseract OCR is not installed on this system. Install Tesseract to enable searchable PDF export."
        )

    import pytesseract
    from PIL import Image

    pdf_parts: list[bytes] = []
    for img_arr in images:
        if img_arr.ndim == 2:
            pil_img = Image.fromarray(img_arr, mode="L")
        else:
            # Convert BGR to RGB for Pillow / Tesseract
            rgb = img_arr[:, :, ::-1]
            pil_img = Image.fromarray(rgb, mode="RGB")

        pdf_bytes = pytesseract.image_to_pdf_or_hocr(pil_img, extension="pdf")
        pdf_parts.append(pdf_bytes)

    # If single page, return directly
    if len(pdf_parts) == 1:
        return pdf_parts[0]

    # Combine pages if multi-page
    try:
        import pikepdf
        merged = pikepdf.Pdf.new()
        for part in pdf_parts:
            src = pikepdf.Pdf.open(io.BytesIO(part))
            merged.pages.extend(src.pages)
        buf = io.BytesIO()
        merged.save(buf)
        return buf.getvalue()
    except Exception as exc:
        logger.warning(f"Failed to merge multi-page OCR PDF with pikepdf: {exc}")
        return pdf_parts[0]
