"""Unit tests for PDF, image, and ZIP export functions."""

import io
import zipfile
import numpy as np
import pytest
from docscan.export.images import create_zip_archive, export_single_image
from docscan.export.pdf import images_to_pdf_bytes


def test_export_single_image_png_and_jpeg():
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    png_bytes, mime, ext = export_single_image(img, "png")
    assert mime == "image/png"
    assert ext == ".png"
    assert png_bytes.startswith(b"\x89PNG")

    jpg_bytes, mime, ext = export_single_image(img, "jpeg", quality=90)
    assert mime == "image/jpeg"
    assert ext == ".jpg"
    assert jpg_bytes.startswith(b"\xff\xd8")


def test_create_zip_archive():
    files = [("test1.txt", b"Hello world"), ("test2.txt", b"CamScanner")]
    zip_bytes = create_zip_archive(files)

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = zf.namelist()
        assert "test1.txt" in names
        assert "test2.txt" in names
        assert zf.read("test1.txt") == b"Hello world"


def test_images_to_pdf_single_and_multipage():
    page1 = np.full((300, 200, 3), 255, dtype=np.uint8)
    page2 = np.full((300, 200), 200, dtype=np.uint8)

    for layout in ("a4", "letter", "fit"):
        pdf_bytes = images_to_pdf_bytes([page1, page2], page_size=layout)
        assert pdf_bytes.startswith(b"%PDF")
        assert len(pdf_bytes) > 500
