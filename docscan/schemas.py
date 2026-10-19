"""Pydantic schemas and parameters for DocScan API and processing pipeline."""

from typing import Any, Literal
from pydantic import BaseModel, Field


class ProcessParams(BaseModel):
    """Parameters governing document enhancement and output modes."""

    mode: Literal[
        "plain_text",
        "magic_color",
        "grayscale",
        "original",
        "auto",
        "bw",
        "color_enhanced",
        "handwriting",
        "plate_text",
    ] = Field(default="plain_text", description="Output mode preset")

    stroke_weight: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Text stroke thickness adjustment (-1.0 minimal/thin, 0.0 clean, 1.0 bold)",
    )

    clean_background: bool = Field(
        default=True,
        description="Flatten paper background to pure white like CamScanner",
    )

    brightness: float = Field(
        default=0.0, ge=-100.0, le=100.0, description="Brightness adjustment offset"
    )
    contrast: float = Field(
        default=0.0, ge=-100.0, le=100.0, description="Contrast adjustment percentage"
    )

    deskew: bool = Field(
        default=False, description="Fine small-angle tilt alignment"
    )
    rotation: int = Field(
        default=0, description="Clockwise rotation in degrees (0, 90, 180, 270)"
    )

    # Legacy parameters preserved for backward compatibility
    auto_invert: bool = Field(
        default=False,
        description="Legacy auto-invert parameter",
    )
    denoise_type: Literal["none", "median", "bilateral", "nlm"] = Field(
        default="none", description="Denoising filter technique"
    )
    denoise_strength: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Denoising intensity"
    )
    sharpen_amount: float = Field(
        default=1.0, ge=0.0, le=3.0, description="Sharpening boost factor"
    )
    sharpen_auto: bool = Field(
        default=False, description="Automatically scale sharpening by measured blur"
    )
    threshold_method: Literal["plain_text", "adaptive", "sauvola", "otsu"] = Field(
        default="plain_text", description="Binarization algorithm for B&W modes"
    )
    threshold_block_size: int = Field(
        default=35, ge=3, le=151, description="Local threshold window size (odd integer)"
    )
    threshold_c: float = Field(
        default=14.0, ge=-50.0, le=50.0, description="Constant offset subtracted from local mean"
    )
    remove_speckles: bool = Field(
        default=True, description="Remove tiny isolated dust/speckles in B&W mode"
    )


class UploadResponse(BaseModel):
    """Response returned upon successful image upload."""

    image_id: str
    width: int
    height: int
    preview_url: str


class DetectResponse(BaseModel):
    """Response containing document boundary coordinates."""

    corners: list[list[float]]
    confidence: float
    detected: bool


class PageExportItem(BaseModel):
    """Export specification for a single page."""

    image_id: str
    corners: list[list[float]]
    params: ProcessParams


class ExportRequest(BaseModel):
    """Request payload for exporting processed documents."""

    pages: list[PageExportItem]
    format: Literal["pdf", "png", "jpeg", "zip"] = Field(default="pdf")
    page_size: Literal["a4", "letter", "fit"] = Field(default="a4")
    jpeg_quality: int = Field(default=85, ge=10, le=100)
    ocr: bool = Field(default=False)


class HealthResponse(BaseModel):
    """System health and capability response."""

    status: str
    version: str
    capabilities: dict[str, bool]
