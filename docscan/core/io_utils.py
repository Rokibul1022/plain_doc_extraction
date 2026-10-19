"""Image I/O, EXIF correction, format conversion, and byte encoding."""

import io
from typing import Literal
import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image_from_bytes(data: bytes, max_dim: int = 4000) -> np.ndarray:
    """Load image from raw bytes, fix EXIF orientation, convert to BGR uint8, and cap size.

    Args:
        data: Raw image file bytes.
        max_dim: Maximum allowed dimension (width or height).

    Returns:
        BGR uint8 NumPy array.
    """
    try:
        pil_img = Image.open(io.BytesIO(data))
        pil_img = ImageOps.exif_transpose(pil_img) or pil_img
    except Exception as exc:
        raise ValueError(f"Failed to decode image data: {exc}") from exc

    # Handle various modes (RGBA, P, LA, L, RGB, CMYK, etc.)
    if pil_img.mode in ("RGBA", "LA") or (
        pil_img.mode == "P" and "transparency" in pil_img.info
    ):
        rgba = pil_img.convert("RGBA")
        # Composite against clean white background
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(bg, rgba).convert("RGB")
        rgb_arr = np.array(composite)
    else:
        rgb_arr = np.array(pil_img.convert("RGB"))

    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

    # Cap longest side if necessary
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest > max_dim:
        scale = max_dim / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(bgr, dtype=np.uint8)


def resize_to_max(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Resize image so its longest side is at most max_dim.

    Args:
        image: BGR or grayscale NumPy array.
        max_dim: Maximum size for width or height.

    Returns:
        Tuple of (resized_image, scale_factor_applied).
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image.copy(), 1.0

    scale = max_dim / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    return resized, scale


def encode_image(
    image: np.ndarray,
    fmt: Literal["jpeg", "png", "webp"] = "jpeg",
    quality: int = 85,
) -> bytes:
    """Encode an image array into specified file format bytes.

    Args:
        image: BGR or 1-channel uint8 array.
        fmt: Target format ('jpeg', 'png', or 'webp').
        quality: JPEG/WebP quality (1-100).

    Returns:
        Encoded file bytes.
    """
    fmt_clean = fmt.lower().strip(".")
    if fmt_clean in ("jpg", "jpeg"):
        ext = ".jpg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, quality))]
    elif fmt_clean == "png":
        ext = ".png"
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 4]
    elif fmt_clean == "webp":
        ext = ".webp"
        params = [int(cv2.IMWRITE_WEBP_QUALITY), max(1, min(100, quality))]
    else:
        raise ValueError(f"Unsupported export image format: {fmt}")

    success, buffer = cv2.imencode(ext, image, params)
    if not success:
        raise RuntimeError(f"Failed to encode image to {fmt}")
    return buffer.tobytes()
