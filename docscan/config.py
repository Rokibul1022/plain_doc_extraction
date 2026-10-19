"""Configuration settings for DocScan."""

from dataclasses import dataclass, field
from pathlib import Path
import tempfile


@dataclass(frozen=True)
class Settings:
    """Application-wide configuration parameters."""

    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_size_bytes: int = 25 * 1024 * 1024  # 25 MB
    max_pixels: int = 40_000_000  # 40 MP
    max_image_dim: int = 4000  # cap longest side
    detection_height: int = 600
    preview_max_dim: int = 1200
    session_ttl_seconds: int = 3600  # 1 hour
    session_cleanup_interval: int = 300  # check every 5 min
    temp_dir: Path = field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "docscan_sessions"
    )
    static_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "static"
    )
    allowed_mime_types: tuple[str, ...] = (
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/tiff",
        "image/heic",
        "image/heif",
    )


settings = Settings()
