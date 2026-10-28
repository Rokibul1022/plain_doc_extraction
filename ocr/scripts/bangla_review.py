"""Review sheets for verifying/correcting the Bengali digit reader on a folder of plate crops.

Each tile shows the rectified plate, the model's reading in Bengali, its confidence and the plate id.
After checking, write corrections into labels.csv (file,digits) - one row per verified plate; plates
that cannot be read by a human are left out (or marked ?) and excluded from accuracy numbers.

Usage:
    python ocr/scripts/bangla_review.py FOLDER --out DIR [--start 0 --count 12]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ocr.bangla_reader import BanglaRowReader  # noqa: E402
from ocr.digits import FONT_PATH  # noqa: E402
from ocr.scan import get_detector, scan_image  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def tile(crop: np.ndarray, text: str, conf: float, idx: int, name: str, w: int = 330) -> np.ndarray:
    s = w / crop.shape[1]
    z = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    h = z.shape[0]
    canvas = np.full((h + 62, w, 3), 28, np.uint8)
    canvas[:h] = z
    pil = Image.fromarray(canvas[..., ::-1])
    d = ImageDraw.Draw(pil)
    color = (90, 220, 120) if conf >= 0.85 else (255, 190, 60) if conf >= 0.6 else (255, 90, 90)
    d.text((6, h + 4), text or "—", font=ImageFont.truetype(FONT_PATH, 34, index=1), fill=color)
    d.text((6, h + 44), f"#{idx} {name[:14]}  {conf:.2f}", font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13), fill=(190, 190, 190))
    return np.asarray(pil)[..., ::-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="review")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--cols", type=int, default=3)
    a = ap.parse_args()

    files = sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in EXT)
    reader, det = BanglaRowReader(), get_detector()
    tiles = []
    for i in range(a.start, min(a.start + a.count, len(files))):
        img = cv2.imread(str(files[i]))
        res = scan_image(img, detector=det, do_ocr=False)
        if not res:
            continue
        crop = res[0].crop
        r = reader.read(crop)
        tiles.append(tile(crop, r.text, r.confidence, i, files[i].name))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    H = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28)) for t in tiles]
    while len(tiles) % a.cols:
        tiles.append(np.full_like(tiles[0], 28))
    sheet = np.vstack([np.hstack(tiles[k: k + a.cols]) for k in range(0, len(tiles), a.cols)])
    path = Path(a.out) / f"review_{a.start:03d}.jpg"
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(path)


if __name__ == "__main__":
    main()
