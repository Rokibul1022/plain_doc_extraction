"""Command line: scan plate images with the OpenCV-only pipeline.

    python -m opencv.cli plate.jpg --out out/
    python -m opencv.cli "folder of plates" --out out/ --relief force     # embossed plates with no ink

Per image it writes  <name>_corners.jpg  <name>_straight.png  <name>_plain.png  into --out.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from opencv.scan import scan

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="an image, or a folder of images")
    ap.add_argument("--out", default="opencv_out", help="output folder (default: opencv_out)")
    ap.add_argument("--relief", choices=("auto", "force", "off"), default="auto",
                    help="3D relief reading for plates stamped in the plate colour (auto = only if the ink map is empty)")
    a = ap.parse_args(argv)
    src = Path(a.path)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXT) if src.is_dir() else [src]
    if not files:
        print("no images found", file=sys.stderr)
        return 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"skip {f.name}: cannot read", file=sys.stderr)
            continue
        r = scan(img, relief=a.relief)
        shown = img.copy()
        cv2.polylines(shown, [r.corners.astype(np.int32)], True, (0, 220, 0), max(2, img.shape[1] // 200), cv2.LINE_AA)
        cv2.imwrite(str(out / f"{f.stem}_corners.jpg"), shown)
        cv2.imwrite(str(out / f"{f.stem}_straight.png"), r.rectified)
        cv2.imwrite(str(out / f"{f.stem}_plain.png"), r.document)
        print(f"{f.name}: corners by {r.method} -> {f.stem}_plain.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
