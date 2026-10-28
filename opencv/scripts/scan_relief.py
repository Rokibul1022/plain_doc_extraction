"""Scan a folder of embossed plates twice (normal and 3D relief) and save pages + a comparison sheet.

    python opencv/scripts/scan_relief.py "new samples+3d" --out eval/relief3d
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from opencv.scan import scan  # noqa: E402

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def fit(z: np.ndarray, w: int, h: int) -> np.ndarray:
    z = z if z.ndim == 3 else cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)
    s = min(w / z.shape[1], h / z.shape[0])
    z = cv2.resize(z, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    o = np.full((h, w, 3), 245, np.uint8)
    o[: z.shape[0], : z.shape[1]] = z
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="eval/relief3d")
    a = ap.parse_args()
    out = Path(a.out)
    for sub in ("normal", "relief", "corners"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in EXT):
        img = cv2.imread(str(f))
        n, r = scan(img, relief="off"), scan(img, relief="force")
        shown = img.copy()
        cv2.polylines(shown, [r.corners.astype(np.int32)], True, (0, 220, 0), 2, cv2.LINE_AA)
        stem = f.stem.replace(" ", "_")
        cv2.imwrite(str(out / "normal" / f"{stem}.png"), n.document)
        cv2.imwrite(str(out / "relief" / f"{stem}.png"), r.document)
        cv2.imwrite(str(out / "corners" / f"{stem}.jpg"), shown)
        row = np.hstack([fit(shown, 250, 170), fit(r.rectified, 250, 170), fit(n.document, 280, 170), fit(r.document, 280, 170)])
        cv2.putText(row, f.name[:24], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1, cv2.LINE_AA)
        rows.append(row)
    head = np.full((22, rows[0].shape[1], 3), 255, np.uint8)
    for x, t in ((5, "original + corners"), (255, "straightened"), (505, "normal scan"), (785, "3D relief scan")):
        cv2.putText(head, t, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "sheet.jpg"), np.vstack([head] + rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{len(rows)} plates -> {out}")


if __name__ == "__main__":
    main()
