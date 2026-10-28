"""Scan every image in a folder (or tier sub-folders) with the OpenCV-only pipeline.

    python opencv/scripts/scan_plain.py "LPR catagory" --out eval/plain_scan

For each image it saves
    <tier>/document/<name>.png   the final plain page (white background, clean characters, level)
    <tier>/corners/<name>.jpg    the original with the 4 detected corners drawn on it
    <tier>/rectified/<name>.png  the straightened plate before cleaning
    <tier>/sheet.jpg             contact sheet: 3 pairs per row, straightened plate on the left, final plain page on the right
and writes results.csv (image, corner method, size). No OCR, no model, no training is used.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from opencv.scan import scan  # noqa: E402

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def tile(z: np.ndarray, w: int = 240, h: int = 120) -> np.ndarray:
    z = z if z.ndim == 3 else cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)
    s = min(w / z.shape[1], h / z.shape[0])
    z = cv2.resize(z, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    o = np.full((h, w, 3), 245, np.uint8)
    o[: z.shape[0], : z.shape[1]] = z
    return o


PAIRS_PER_ROW = 3  # (straightened plate | final page) pairs per row, as in eval/lpr_tiers_deblur


def run_folder(src: Path, out: Path, rows: list[dict], tier: str) -> None:
    for sub in ("document", "corners", "rectified"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for old in out.glob("sheet*.jpg"):
        old.unlink()
    tiles = []
    for f in sorted(p for p in src.iterdir() if p.suffix.lower() in EXT):
        img = cv2.imread(str(f))
        if img is None:
            continue
        r = scan(img)
        shown = img.copy()
        cv2.polylines(shown, [r.corners.astype(np.int32)], True, (0, 220, 0), max(1, img.shape[1] // 200), cv2.LINE_AA)
        for x, y in r.corners:
            cv2.circle(shown, (int(x), int(y)), max(2, img.shape[1] // 60), (0, 120, 255), -1, cv2.LINE_AA)
        cv2.imwrite(str(out / "document" / f"{f.stem}.png"), r.document)
        cv2.imwrite(str(out / "corners" / f"{f.stem}.jpg"), shown)
        cv2.imwrite(str(out / "rectified" / f"{f.stem}.png"), r.rectified)
        rows.append(dict(tier=tier, image=f.name, corner_method=r.method,
                         corners=";".join(f"{x:.0f},{y:.0f}" for x, y in r.corners),
                         document_px=f"{r.document.shape[1]}x{r.document.shape[0]}"))
        left = tile(r.rectified)
        cv2.putText(left, f.name[:16], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1, cv2.LINE_AA)
        tiles.append(np.hstack([left, tile(r.document)]))
    if tiles:  # same layout as eval/lpr_tiers_deblur/<tier>/sheet.jpg
        while len(tiles) % PAIRS_PER_ROW:
            tiles.append(np.full_like(tiles[0], 245))
        sheet = np.vstack([np.hstack(tiles[k: k + PAIRS_PER_ROW]) for k in range(0, len(tiles), PAIRS_PER_ROW)])
        cv2.imwrite(str(out / "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="a folder of images, or a folder whose sub-folders are tiers (easy/medium/extreme)")
    ap.add_argument("--out", default="eval/plain_scan")
    a = ap.parse_args()
    src, out = Path(a.folder), Path(a.out)
    rows: list[dict] = []
    subs = sorted(d for d in src.iterdir() if d.is_dir())
    if subs:
        for d in subs:
            run_folder(d, out / d.name, rows, d.name)
    else:
        run_folder(src, out, rows, src.name)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["tier", "image", "corner_method", "corners", "document_px"])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter

    print(f"{len(rows)} images scanned -> {out}")
    print("corner methods:", dict(Counter(r["corner_method"] for r in rows)))


if __name__ == "__main__":
    main()
