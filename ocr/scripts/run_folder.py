"""Scan every image in a folder and save plain-text plates, a results table and contact sheets.

Usage:
    python ocr/scripts/run_folder.py "LPR_200 1st" --out eval/lpr200

Writes, under --out:
    document/<name>.png  THE RESULT: the plain document - a clean white page with only the plate text
                         (typeset from the reading when it is reliable, otherwise the cleaned scan)
    scan/<name>.png      the faithful cleaned scan of the plate (always available)
    restored/<name>.png  restored colour plate
    results.csv          one row per image: plate found, source, quality, text, reliability, verdict
    sheet_N.jpg          contact sheets: original+detected plate | rectified | plain document
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ocr.scan import get_detector, get_ocr, scan_image  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def tile(img: np.ndarray, w: int = 200, h: int = 120, label: str | None = None) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = min(w / img.shape[1], h / img.shape[0])
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    out = np.full((h, w, 3), 35, np.uint8)
    out[: img.shape[0], : img.shape[1]] = img
    if label:
        cv2.putText(out, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="eval/folder_scan")
    ap.add_argument("--per-sheet", type=int, default=48)
    a = ap.parse_args()

    files = sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in EXT)
    out = Path(a.out)
    for sub in ("document", "typed", "scan", "restored"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    det, ocr = get_detector(), get_ocr()

    rows, tiles, t0 = [], [], time.time()
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            rows.append(dict(file=f.name, found=0, note="unreadable file"))
            continue
        res = scan_image(img, detector=det, ocr=ocr)
        if not res:
            rows.append(dict(file=f.name, found=0, note="no plate found"))
            tiles.append(np.hstack([tile(img, label=f.name[:18]), tile(np.zeros((8, 8, 3), np.uint8)),
                                    tile(np.zeros((8, 8, 3), np.uint8))]))
            continue
        r = max(res, key=lambda r: r.detection.confidence * r.detection.width_px)
        cv2.imwrite(str(out / "document" / f"{f.stem}.png"), r.document)
        cv2.imwrite(str(out / "scan" / f"{f.stem}.png"), r.scan_document)
        if r.typed is not None:
            cv2.imwrite(str(out / "typed" / f"{f.stem}.png"), r.typed)
        cv2.imwrite(str(out / "restored" / f"{f.stem}.png"), r.restored)
        rows.append(dict(file=f.name, found=len(res), source=r.detection.source, plate_width_px=round(r.detection.width_px),
                         quality=round(r.quality.score), quality_level=r.quality.level, issues=" ".join(r.quality.issues),
                         steps=" > ".join(r.steps), text=r.reading.text, script=r.reading.script,
                         document="typed" if r.document is not r.scan_document else "scan",
                         reliability=None if r.reading.reliability is None else round(r.reading.reliability, 2),
                         verdict=r.verdict, ms=round(sum(r.timings_ms.values()))))
        shown = img.copy()
        cv2.polylines(shown, [r.detection.quad.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
        tiles.append(np.hstack([tile(shown, label=f.name[:18]), tile(r.crop), tile(r.document, label=r.verdict)]))

    cols = sorted({k for r in rows for k in r})
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    for i in range(0, len(tiles), a.per_sheet):
        chunk = tiles[i: i + a.per_sheet]
        lines = [np.hstack(chunk[j: j + 2] + [np.zeros_like(chunk[0])] * (2 - len(chunk[j: j + 2])))
                 for j in range(0, len(chunk), 2)]
        cv2.imwrite(str(out / f"sheet_{i // a.per_sheet + 1}.jpg"), np.vstack(lines), [cv2.IMWRITE_JPEG_QUALITY, 88])

    found = sum(1 for r in rows if r.get("found"))
    print(f"{len(files)} images, plate found in {found}, {time.time() - t0:.0f}s total -> {out}")
    from collections import Counter

    print("verdicts:", dict(Counter(r.get("verdict", "none") for r in rows)))


if __name__ == "__main__":
    main()
