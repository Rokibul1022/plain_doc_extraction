"""Read every plate image in a folder with the full pipeline and save the result pictures + a table.

    python ocr/scripts/read_plates.py "new samples+3d" --out eval/ocr_relief3d [--truth truth.csv]

Writes  <out>/results/<name>.png (cleaned plate + reading + verdict),  <out>/sheet.jpg,
<out>/results.csv. With --truth (CSV: file,digits[,city,letter]; Latin or Bengali numerals) the reading is also scored against it.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ocr.digits import normalize_digits  # noqa: E402
from ocr.pipeline import read_plate, render_report, use_models  # noqa: E402

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="eval/ocr_read")
    ap.add_argument("--truth", help="CSV file,digits[,top] to score against")
    for k in ("ink-row", "ink-top", "relief-row", "relief-top"):
        ap.add_argument(f"--{k}", help=f"use this ONNX file for the {k.replace('-', ' ')} reader (e.g. to evaluate held-out models)")
    a = ap.parse_args()
    over = {k.replace('-', '_'): getattr(a, k.replace('-', '_')) for k in ('ink-row', 'ink-top', 'relief-row', 'relief-top') if getattr(a, k.replace('-', '_'))}
    if over:
        use_models(**over)
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    truth = {}
    if a.truth:
        for r in csv.DictReader(open(a.truth, encoding="utf-8-sig")):
            truth[r["file"]] = r
    files = sorted(p for p in Path(a.folder).rglob("*") if p.suffix.lower() in EXT)
    rows, tiles = [], []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        t0 = time.perf_counter()
        rep = read_plate(img)
        dt = time.perf_counter() - t0
        name = f.relative_to(a.folder).as_posix().replace("/", "__")
        pic = render_report(rep)
        cv2.imwrite(str(out / "results" / f"{Path(name).stem}.png"), pic)
        row = {"file": f.relative_to(a.folder).as_posix(), "verdict": rep.verdict, "reader": rep.mode, "top": rep.top, "letter": rep.letter,
               "digits": rep.digits,
               "corners": rep.corner_method, "seconds": f"{dt:.1f}"}
        t = truth.get(f.name) or truth.get(row["file"])
        if t:
            want = normalize_digits(t["digits"])  # Latin or Bengali numerals both work
            row["truth"] = want
            row["digits_right"] = sum(x == y for x, y in zip(rep.digits, want))
            row["exact"] = int(rep.digits == want)
            if t.get("city"):
                row["city_label"], row["city_ok"] = t["city"], int(rep.top == t["city"])
            if t.get("letter"):
                row["letter_label"], row["letter_ok"] = t["letter"], int(rep.letter == t["letter"])
        rows.append(row)
        s = 520 / pic.shape[1]
        tiles.append(cv2.resize(pic, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        print(f"{row['file'][:34]:34} {rep.verdict:10} {rep.mode:6} {rep.text}")
    fields = list(rows[0].keys()) if rows else []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(out / "results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if tiles:
        hmax = max(t.shape[0] for t in tiles)
        tiles = [np.vstack([t, np.full((hmax - t.shape[0], t.shape[1], 3), 255, np.uint8)]) if t.shape[0] < hmax else t for t in tiles]
        if len(tiles) % 2:
            tiles.append(np.full_like(tiles[0], 255))
        sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        cv2.imwrite(str(out / "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{len(rows)} plates -> {out}")


if __name__ == "__main__":
    main()
