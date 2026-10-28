"""Build a single-file HTML labelling page for a folder of plate crops.

    python ocr/scripts/make_label_page.py "LPR_200 1st" --out eval/lpr200/label.html

Open the page in any browser (no server needed). Each plate is shown large with the digit region
zoomed (the model's guess is hidden on purpose). Type the true six digits (Bengali or Latin numerals - both
work), or tick "unreadable". Progress autosaves in the browser; "Download labels.csv" writes
`file,digits,city,letter,status` for `ocr/scripts/calibrate_bangla.py` and fine-tuning.
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ocr.bangla_reader import region  # noqa: E402
from ocr.scan import get_detector, scan_image  # noqa: E402
from ocr.synth_row import CITY_WORDS, LETTERS  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def b64(img: np.ndarray, q: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


HTML = (Path(__file__).with_name("label_page_template.html")).read_text(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="label.html")
    ap.add_argument("--skip-labelled", help="labels.csv: leave out plates that already have a label or an unreadable mark")
    a = ap.parse_args()

    files = sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in EXT)
    if a.skip_labelled:
        import csv

        done = {r["file"] for r in csv.DictReader(open(a.skip_labelled, encoding="utf-8-sig")) if r.get("status") in ("labelled", "unreadable")}
        files = [f for f in files if f.name not in done]
        print(f"{len(done)} already labelled; {len(files)} left")
    det = get_detector()
    data = []
    for f in files:
        img = cv2.imread(str(f))
        res = scan_image(img, detector=det, do_ocr=False) if img is not None else []
        if not res:
            continue
        crop = res[0].crop
        full = cv2.resize(crop, None, fx=480 / crop.shape[1], fy=480 / crop.shape[1], interpolation=cv2.INTER_CUBIC)
        zoom = cv2.resize(cv2.normalize(region(crop, 0.36, 0.0), None, 0, 255, cv2.NORM_MINMAX), (480, 180), interpolation=cv2.INTER_CUBIC)
        data.append({"file": f.name, "full": b64(full), "zoom": b64(zoom)})
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__CITIES__", json.dumps(CITY_WORDS, ensure_ascii=False))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(html, encoding="utf-8")
    print(f"{len(data)} plates -> {a.out} ({Path(a.out).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
