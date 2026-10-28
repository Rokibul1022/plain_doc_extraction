"""Build the real-plate fine-tuning/evaluation set from verified labels.

    python ocr/scripts/make_real_set.py FOLDER labels.csv --out real_set.npz [--fold K --folds 5]

Each verified plate contributes several views of its digit region (different top/trim offsets - the
same views the reader votes over - so the model sees how the real crop varies). `--fold K` holds out
every plate whose hash falls in fold K (written to `<out>_test.npz`) for honest accuracy measurement.
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ocr.bangla_reader import VIEWS, region  # noqa: E402
from ocr.digits import normalize_digits  # noqa: E402
from ocr.scan import get_detector, scan_image  # noqa: E402


def fold_of(name: str, folds: int) -> int:
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % folds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("labels")
    ap.add_argument("--out", default="real_set.npz")
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    labels = {}
    for r in csv.DictReader(open(a.labels, encoding="utf-8-sig")):
        d = normalize_digits(r.get("digits") or "")
        if len(d) == 6 and r.get("status", "labelled") != "unreadable":
            labels[r["file"]] = d
    det = get_detector()
    tx, ty, ex, ey, en = [], [], [], [], []
    for name, digits in sorted(labels.items()):
        img = cv2.imread(str(Path(a.folder) / name))
        res = scan_image(img, detector=det, do_ocr=False) if img is not None else []
        if not res or len(digits) != 6:
            continue
        crop = res[0].crop
        views = [region(crop, top, trim) for top, trim in VIEWS]
        if a.fold >= 0 and fold_of(name, a.folds) == a.fold:
            ex += views
            ey += [digits] * len(views)
            en += [name] * len(views)
        else:
            tx += views
            ty += [digits] * len(views)
    out = Path(a.out)
    np.savez(out, x=np.stack(tx), y=np.array(ty))
    print(f"train: {len(ty) // len(VIEWS)} plates / {len(ty)} views -> {out}")
    if ex:
        test = out.with_name(out.stem + "_test.npz")
        np.savez(test, x=np.stack(ex), y=np.array(ey), name=np.array(en))
        print(f"held out: {len(ey) // len(VIEWS)} plates -> {test}")


if __name__ == "__main__":
    main()
