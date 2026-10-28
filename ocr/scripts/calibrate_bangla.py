"""Measure the Bengali digit reader on verified labels and fit its reliability model.

Inputs
  --labels labels.csv   columns: file,digits   (digits = the six Bengali digits, verified by a person;
                        plates a human cannot read are simply not listed)
  --folder DIR          the folder those files live in
  --negatives DIR       optional folder of non-Bengali plates (OpenALPR format); the reader cannot read
                        these, so any six-digit output for them is a false read that the reliability
                        model must learn to score low

Prints exact-match accuracy (all plates, and only those the system calls "readable"), a
cross-validated calibration table, and writes `verdict_bn_model.json`.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ocr import verdict_bn as V  # noqa: E402
from ocr.bangla_reader import BanglaRowReader  # noqa: E402
from ocr.digits import normalize_digits  # noqa: E402
from ocr.scan import get_detector, scan_image  # noqa: E402


def collect(folder: str, names: list[str], truth: list[str | None], reader: BanglaRowReader, det):
    xs, ys, rows = [], [], []
    for name, t in zip(names, truth):
        img = cv2.imread(str(Path(folder) / name))
        if img is None:
            continue
        res = scan_image(img, detector=det, do_ocr=False)
        if not res:
            continue
        crop, width = res[0].crop, res[0].detection.width_px
        row = reader.read(crop)
        xs.append(V.features(row, width))
        ys.append(1.0 if (t is not None and row.text == t) else 0.0)
        rows.append((name, t, row.text, row.confidence))
    return xs, ys, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", help="verified labels.csv for real Bengali plates (optional)")
    ap.add_argument("--folder", help="folder the labelled files live in")
    ap.add_argument("--synthetic", type=int, default=0, help="also use N held-out synthetic plates (exact labels)")
    ap.add_argument("--negatives")
    ap.add_argument("--neg-limit", type=int, default=120)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()

    reader, det = BanglaRowReader(), get_detector()
    xs, ys, rows = [], [], []
    if a.labels:
        labels = {}
        for r in csv.DictReader(open(a.labels, encoding="utf-8-sig")):
            d = normalize_digits(r.get("digits") or "")
            if len(d) == 6 and r.get("status", "labelled") != "unreadable":
                labels[r["file"]] = d
        names = sorted(labels)
        xs, ys, rows = collect(a.folder, names, [labels[n] for n in names], reader, det)
    if a.synthetic:
        from ocr.synth_row import degraded_crop, render_plate

        rng = np.random.default_rng(987654)  # held out: training used other seeds
        ok_syn = 0
        for _ in range(a.synthetic):
            plate, label, _ = render_plate(rng)
            crop = degraded_crop(plate, rng)
            row = reader.read(crop)
            xs.append(V.features(row, float(rng.uniform(50, 160))))
            ys.append(1.0 if row.text == label else 0.0)
            ok_syn += row.text == label
        print(f"{a.synthetic} synthetic plates: exact {ok_syn / a.synthetic:.3f}")
    n_pos = len(ys)

    if a.negatives:
        neg = sorted(Path(a.negatives).glob("*.jpg"))[: a.neg_limit]
        xn, yn, _ = collect(a.negatives, [p.name for p in neg], [None] * len(neg), reader, det)
        xs += xn
        ys += yn
        print(f"{len(yn)} non-Bengali negatives")
    x, y = np.array(xs), np.array(ys)

    real = np.arange(len(y)) < n_pos
    ok = y[real]
    print(f"\nexact match (all {n_pos} verified plates): {ok.mean():.3f}")
    chars = [sum(a_ == b_ for a_, b_ in zip(r[2].ljust(6, "?"), r[1])) / 6 for r in rows]
    print(f"digit accuracy: {np.mean(chars):.3f}")

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, 5)
    cv = np.zeros(len(y))
    for k in range(5):
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        w = _fit(x[tr], y[tr])
        cv[folds[k]] = 1 / (1 + np.exp(-np.clip(x[folds[k]] @ w, -30, 30)))
    print("\ncross-validated reliability on the Bengali plates (predicted vs observed)")
    for lo, hi in ((0, .2), (.2, .5), (.5, .8), (.8, .9), (.9, .95), (.95, 1.01)):
        m = real & (cv >= lo) & (cv < hi)
        if m.sum():
            print(f"  P [{lo:.2f},{hi:.2f}): n={m.sum():3d}  observed {y[m].mean():.2f}")
    for thr in (0.5, 0.8, 0.9, 0.95):
        m = real & (cv >= thr)
        fa = (~real & (cv >= thr)).sum()
        print(f"  'readable' at P>={thr:.2f}: {m.sum():3d}/{max(n_pos,1)} Bengali plates called readable "
              f"({m.sum() / max(n_pos,1):.0%}), precision {y[m].mean() if m.sum() else float('nan'):.3f}; non-Bengali false reads {fa}")
    w = _fit(x, y)
    print("weights:", {k: round(float(v), 2) for k, v in zip(V.FEATURES, w)})
    if not a.no_write:
        V.MODEL_PATH.write_text(json.dumps({"features": V.FEATURES, "weights": [float(v) for v in w], "n": int(len(y))}, indent=1))
        print("wrote", V.MODEL_PATH)


def _fit(x: np.ndarray, y: np.ndarray, l2: float = 0.05) -> np.ndarray:
    from ocr.verdict import fit

    return fit(x, y, l2)


if __name__ == "__main__":
    main()
