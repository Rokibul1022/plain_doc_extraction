"""Fit and validate the plate-reading reliability model.

Runs the full system on real labelled frames plus the synthetic CCTV tiers, records the verdict
features and whether the read was exactly right, fits `verdict_model.json` by logistic regression
and reports cross-validated calibration, so `READABLE_AT` / `UNCERTAIN_AT` rest on evidence.

Usage: python ocr/scripts/calibrate_verdict.py DATA_DIR [--limit N] [--no-write] [--records out.json]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plate_bench import iou, load  # noqa: E402

from ocr import verdict as V  # noqa: E402
from ocr.degrade import TIERS, degrade_crop  # noqa: E402
from ocr.detector import PlateDetector, rectify  # noqa: E402
from ocr.ocr import PlateOCR  # noqa: E402
from ocr.scan import read_plate_crop  # noqa: E402


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def collect(rows, det, ocr):
    xs, ys, src = [], [], []
    for f, x, y, w, h, truth in rows:
        img = cv2.imread(f)
        d = max(det.detect(img), key=lambda d: iou(d.box, (x, y, x + w, y + h)), default=None)
        if d is None or iou(d.box, (x, y, x + w, y + h)) < 0.3:
            continue
        crop = rectify(img, d.quad)
        _, _, q0, _, _, reading, _ = read_plate_crop(crop, plate_width=d.width_px, ocr=ocr)
        xs.append(V.features(reading, q0)); ys.append(reading.text == truth); src.append("real")
    for tier in TIERS:
        rng = np.random.default_rng(21)
        for f, x, y, w, h, truth in rows:
            img = cv2.imread(f)
            pad = int(0.12 * max(w, h))
            crop = img[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad]
            dg = degrade_crop(crop, tier, rng, w)
            pw = w * dg.shape[1] / crop.shape[1]
            _, _, q0, _, _, reading, _ = read_plate_crop(dg, plate_width=pw, ocr=ocr)
            xs.append(V.features(reading, q0)); ys.append(reading.text == truth); src.append(f"syn-{tier}")
    return np.array(xs), np.array(ys, dtype=float), np.array(src)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--records")
    a = ap.parse_args()

    x, y, src = collect(load(a.data, a.limit), PlateDetector(), PlateOCR())
    print(f"{len(y)} records ({', '.join(f'{s}:{(src == s).sum()}' for s in sorted(set(src)))}), exact {y.mean():.2f}")
    if a.records:
        Path(a.records).write_text(json.dumps({"x": x.tolist(), "y": y.tolist(), "src": src.tolist()}))

    # 5-fold cross-validated predictions
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, 5)
    cv_p = np.zeros(len(y))
    for k in range(5):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(5) if j != k])
        w = V.fit(x[train], y[train])
        cv_p[test] = 1 / (1 + np.exp(-np.clip(x[test] @ w, -30, 30)))

    base = x[:, 1]  # raw mean OCR confidence
    print(f"\nAUC  raw OCR confidence {auc(base, y):.3f}   |   cross-validated reliability {auc(cv_p, y):.3f}")
    print("\ncross-validated calibration (predicted P vs observed accuracy)")
    for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .85), (.85, .95), (.95, 1.01)):
        m = (cv_p >= lo) & (cv_p < hi)
        if m.sum():
            print(f"  P in [{lo:.2f},{hi:.2f}): n={m.sum():4d}  mean P {cv_p[m].mean():.2f}  observed {y[m].mean():.2f}")
    for thr in (0.5, 0.7, 0.85, 0.9, 0.95):
        m = cv_p >= thr
        print(f"  flag 'readable' at P>={thr:.2f}: {m.sum():4d} flagged, precision {y[m].mean() if m.sum() else float('nan'):.3f}, "
              f"recall {y[m].sum() / max(y.sum(), 1):.2f}")
    for s in sorted(set(src)):
        m = src == s
        print(f"  {s:12} exact {y[m].mean():.2f}  mean reliability {cv_p[m].mean():.2f}  AUC {auc(cv_p[m], y[m]):.3f}")

    w = V.fit(x, y)
    print("\nweights:", {n: round(float(v), 2) for n, v in zip(V.FEATURES, w)})
    if not a.no_write:
        V.MODEL_PATH.write_text(json.dumps({"features": V.FEATURES, "weights": [float(v) for v in w],
                                            "n": int(len(y)), "cv_auc": auc(cv_p, y)}, indent=1))
        print("wrote", V.MODEL_PATH)


if __name__ == "__main__":
    main()
