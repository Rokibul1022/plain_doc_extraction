"""Plate benchmark on labelled real frames (OpenALPR benchmark format) and synthetic tiers.

Dataset format: for every `X.jpg` a sibling `X.txt` holding `name<TAB>x<TAB>y<TAB>w<TAB>h<TAB>TEXT`.

Real frames are tiered by the measured quality of the ground-truth plate crop (easy / medium /
extreme), so the tier reflects how hard the plate actually is. `--synthetic` additionally runs the
CCTV degradation simulator over the same plates.

Usage:
    python ocr/scripts/plate_bench.py DATA_DIR [--limit N] [--synthetic] [--json out.json]
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ocr.degrade import TIERS, degrade_crop  # noqa: E402
from ocr.detector import PlateDetector, rectify  # noqa: E402
from ocr.ocr import PlateOCR  # noqa: E402
from opencv.quality import assess_quality  # noqa: E402
from ocr.scan import read_plate_crop  # noqa: E402


def load(data_dir: str, limit: int):
    rows = []
    for f in sorted(glob.glob(str(Path(data_dir) / "*.jpg")))[:limit]:
        meta = Path(f).with_suffix(".txt")
        if not meta.exists():
            continue
        t = meta.read_text().strip().split("\t")
        rows.append((f, *map(int, t[1:5]), t[5]))
    return rows


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def tier_of(score: float) -> str:
    return "easy" if score >= 62 else "medium" if score >= 40 else "extreme"


def char_acc(pred: str, truth: str) -> float:
    return sum(a == b for a, b in zip(pred.ljust(len(truth)), truth)) / max(1, len(truth))


def real_frames(rows, detector, ocr):
    stats = {t: dict(n=0, det=0, gt_raw=0, raw=0, full=0, ms=0.0, readable=0, wrong_readable=0) for t in TIERS}
    detail = []
    for f, x, y, w, h, truth in rows:
        img = cv2.imread(f)
        gt_box = (x, y, x + w, y + h)
        pad = int(0.12 * max(w, h))
        gt_crop = img[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad]
        q = assess_quality(gt_crop, w)
        tier = tier_of(q.score)
        s = stats[tier]
        s["n"] += 1
        s["gt_raw"] += ocr.read(gt_crop).text == truth  # OCR ceiling: perfect detection, no restoration

        t0 = time.perf_counter()
        dets = detector.detect(img)
        match = max(dets, key=lambda d: iou(d.box, gt_box), default=None)
        if match is not None and iou(match.box, gt_box) >= 0.3:
            s["det"] += 1
            crop = rectify(img, match.quad)
            s["raw"] += ocr.read(crop).text == truth
            restored, readable, q0, q1, steps, reading, _ = read_plate_crop(crop, plate_width=match.width_px, ocr=ocr)
            ok = reading.text == truth
            s["full"] += ok
            from ocr.scan import verdict_for

            v = verdict_for(reading, q0, q1)
            s["readable"] += v == "readable"
            s["wrong_readable"] += (v == "readable") and not ok
            detail.append(dict(file=Path(f).name, tier=tier, quality=round(q.score, 1), truth=truth,
                               pred=reading.text, conf=round(reading.confidence, 2), verdict=v,
                               steps=steps, ok=bool(ok)))
        s["ms"] += (time.perf_counter() - t0) * 1000
    return stats, detail


def synthetic(rows, ocr, n_per_tier):
    out = {}
    for tier in TIERS:
        rng = np.random.default_rng(11)
        hit = dict(raw=0, restored=0)
        sel = rows[:n_per_tier]
        for f, x, y, w, h, truth in sel:
            img = cv2.imread(f)
            pad = int(0.12 * max(w, h))
            crop = img[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad]
            d = degrade_crop(crop, tier, rng, w)
            pw = w * d.shape[1] / crop.shape[1]
            hit["raw"] += ocr.read(d).text == truth
            *_, reading, _ = read_plate_crop(d, plate_width=pw, ocr=ocr)
            hit["restored"] += reading.text == truth
        out[tier] = {k: v / len(sel) for k, v in hit.items()} | {"n": len(sel)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()

    rows = load(a.data, a.limit)
    detector, ocr = PlateDetector(), PlateOCR()
    stats, detail = real_frames(rows, detector, ocr)

    print(f"\nREAL FRAMES ({len(rows)} images; tier = measured quality of the true plate)")
    print(f"{'tier':8}{'n':>4}{'detected':>10}{'OCR ceiling*':>14}{'raw crop':>10}{'full system':>13}{'verdict=readable':>18}{'(wrong)':>9}{'ms/img':>8}")
    for t, s in stats.items():
        n = max(1, s["n"])
        print(f"{t:8}{s['n']:>4}{s['det']/n:>10.2f}{s['gt_raw']/n:>14.2f}{s['raw']/n:>10.2f}{s['full']/n:>13.2f}"
              f"{s['readable']/n:>18.2f}{s['wrong_readable']:>9}{s['ms']/n:>8.0f}")
    print("*OCR on the true-box crop with no restoration = upper bound for detection + rectification")
    tot = sum(s["n"] for s in stats.values())
    print(f"overall: detect {sum(s['det'] for s in stats.values())/tot:.2f}  full-system exact {sum(s['full'] for s in stats.values())/tot:.2f}")

    result = {"real": stats, "detail": detail}
    if a.synthetic:
        syn = synthetic(rows, ocr, min(len(rows), 60))
        print("\nSYNTHETIC CCTV TIERS (OCR exact-match on degraded true-plate crops)")
        for t, r in syn.items():
            print(f"{t:8} raw {r['raw']:.2f}  restored {r['restored']:.2f}  (n={r['n']})")
        result["synthetic"] = syn
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
