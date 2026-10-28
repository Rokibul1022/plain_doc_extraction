"""Evaluate the Bangla plate reader on the easy / medium / extreme folders and save everything.

    python ocr/scripts/eval_tiers.py "LPR catagory" --labels labels.csv --cv-dir DIR --calib-folder "LPR_200 1st" --out eval/lpr_tiers

Honesty rule: a plate you labelled is NEVER read by a model that trained on it. `--cv-dir` holds the four
cross-validation models written by `ocr/scripts/cv_bangla.py` (fold = md5(plate number) % 4, so every capture of
one physical plate shares a fold). Unlabelled plates have no ground truth: they are only counted as read /
not read, with no accuracy claimed.

Writes under --out
    results.csv        one row per image (truth, prediction, correctness, confidence, verdict, file paths)
    summary.csv        per tier: images, labelled, exact-match, per-digit, top-row, read rate, precision
    <tier>/document/   THE PLAIN DOCUMENT per plate (typeset page when reliable, else the aligned clean scan)
    <tier>/typed/      best-guess typeset page (always, so you can compare with the plate)
    <tier>/scan/       aligned clean scan of the actual plate
    <tier>/sheet.jpg   contact sheet: plate | plain document | reading
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ocr import verdict_bn as V  # noqa: E402
from ocr.bangla_reader import BanglaRowReader  # noqa: E402
from ocr.bangla_top import BanglaTopReader  # noqa: E402
from ocr.deblur import deblur_guided_bn  # noqa: E402
from ocr.digits import normalize_digits  # noqa: E402
from ocr.typeset import render_typed  # noqa: E402
from ocr.scan import bengali_text, get_detector, scan_image  # noqa: E402
from ocr.verdict import fit, verdict_from  # noqa: E402

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
FOLDS = 4


def fold_of(number: str) -> int:
    return int(hashlib.md5(number.encode()).hexdigest(), 16) % FOLDS


def load_labels(path: str) -> dict[str, dict]:
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        d = normalize_digits(r.get("digits") or "")
        st = r.get("status", "labelled")
        if st == "unreadable":
            out[r["file"]] = {"status": "unreadable"}
        elif len(d) == 6:
            out[r["file"]] = {"status": "labelled", "digits": d, "city": (r.get("city") or "").strip(),
                              "letter": (r.get("letter") or "").strip()}
    return out


USE_DEBLUR = True


def read_plate(res, reader: BanglaRowReader):
    """Digit reading of one scanned plate with the given reader; blur removal is tried first (judged by the
    same reader's own reliability, never by labels) and kept only when the reader trusts the result more."""
    base = reader.read([res.crop, res.restored])
    if not USE_DEBLUR:
        return base
    g = deblur_guided_bn(res.crop, reader, res.detection.width_px)
    if g is None:
        return base
    cand = reader.read([res.crop, res.restored, g[0]])
    return cand if (cand.ok and V.reliability(cand, res.detection.width_px) > V.reliability(base, res.detection.width_px)) else base


def calibrate(labels, folder, cv_readers, det) -> np.ndarray:
    """Fit the Bengali reliability model on honest (cross-validated) reads of every labelled plate."""
    xs, ys = [], []
    for name, lab in labels.items():
        if lab["status"] != "labelled":
            continue
        img = cv2.imread(str(Path(folder) / name))
        res = scan_image(img, detector=det, do_ocr=False) if img is not None else []
        if not res:
            continue
        row = read_plate(res[0], cv_readers[fold_of(lab["digits"])])
        xs.append(V.features(row, res[0].detection.width_px))
        ys.append(1.0 if row.text == lab["digits"] else 0.0)
    x, y = np.array(xs), np.array(ys)
    w = fit(x, y, 0.05)
    p = 1 / (1 + np.exp(-np.clip(x @ w, -30, 30)))
    V.MODEL_PATH.write_text(json.dumps({"features": V.FEATURES, "weights": [float(v) for v in w], "n": int(len(y)),
                                        "fitted_on": "cross-validated reads of verified real plates"}, indent=1))
    V._model = w
    print(f"calibrated reliability on {len(y)} verified plates (in-sample fit; reads were cross-validated)")
    for lo, hi in ((0, .3), (.3, .6), (.6, .8), (.8, .9), (.9, 1.01)):
        m = (p >= lo) & (p < hi)
        if m.sum():
            print(f"   P in [{lo:.1f},{hi:.1f}): n={int(m.sum()):3d}  observed accuracy {y[m].mean():.2f}")
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tiers_dir")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cv-dir", required=True)
    ap.add_argument("--calib-folder", required=True)
    ap.add_argument("--top-cv-dir", help="folder with top_0..3.onnx unseen-plate models for the top row (from cv_top.py)")
    ap.add_argument("--out", default="eval/lpr_tiers")
    ap.add_argument("--no-deblur", action="store_true")
    a = ap.parse_args()

    global USE_DEBLUR
    USE_DEBLUR = not a.no_deblur
    labels = load_labels(a.labels)
    cv_readers = [BanglaRowReader(Path(a.cv_dir) / f"m_{k}.onnx") for k in range(FOLDS)]
    shipped, top, det = BanglaRowReader(), BanglaTopReader(), get_detector()
    top_cv = [BanglaTopReader(Path(a.top_cv_dir) / f"top_{k}.onnx") for k in range(FOLDS)] if a.top_cv_dir else None
    calibrate(labels, a.calib_folder, cv_readers, det)

    out = Path(a.out)
    rows = []
    for tier in ("easy", "medium", "extreme"):
        files = sorted(p for p in (Path(a.tiers_dir) / tier).iterdir() if p.suffix.lower() in EXT)
        for sub in ("document", "typed", "scan"):
            (out / tier / sub).mkdir(parents=True, exist_ok=True)
        tiles = []
        for f in files:
            img = cv2.imread(str(f))
            res = scan_image(img, detector=det, do_ocr=False)
            lab = labels.get(f.name, {})
            row = dict(tier=tier, file=f.name, label_status=lab.get("status", "none"),
                       truth_digits=lab.get("digits", ""), truth_city=lab.get("city", ""), truth_letter=lab.get("letter", ""))
            if not res:
                row.update(plate_found=0, verdict="no plate found")
                rows.append(row)
                continue
            r = res[0]
            if lab.get("status") == "labelled":
                reader, source = cv_readers[fold_of(lab["digits"])], "cv-model-unseen"
            else:
                reader, source = shipped, "shipped-model"
            dr = read_plate(r, reader)
            rel = V.reliability(dr, r.detection.width_px)
            verdict = V.verdict(rel) if dr.ok else "unreadable"
            top_reader = top_cv[fold_of(lab["digits"])] if (top_cv and lab.get("status") == "labelled") else top
            tr = top_reader.read([r.crop, r.restored])  # what the page shows (low-confidence parts omitted)
            tr_raw = top_reader.read([r.crop, r.restored], min_conf=0.0)  # used for scoring only
            digits_bn = dr.text if dr.ok else ""
            typed = None
            if digits_bn:
                try:
                    typed = render_typed([tr.text, bengali_text(digits_bn)] if tr.text else [bengali_text(digits_bn)])
                except Exception:
                    typed = None
            document = typed if (verdict == "readable" and typed is not None) else r.scan_document
            cv2.imwrite(str(out / tier / "document" / f"{f.stem}.png"), document)
            cv2.imwrite(str(out / tier / "scan" / f"{f.stem}.png"), r.scan_document)
            if typed is not None:
                cv2.imwrite(str(out / tier / "typed" / f"{f.stem}.png"), typed)

            truth = lab.get("digits", "")
            exact = (digits_bn == truth) if truth else ""
            dig = (sum(x == y for x, y in zip(digits_bn.ljust(6, "?"), truth)) / 6) if truth else ""
            city_ok = (tr_raw.city.replace(" ", "") == lab["city"].replace(" ", "")) if lab.get("city") else ""
            letter_ok = (tr_raw.letter == lab["letter"].replace(" ", "")) if lab.get("letter") else ""
            row.update(plate_found=1, source=source, pred_digits=digits_bn, pred_top=tr.text, exact=exact if exact == "" else int(exact),
                       digit_acc=dig if dig == "" else round(dig, 3), city_correct=city_ok if city_ok == "" else int(city_ok),
                       letter_correct=letter_ok if letter_ok == "" else int(letter_ok), confidence=round(dr.confidence, 3),
                       reliability=round(rel, 3), verdict=verdict, document="typed" if document is typed and typed is not None else "scan",
                       plate_px=round(r.detection.width_px))
            rows.append(row)

            def tile(z, w=240, h=120):
                z = z if z.ndim == 3 else cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)
                s = min(w / z.shape[1], h / z.shape[0])
                z = cv2.resize(z, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                o = np.full((h, w, 3), 245, np.uint8)
                o[: z.shape[0], : z.shape[1]] = z
                return o

            label = f"{f.name[:12]} {verdict[:4]}" + ("" if exact == "" else (" OK" if exact else " X"))
            t1 = tile(r.crop)
            cv2.putText(t1, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
            tiles.append(np.hstack([t1, tile(document)]))
        if tiles:
            while len(tiles) % 3:
                tiles.append(np.full_like(tiles[0], 245))
            sheet = np.vstack([np.hstack(tiles[k: k + 3]) for k in range(0, len(tiles), 3)])
            cv2.imwrite(str(out / tier / "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

    cols = sorted({k for r in rows for k in r}, key=lambda k: ["tier", "file", "label_status", "truth_digits", "pred_digits", "exact",
                                                               "digit_acc", "truth_city", "truth_letter", "pred_top", "city_correct",
                                                               "letter_correct", "verdict", "reliability", "confidence", "document",
                                                               "plate_found", "source", "plate_px"].index(k)
                  if k in ["tier", "file", "label_status", "truth_digits", "pred_digits", "exact", "digit_acc", "truth_city",
                           "truth_letter", "pred_top", "city_correct", "letter_correct", "verdict", "reliability", "confidence",
                           "document", "plate_found", "source", "plate_px"] else 99)
    with open(out / "results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    summ = []
    for tier in ("easy", "medium", "extreme", "ALL"):
        rs = [r for r in rows if tier == "ALL" or r["tier"] == tier]
        lab = [r for r in rs if r.get("label_status") == "labelled" and r.get("plate_found")]
        ex = [r for r in lab if r.get("exact") != ""]
        rd = [r for r in rs if r.get("verdict") == "readable"]
        rd_lab = [r for r in rd if r in ex]
        city = [r for r in lab if r.get("city_correct") != ""]
        let = [r for r in lab if r.get("letter_correct") != ""]
        summ.append(dict(
            tier=tier, images=len(rs), labelled=len(ex),
            exact_match_pct=round(100 * np.mean([r["exact"] for r in ex]), 1) if ex else "",
            digit_accuracy_pct=round(100 * np.mean([r["digit_acc"] for r in ex]), 1) if ex else "",
            city_accuracy_pct=round(100 * np.mean([r["city_correct"] for r in city]), 1) if city else "",
            letter_accuracy_pct=round(100 * np.mean([r["letter_correct"] for r in let]), 1) if let else "",
            called_readable=len(rd), read_rate_pct=round(100 * len(rd) / len(rs), 1) if rs else "",
            precision_when_readable_pct=round(100 * np.mean([r["exact"] for r in rd_lab]), 1) if rd_lab else "",
            labelled_readable=len(rd_lab)))
    with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    print()
    for s in summ:
        print(s)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
