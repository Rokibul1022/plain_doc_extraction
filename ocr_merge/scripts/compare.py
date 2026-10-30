"""Separate readers vs. merged reader on the same images, scored against the labels. Writes eval/ocr_merge/.

    python ocr_merge/scripts/compare.py                       # all sets below
    python ocr_merge/scripts/compare.py --only lpr new3d

Every image is scanned once (YOLO for large frames + the OpenCV scan); both pipelines then read that same scan, so any
difference comes from the readers alone. Sets:
  lpr        LPR catagory (easy / medium / extreme), labels.csv          - deployed models (both have seen these labels)
  new3d      new samples+3d, labels_embossed10.csv                      - deployed models (both have seen these labels)
  heldout_extreme   the 26 labelled extreme plates, models tuned WITHOUT them (and their twins)            - honest
  heldout_embossed  the 10 embossed plates, 5 folds, each tested by models tuned without that fold        - honest
                    (separate: relief readers + the ink top reader re-tuned per fold, see separate_ink_top_folds.py)
Per set and pipeline: results.csv, results/<image>.png, sheet.jpg; overall summary.json + summary.md.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import ocr.pipeline as sep  # noqa: E402
import ocr_merge.pipeline as mer  # noqa: E402
from ocr.digits import normalize_digits  # noqa: E402
from opencv.scan import scan as cv_scan  # noqa: E402

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
W = ROOT / "ocr_merge" / "weights"
SEP_HELDOUT = W / "heldout_separate"  # the separate readers' held-out models from the earlier tests (eval/ocr_extreme, eval/ocr_relief3d)
EMB_FOLDS = [["46d4aeaf-f0ce-4362-82e4-4d1495067caa.jpeg", "Screenshot_5.png"], ["Screenshot_6.png", "Screenshot_7.png"],
             ["Screenshot_10.png", "Screenshot_11.png", "Screenshot_11 (1).png"], ["Screenshot_12.png", "Screenshot_13.png"], ["Screenshot_14.png"]]


def load_truth(path: Path) -> dict:
    return {r["file"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


def images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXT)


_scans: dict = {}


def scanned(p: Path):
    if p not in _scans:
        t = time.perf_counter()
        r = cv_scan(sep.locate_plate(cv2.imread(str(p))), relief="off", make_page=False)
        _scans[p] = (r, time.perf_counter() - t)
    return _scans[p]


def read(pipe: str, r):
    t = time.perf_counter()
    if pipe == "merged":  # the same code the app runs (views, class priors, denser re-read of doubtful plates)
        rep = mer.read_scan(r)
        return rep, time.perf_counter() - t
    views = [r.rectified, r.restored]
    cands = {m: sep._candidate(m, views) for m in ("ink", "relief")}
    t_read = time.perf_counter() - t
    return sep.build_report(r, cands), t_read


def score(rep, t: dict | None) -> dict:
    if not t or len(normalize_digits(t.get("digits") or "")) != 6:
        return {}
    want = normalize_digits(t["digits"])
    s = {"truth": want, "digits_right": sum(x == y for x, y in zip(rep.digits, want)), "exact": int(rep.digits == want)}
    city, letter = (t.get("city") or "").strip(), (t.get("letter") or "").strip()
    if city:
        s["city_label"], s["city_ok"] = city, int(rep.top == city)
    if letter:
        s["letter_label"], s["letter_ok"] = letter, int(rep.letter == letter)
    return s


def run_set(name: str, items: list[tuple[Path, str, dict | None]], pipe: str, out: Path) -> dict:
    """items: (image, shown name, truth row). Returns the totals."""
    d = out / name / pipe
    (d / "results").mkdir(parents=True, exist_ok=True)
    rows, tiles, t_read, t_scan = [], [], [], []
    for p, shown, truth in items:
        r, ts = scanned(p)
        rep, tr = read(pipe, r)
        t_scan.append(ts)
        t_read.append(tr)
        pic = sep.render_report(rep)
        cv2.imwrite(str(d / "results" / f"{Path(shown.replace('/', '__')).stem}.png"), pic)
        rows.append({"file": shown, "verdict": rep.verdict, "reader": rep.mode, "top": rep.top, "letter": rep.letter, "digits": rep.digits,
                     "reader_ms": round(1000 * tr, 1), **score(rep, truth)})
        s = 520 / pic.shape[1]
        tiles.append(cv2.resize(pic, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(d / "results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if tiles:
        hmax = max(t.shape[0] for t in tiles)
        tiles = [np.vstack([t, np.full((hmax - t.shape[0], t.shape[1], 3), 255, np.uint8)]) for t in tiles]
        if len(tiles) % 2:
            tiles.append(np.full_like(tiles[0], 255))
        cv2.imwrite(str(d / "sheet.jpg"), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    lab = [r for r in rows if "exact" in r]
    tot = {"images": len(rows), "labelled": len(lab),
           "digits": f"{sum(r['digits_right'] for r in lab)}/{6 * len(lab)}", "whole_number": f"{sum(r['exact'] for r in lab)}/{len(lab)}",
           "city": f"{sum(r['city_ok'] for r in lab if 'city_ok' in r)}/{sum('city_ok' in r for r in lab)}",
           "letter_shown_right": f"{sum(r['letter_ok'] for r in lab if 'letter_ok' in r)}/{sum('letter_ok' in r for r in lab)}",
           "verdicts": {v: sum(r["verdict"] == v for r in rows) for v in ("readable", "uncertain", "unreadable")},
           "readable_and_right": f"{sum(r['exact'] for r in lab if r['verdict'] == 'readable')}/{sum(r['verdict'] == 'readable' for r in lab)}",
           "reader_ms_median": round(1000 * float(np.median(t_read)), 1) if t_read else 0,
           "scan_s_median": round(float(np.median(t_scan)), 2) if t_scan else 0}
    print(f"  {name:18} {pipe:9} digits {tot['digits']:>8}  whole {tot['whole_number']:>6}  city {tot['city']:>6}  "
          f"readable-right {tot['readable_and_right']:>6}  reader {tot['reader_ms_median']} ms", flush=True)
    return tot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "eval" / "ocr_merge"))
    ap.add_argument("--only", nargs="*", default=["lpr", "new3d", "heldout_extreme", "heldout_embossed"])
    a = ap.parse_args()
    out = Path(a.out)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    labels = load_truth(ROOT / "labels.csv")
    emb = load_truth(ROOT / "labels_embossed10.csv")

    if "lpr" in a.only:
        folder = ROOT / "LPR catagory"
        items = [(p, p.relative_to(folder).as_posix(), labels.get(p.name)) for p in images(folder)]
        summary["lpr"] = {pipe: run_set("lpr", items, pipe, out) for pipe in ("separate", "merged")}
    if "new3d" in a.only:
        folder = ROOT / "new samples+3d"
        items = [(p, p.relative_to(folder).as_posix(), emb.get(p.name)) for p in images(folder)]
        summary["new3d"] = {pipe: run_set("new3d", items, pipe, out) for pipe in ("separate", "merged")}
    if "heldout_extreme" in a.only:
        test = load_truth(ROOT / "labels_extreme_test.csv")
        folder = ROOT / "LPR catagory" / "extreme"
        items = [(p, p.name, test[p.name]) for p in images(folder) if p.name in test]
        sep.use_models(ink_row=SEP_HELDOUT / "ink_row_extreme_out.onnx", ink_top=SEP_HELDOUT / "ink_top_extreme_out.onnx")
        mer.use_model(W / "plate_net_extreme_out.onnx")
        summary["heldout_extreme"] = {pipe: run_set("heldout_extreme", items, pipe, out) for pipe in ("separate", "merged")}
        sep.use_models(ink_row=ROOT / "ocr" / "bangla_row.onnx", ink_top=ROOT / "ocr" / "bangla_top.onnx")
        mer.use_model(mer.MODEL_PATH)
    if "heldout_embossed" in a.only:
        paths = {p.name: p for p in images(ROOT / "new samples+3d")}
        res = {}
        for pipe in ("separate", "merged"):
            rows_all = []
            for k, fold in enumerate(EMB_FOLDS):
                sep.use_models(relief_row=SEP_HELDOUT / f"relief_row_fold{k}_out.onnx", relief_top=SEP_HELDOUT / f"relief_top_fold{k}_out.onnx",
                               ink_top=SEP_HELDOUT / f"ink_top_fold{k}_out.onnx")  # the deployed ink top reader saw these plates
                mer.use_model(W / f"plate_net_emb_fold{k}_out.onnx")
                rows_all.append(run_set(f"heldout_embossed/fold{k}", [(paths[f], f, emb[f]) for f in fold], pipe, out))
            n = lambda key: [tuple(map(int, t[key].split("/"))) for t in rows_all]  # noqa: E731
            res[pipe] = {key: f"{sum(x for x, _ in n(key))}/{sum(y for _, y in n(key))}" for key in ("digits", "whole_number", "city", "letter_shown_right")}
            print(f"  heldout_embossed   {pipe:9} {res[pipe]}", flush=True)
        sep.use_models(relief_row=ROOT / "ocr" / "bangla_row_relief.onnx", relief_top=ROOT / "ocr" / "bangla_top_relief.onnx",
                       ink_top=ROOT / "ocr" / "bangla_top.onnx")
        mer.use_model(mer.MODEL_PATH)
        summary["heldout_embossed"] = res
    out.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
