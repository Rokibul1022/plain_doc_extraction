"""Continue tuning the readers on labelled plates (the digit-row and first-row models, ink or embossed).

    python ocr/scripts/finetune_on_labels.py labels.csv "<image folder>" --style ink    --out tuned_ink/
    python ocr/scripts/finetune_on_labels.py labels.csv "<image folder>" --style relief  --out tuned_relief/ --init-dir ocr/weights/base_synthetic

labels.csv columns: file,digits[,city,letter]  (digits Latin or Bengali; city = a word the first-row model knows; letter optional).
Starts from the weights in --init-dir (default ocr/weights/current, i.e. what is deployed; use ocr/weights/base_synthetic to
re-tune from the synthetic-only start, which does not carry over plates the deployed models were already tuned on) and writes
<out>/<style>_row.{pt,onnx} and <out>/<style>_top.{pt,onnx}. Copy the .onnx files to ocr/ to deploy them.

Honest testing: pass --holdout a.png,b.png to keep test plates out of training (any other photo with the same plate number is
dropped too), then score them with ocr/scripts/read_plates.py using --ink-row/--ink-top/--relief-row/--relief-top.
Training needs torch (GPU recommended). The synthetic plates are read from `synthetic data/` when present, else generated once
(about 4 minutes) and cached in --out.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ocr.bangla_reader import VIEWS, region  # noqa: E402
from ocr.digits import normalize_digits  # noqa: E402
from ocr.synth_row import CITY_WORDS, LETTERS, to_top_region  # noqa: E402
from opencv.relief import relief_view  # noqa: E402
from opencv.scan import scan  # noqa: E402

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def shifted(c: np.ndarray, s: float) -> np.ndarray:
    h = c.shape[0]
    if s > 0:
        return c[int(s * h):]
    if s < 0:
        return cv2.copyMakeBorder(c, int(-s * h), 0, 0, 0, cv2.BORDER_REPLICATE)
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels")
    ap.add_argument("folder")
    ap.add_argument("--style", choices=("ink", "relief"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-dir", default=str(ROOT / "ocr" / "weights" / "current"))
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", default="3e-4")
    ap.add_argument("--holdout", default="", help="comma separated file names kept out of training")
    ap.add_argument("--dataset-row", help="cached synthetic digit-row plates (npz); created if missing")
    ap.add_argument("--dataset-top", help="cached synthetic first-row plates (npz); created if missing")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    rows = [r for r in csv.DictReader(open(a.labels, encoding="utf-8-sig")) if len(normalize_digits(r.get("digits") or "")) == 6
            and r.get("status", "labelled") != "unreadable"]
    held = {x.strip() for x in a.holdout.split(",") if x.strip()}
    held_numbers = {normalize_digits(r["digits"]) for r in rows if r["file"] in held}
    train = [r for r in rows if r["file"] not in held and normalize_digits(r["digits"]) not in held_numbers]
    paths = {p.name: p for p in Path(a.folder).rglob("*") if p.suffix.lower() in EXT}
    train = [r for r in train if r["file"] in paths]
    print(f"{len(rows)} labelled | {len(held)} held out | training on {len(train)} plates", flush=True)
    if not train:
        sys.exit("nothing to train on")

    view = relief_view if a.style == "relief" else (lambda c: c)
    xr, yr, xt, ct, lt = [], [], [], [], []
    for i, r in enumerate(train):
        s = scan(cv2.imread(str(paths[r["file"]])), relief="off", make_page=False)
        digits = normalize_digits(r["digits"])
        city, letter = (r.get("city") or "").strip(), (r.get("letter") or "").strip()
        ci = CITY_WORDS.index(city) if city in CITY_WORDS else -1
        li = LETTERS.index(letter) if letter in LETTERS else -1
        for c in (s.rectified, s.restored):
            v = view(c)
            for top, trim in VIEWS:
                xr.append(region(v, top, trim)); yr.append(digits)
            for sh in (-0.03, 0.0, 0.03):
                xt.append(to_top_region(view(shifted(c, sh)))); ct.append(ci); lt.append(li)
        if i % 20 == 0:
            print(f"  prepared {i + 1}/{len(train)}", flush=True)
    np.savez(out / "real_row.npz", x=np.stack(xr), y=np.array(yr))
    np.savez(out / "real_top.npz", x=np.stack(xt), city=np.array(ct), letter=np.array(lt))

    init = Path(a.init_dir)
    st = a.style
    stored = ROOT / "synthetic data"  # the saved 20,000-plate sets, if present; otherwise they are generated into --out
    ds_row = a.dataset_row or str(stored / f"{st}_row_20k.npz" if (stored / f"{st}_row_20k.npz").exists() else out / f"syn_{st}_row.npz")
    ds_top = a.dataset_top or str(stored / f"{st}_top_20k.npz" if (stored / f"{st}_top_20k.npz").exists() else out / f"syn_{st}_top.npz")
    common = ["--style", st, "--steps", str(a.steps), "--lr", a.lr, "--real-frac", "0.35", "--eval-n", "60"]
    row = subprocess.Popen([py, "-u", "-W", "ignore", str(ROOT / "ocr/scripts/train_bangla_row.py"), *common, "--real-aug", "--init", str(init / f"{st}_row.pt"),
                            "--dataset", ds_row, "--real", str(out / "real_row.npz"), "--save", str(out / f"{st}_row.pt"), "--onnx-out", str(out / f"{st}_row.onnx")],
                           stdout=open(out / f"train_{st}_row.log", "w"), stderr=subprocess.STDOUT)
    top = subprocess.Popen([py, "-u", "-W", "ignore", str(ROOT / "ocr/scripts/train_bangla_top.py"), *common, "--init", str(init / f"{st}_top.pt"),
                            "--dataset", ds_top, "--real", str(out / "real_top.npz"), "--save", str(out / f"{st}_top.pt"), "--onnx-out", str(out / f"{st}_top.onnx")],
                           stdout=open(out / f"train_{st}_top.log", "w"), stderr=subprocess.STDOUT)
    row.wait()
    top.wait()
    for kind in ("row", "top"):
        f = out / f"{st}_{kind}.onnx"
        if not f.exists():
            sys.exit(f"{kind} training failed: see {out / f'train_{st}_{kind}.log'}")
        print(f"wrote {f}")
    print("done. Deploy by copying the .onnx files over ocr/bangla_{row,top}[_relief].onnx and the .pt files into ocr/weights/current/.")


if __name__ == "__main__":
    main()
