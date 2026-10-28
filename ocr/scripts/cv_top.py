"""Cross-validated fine-tuning of the top-row reader (city word + class letter) on verified labels.

    python ocr/scripts/cv_top.py labels.csv "LPR_200 1st" --init top.pt --dataset syn_top.npz [--final]

Folds are grouped by plate number (same split as cv_bangla.py). With --final the model is also trained on
every label and exported as the shipped `bangla_top.onnx`.
"""

import argparse
import csv
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ocr.bangla_top import BanglaTopReader  # noqa: E402
from ocr.digits import normalize_digits  # noqa: E402
from ocr.scan import get_detector, scan_image  # noqa: E402
from ocr.synth_row import CITY_WORDS, LETTERS, to_top_region  # noqa: E402

CITY_IDX = {c.replace(" ", ""): i for i, c in enumerate(CITY_WORDS)}
LETTER_IDX = {c: i for i, c in enumerate(LETTERS)}
JITTER = (0.0, 0.03, -0.03)


def views(crops: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for c in crops:
        h = c.shape[0]
        for j in JITTER:
            t = int(max(0, j * h))
            out.append(to_top_region(c[t:] if j >= 0 else c))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("folder")
    ap.add_argument("--init", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--real-frac", type=float, default=0.45)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()

    labels = {}
    for r in csv.DictReader(open(a.labels, encoding="utf-8-sig")):
        d = normalize_digits(r.get("digits") or "")
        if len(d) == 6 and r.get("status", "labelled") != "unreadable":
            labels[r["file"]] = (d, r.get("city", "").replace(" ", "").strip(), r.get("letter", "").replace(" ", "").strip())
    det = get_detector()
    crops, truth = {}, {}
    for name, (d, city, letter) in labels.items():
        res = scan_image(cv2.imread(str(Path(a.folder) / name)), detector=det, do_ocr=False)
        if res:
            crops[name] = [res[0].crop, res[0].restored]
            truth[name] = (d, CITY_IDX.get(city, -1), LETTER_IDX.get(letter, -1))
    names = sorted(crops)
    print(f"{len(names)} plates; with known city {sum(truth[n][1] >= 0 for n in names)}, known letter {sum(truth[n][2] >= 0 for n in names)}")
    fold = {n: int(hashlib.md5(truth[n][0].encode()).hexdigest(), 16) % 4 for n in names}
    tmp = Path(tempfile.mkdtemp(prefix="cv_top_"))
    train_py = str(Path(__file__).with_name("train_bangla_top.py"))

    def train(train_names: list[str], tag: str) -> Path:
        xs, cs, ls = [], [], []
        for n in train_names:
            v = views(crops[n])
            xs += v
            cs += [truth[n][1]] * len(v)
            ls += [truth[n][2]] * len(v)
        real = tmp / f"real_{tag}.npz"
        np.savez(real, x=np.stack(xs), city=np.array(cs), letter=np.array(ls))
        onnx = tmp / f"top_{tag}.onnx"
        proc = subprocess.run([sys.executable, train_py, "--steps", str(a.steps), "--init", a.init, "--dataset", a.dataset,
                               "--real", str(real), "--real-frac", str(a.real_frac), "--lr", str(a.lr),
                               "--save", str(tmp / f"top_{tag}.pt"), "--onnx-out", str(onnx)], capture_output=True, text=True)
        if not onnx.exists():
            sys.exit(f"training {tag} failed:\n{proc.stdout[-800:]}\n{proc.stderr[-1500:]}")
        return onnx

    c_ok = c_n = l_ok = l_n = 0
    for k in range(4):
        te = [n for n in names if fold[n] == k]
        reader = BanglaTopReader(train([n for n in names if fold[n] != k], str(k)))
        for n in te:
            r = reader.read(crops[n], min_conf=0.0)
            if truth[n][1] >= 0:
                c_n += 1
                c_ok += r.city.replace(" ", "") == CITY_WORDS[truth[n][1]].replace(" ", "")
            if truth[n][2] >= 0:
                l_n += 1
                l_ok += r.letter == LETTERS[truth[n][2]]
        print(f"fold {k} done", flush=True)
    print(f"\nTOP ROW, plates never seen in training: city {c_ok}/{c_n} = {c_ok / c_n:.0%}   letter {l_ok}/{l_n} = {l_ok / l_n:.0%}")

    if a.final:
        onnx = train(names, "final")
        target = Path(__file__).resolve().parent.parent / "bangla_top.onnx"
        for ext in ("", ".data"):
            src = Path(str(onnx) + ext)
            if src.exists():
                (Path(str(target) + ext)).write_bytes(src.read_bytes())
        print("shipped top-row model updated:", target)


if __name__ == "__main__":
    main()
