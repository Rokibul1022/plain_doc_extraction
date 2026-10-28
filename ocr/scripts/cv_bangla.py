"""K-fold cross-validated fine-tuning on verified labels: honest accuracy on plates never trained on.

    python ocr/scripts/cv_bangla.py labels.csv "LPR_200 1st" --init row_v2.pt --dataset syn.npz [--folds 4]
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

from ocr.bangla_reader import VIEWS, BanglaRowReader, region  # noqa: E402
from ocr.digits import normalize_digits  # noqa: E402
from ocr.scan import get_detector, scan_image  # noqa: E402


def load_labels(path: str) -> dict[str, str]:
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        d = normalize_digits(r.get("digits") or "")
        if len(d) == 6 and r.get("status", "labelled") != "unreadable":
            out[r["file"]] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("folder")
    ap.add_argument("--init", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--real-frac", type=float, default=0.4)
    ap.add_argument("--real-aug", action="store_true")
    a = ap.parse_args()

    labels = load_labels(a.labels)
    det = get_detector()
    crops = {}
    for name in labels:
        res = scan_image(cv2.imread(str(Path(a.folder) / name)), detector=det, do_ocr=False)
        if res:
            crops[name] = [res[0].crop, res[0].restored]
    names = sorted(crops)
    # group by plate number: several captures of one physical plate must land in the same fold, otherwise
    # the model is tested on a plate it effectively trained on
    fold = {n: int(hashlib.md5(labels[n].encode()).hexdigest(), 16) % a.folds for n in names}
    tmp = Path(tempfile.mkdtemp(prefix="cv_bn_"))
    ok = n_tot = dig = 0
    wrong = []
    for k in range(a.folds):
        tr = [n for n in names if fold[n] != k]
        te = [n for n in names if fold[n] == k]
        xs, ys = [], []
        for n in tr:
            for c in crops[n]:
                for top, trim in VIEWS:
                    xs.append(region(c, top, trim))
                    ys.append(labels[n])
        real = tmp / f"real_{k}.npz"
        np.savez(real, x=np.stack(xs), y=np.array(ys))
        onnx = tmp / f"m_{k}.onnx"
        proc = subprocess.run([sys.executable, str(Path(__file__).with_name("train_bangla_row.py")), "--steps", str(a.steps),
                        "--init", a.init, "--dataset", a.dataset, "--real", str(real), "--real-frac", str(a.real_frac),
                        "--lr", str(a.lr)] + (["--real-aug"] if a.real_aug else []) + ["--save", str(tmp / f"m_{k}.pt"), "--onnx-out", str(onnx)],
                       capture_output=True, text=True)
        if not onnx.exists():
            sys.exit("fold %d training failed (exit %s):\n%s\n%s" % (k, proc.returncode, proc.stdout[-1500:], proc.stderr[-2500:]))
        reader = BanglaRowReader(onnx)
        f_ok = 0
        for n in te:
            r = reader.read(crops[n])
            f_ok += r.text == labels[n]
            dig += sum(x == y for x, y in zip(r.text.ljust(6, "?"), labels[n]))
            if r.text != labels[n]:
                wrong.append((n, labels[n], r.text, round(r.confidence, 2)))
        ok += f_ok
        n_tot += len(te)
        print(f"fold {k}: held-out exact {f_ok}/{len(te)}", flush=True)
    print(f"\nCROSS-VALIDATED (each plate judged by a model that never saw it): exact {ok}/{n_tot} = {ok / n_tot:.0%}, "
          f"digit accuracy {dig / (6 * n_tot):.0%}")
    for w in wrong[:10]:
        print("  wrong:", w)
    print("models kept in", tmp)


if __name__ == "__main__":
    main()
