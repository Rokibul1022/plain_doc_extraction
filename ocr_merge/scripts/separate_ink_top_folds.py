"""Held-out baseline for compare.py: the separate ink top-row reader re-tuned with each embossed fold left out.

The deployed `ocr/bangla_top.onnx` was tuned on all labels including the 10 embossed plates, and the pipeline names the
first row with it ("ink_first"), so scoring the embossed folds with it would be memorised. This re-runs its recipe
(base_synthetic start, all labelled plates through the ink look, 800 steps, lr 3e-4) once per fold without that fold:

    python ocr_merge/scripts/separate_ink_top_folds.py      # -> ocr_merge/weights/heldout_separate/ink_top_fold{k}_out.onnx
"""

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from compare import EMB_FOLDS  # noqa: E402
from ocr.synth_row import to_top_region  # noqa: E402
from train_merged import shifted  # noqa: E402

W = ROOT / "ocr_merge" / "weights"
OUT = W / "heldout_separate"


def main() -> None:
    plates = pickle.load(open(W / "real_plates.pkl", "rb"))
    for k, fold in enumerate(EMB_FOLDS):
        numbers = {plates[f]["digits"] for f in fold}
        train = [f for f in plates if f not in fold and plates[f]["digits"] not in numbers]
        xt, ct, lt = [], [], []
        for f in train:
            for c in plates[f]["crops"]:
                for sh in (-0.03, 0.0, 0.03):
                    xt.append(to_top_region(shifted(c, sh)))
                    ct.append(plates[f]["city"])
                    lt.append(plates[f]["letter"])
        real = OUT / f"real_ink_top_fold{k}.npz"
        np.savez(real, x=np.stack(xt), city=np.array(ct), letter=np.array(lt))
        onnx = OUT / f"ink_top_fold{k}_out.onnx"
        log = open(OUT / f"train_ink_top_fold{k}.log", "w")
        subprocess.run([sys.executable, "-u", "-W", "ignore", str(ROOT / "ocr/scripts/train_bangla_top.py"), "--style", "ink", "--steps", "800",
                        "--lr", "3e-4", "--real-frac", "0.35", "--eval-n", "60", "--init", str(ROOT / "ocr/weights/base_synthetic/ink_top.pt"),
                        "--dataset", str(ROOT / "synthetic data/ink_top_20k.npz"), "--real", str(real),
                        "--save", str(OUT / f"ink_top_fold{k}_out.pt"), "--onnx-out", str(onnx)], stdout=log, stderr=subprocess.STDOUT)
        real.unlink()
        print(f"fold {k}: trained on {len(train)} plates ->", onnx.name if onnx.exists() else "FAILED", flush=True)


if __name__ == "__main__":
    main()
