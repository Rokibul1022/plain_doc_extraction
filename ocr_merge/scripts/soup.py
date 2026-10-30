"""Average the weights of several fine-tuned models that started from the same synthetic weights ("model soup"): one model, same cost.

    python ocr_merge/scripts/soup.py OUT.pt A.pt B.pt [C.pt ...]
    python ocr_merge/scripts/soup.py --cv OUT_NAME A_NAME B_NAME [...]      # per fold: weights/cv/OUT_NAME/fold{k}.pt
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent


def average(paths: list[Path]) -> dict:
    sds = [torch.load(p, map_location="cpu") for p in paths]
    out = {}
    for k, v in sds[0].items():
        out[k] = sum(sd[k].float() for sd in sds) / len(sds) if v.dtype.is_floating_point else v
    return out


if __name__ == "__main__":
    if sys.argv[1] == "--cv":
        out, names = sys.argv[2], sys.argv[3:]
        d = ROOT / "ocr_merge" / "weights" / "cv"
        (d / out).mkdir(parents=True, exist_ok=True)
        for k in range(5):
            torch.save(average([d / n / f"fold{k}.pt" for n in names]), d / out / f"fold{k}.pt")
        print("wrote", d / out)
    else:
        torch.save(average([Path(p) for p in sys.argv[2:]]), sys.argv[1])
        print("wrote", sys.argv[1])
