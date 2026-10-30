"""Reader-step latency of the merged model: CPU (ONNX Runtime) vs GPU (PyTorch CUDA), on the same scanned plates.

    python ocr_merge/scripts/gpu_timing.py merge_test --n 20 --out eval/ocr_merge/merge_test/gpu_timing.json

Only the model call (all views of one plate in one batch, plus the voting) is timed, not YOLO or the OpenCV scan, which stay on
the CPU. The CPU run happens before torch is imported, so the two do not compete for threads.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from opencv.scan import scan  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()
    files = sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})[: a.n]
    crops = []
    for f in files:
        r = scan(cv2.imread(str(f)), relief="off", make_page=False)
        crops.append([r.rectified, r.restored])
    from ocr_merge.pipeline import reader

    rd = reader()
    rd.candidates(crops[0])  # load + first call

    def bench() -> dict:
        t = []
        for _ in range(a.repeat):
            for c in crops:
                s = time.perf_counter()
                rd.candidates(c)
                t.append(time.perf_counter() - s)
        return {"median_ms": round(1000 * float(np.median(t)), 1), "p90_ms": round(1000 * float(np.percentile(t, 90)), 1), "n": len(t)}

    res = {"plates": len(crops), "cpu_onnx": bench()}
    from ocr_merge.gpu import use_gpu

    res["gpu_device"] = use_gpu()
    rd.candidates(crops[0])
    rd.candidates(crops[-1])  # warm-up (CUDA kernels)
    res["gpu_torch"] = bench()
    res["speedup"] = round(res["cpu_onnx"]["median_ms"] / res["gpu_torch"]["median_ms"], 1)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
