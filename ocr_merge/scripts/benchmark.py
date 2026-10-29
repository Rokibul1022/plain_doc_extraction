"""Before / after: the four separate readers vs. the merged reader - size, parameters, load time, memory, latency.

    python ocr_merge/scripts/benchmark.py --images "LPR catagory/extreme" --n 20 --out eval/ocr_merge/benchmark.json

Each setup runs in its own fresh process (so memory numbers are not mixed). Measured per setup:
  - model files on disk and parameter count
  - load time (create the ONNX sessions + the first call) and process memory before / after loading / after reading
  - reader latency: the reader step alone on already-scanned plates (all views of one plate), median and p90
  - end-to-end latency: `read_plate` on the image (YOLO for large frames, OpenCV scan, readers, decision, page), and the
    same without the display page (its cost depends on which look won, not on the readers)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SEPARATE = ["ocr/bangla_row.onnx", "ocr/bangla_top.onnx", "ocr/bangla_row_relief.onnx", "ocr/bangla_top_relief.onnx"]
MERGED = ["ocr_merge/plate_net.onnx"]


def rss_mb() -> tuple[float, float]:
    """(current, peak) resident memory of this process in MB."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.WorkingSetSize / 2 ** 20, c.PeakWorkingSetSize / 2 ** 20
    import resource

    cur = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 20
    return cur, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def params(files: list[str]) -> int:
    import onnx
    import numpy as np

    return int(sum(np.prod(t.dims) for f in files for t in onnx.load(str(ROOT / f)).graph.initializer if len(t.dims) > 0))


def worker(mode: str, images: list[str], repeat: int) -> dict:
    import cv2
    import numpy as np

    from opencv.scan import scan as cv_scan

    mem0 = rss_mb()[0]
    crops = []
    for f in images:  # scanning first, so the reader timing below sees exactly the same crops in both setups
        from ocr.pipeline import locate_plate

        r = cv_scan(locate_plate(cv2.imread(f)), relief="off", make_page=False)
        crops.append([r.rectified, r.restored])
    mem_scan = rss_mb()[0]

    t0 = time.perf_counter()
    if mode == "separate":
        from ocr.pipeline import _candidate, read_plate

        def readers(views):
            return {m: _candidate(m, views) for m in ("ink", "relief")}
    else:
        from ocr_merge.pipeline import read_plate, reader

        def readers(views):
            return reader().candidates(views)
    readers(crops[0])  # loads the sessions + first call
    load_s = time.perf_counter() - t0
    mem_loaded = rss_mb()[0]

    lat = []
    for _ in range(repeat):
        for v in crops:
            t = time.perf_counter()
            readers(v)
            lat.append(time.perf_counter() - t)
    e2e = []
    for f in images:
        img = cv2.imread(f)
        t = time.perf_counter()
        read_plate(img)
        e2e.append(time.perf_counter() - t)
    import ocr.pipeline as sep  # the same again without the display page (make_document 1.6 s / relief_page 0.2 s per plate
    sep.make_document = sep.relief_page = lambda x: x  # depending on which look won), so only the reading is timed
    e2e_np = []
    for f in images:
        img = cv2.imread(f)
        t = time.perf_counter()
        read_plate(img)
        e2e_np.append(time.perf_counter() - t)
    cur, peak = rss_mb()
    files = SEPARATE if mode == "separate" else MERGED
    return {"mode": mode, "model_files": files, "disk_mb": round(sum((ROOT / f).stat().st_size for f in files) / 2 ** 20, 2),
            "params": params(files), "onnx_sessions": len(files), "calls_per_plate": 4 if mode == "separate" else 1,
            "load_s": round(load_s, 3), "memory_mb": {"start": round(mem0, 1), "after_scan": round(mem_scan, 1),
                                                      "readers_loaded": round(mem_loaded, 1), "readers_delta": round(mem_loaded - mem_scan, 1),
                                                      "after_reading": round(cur, 1), "peak": round(peak, 1)},
            "reader_ms": {"median": round(1000 * float(np.median(lat)), 1), "p90": round(1000 * float(np.percentile(lat, 90)), 1),
                          "mean": round(1000 * float(np.mean(lat)), 1), "n": len(lat)},
            "end_to_end_s": {"median": round(float(np.median(e2e)), 2), "p90": round(float(np.percentile(e2e, 90)), 2),
                             "mean": round(float(np.mean(e2e)), 2), "n": len(e2e)},
            "end_to_end_without_page_s": {"median": round(float(np.median(e2e_np)), 2), "p90": round(float(np.percentile(e2e_np, 90)), 2),
                                          "mean": round(float(np.mean(e2e_np)), 2), "n": len(e2e_np)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", default=[str(ROOT / "LPR catagory" / "extreme")])
    ap.add_argument("--n", type=int, default=20, help="images per folder")
    ap.add_argument("--repeat", type=int, default=5, help="reader-only passes over the scanned plates")
    ap.add_argument("--out", default=str(ROOT / "eval" / "ocr_merge" / "benchmark.json"))
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--files", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        print("RESULT " + json.dumps(worker(a.worker, json.loads(Path(a.files).read_text(encoding="utf-8")), a.repeat)))
        return
    images = []
    for folder in a.images:
        images += sorted(str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in EXT)[: a.n]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lst = out.with_suffix(".images.json")
    lst.write_text(json.dumps(images), encoding="utf-8")
    res = {"images": len(images), "cpu_threads": os.cpu_count(), "ORT_INTRA_THREADS": os.environ.get("ORT_INTRA_THREADS", "default"),
           "ORT_ALLOW_SPINNING": os.environ.get("ORT_ALLOW_SPINNING", "default")}
    for mode in ("separate", "merged"):
        p = subprocess.run([sys.executable, "-W", "ignore", __file__, "--worker", mode, "--files", str(lst), "--repeat", str(a.repeat)],
                           capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            sys.exit(f"{mode} benchmark failed:\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}")
        res[mode] = json.loads(line[7:])
        print(mode, json.dumps(res[mode], indent=1))
    lst.unlink()
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
