"""Test the merged reader on a folder of unseen plates (GPU), with cross-checks and, if labels are given, accuracy.

    python ocr_merge/scripts/test_folder.py merge_test --gpu [--truth merge_test_labels.csv] [--out eval/ocr_merge/merge_test]

Every plate is scanned once (YOLO for large frames + OpenCV scan). Then:
  merged   the merged model, on the GPU with --gpu (PyTorch/CUDA, ocr_merge/gpu.py), else ONNX on the CPU
  cpu      the deployed plate_net.onnx on the CPU - must agree with the GPU run
  separate the four separate readers on the CPU (the 5-model pipeline), for comparison
Without --truth there is no accuracy (nothing to compare with); you get the readings, the verdicts and how often the three
agree. With --truth (CSV file,digits[,city,letter]) the accuracy is computed, overall and per verdict, and plates whose number
is also in the training labels (labels.csv, labels_embossed10.csv) are counted separately: those would not be unseen.
Writes <out>/{merged,separate}/{results.csv,results/,sheet.jpg} and <out>/summary.json. (Reader speed GPU vs CPU: gpu_timing.py;
timing both in one process is not fair, torch and ONNX Runtime threads compete.)
"""

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import compare as C  # noqa: E402
from ocr.digits import normalize_digits  # noqa: E402
from ocr_merge.reader import MergedReader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--truth", help="CSV file,digits[,city,letter]")
    ap.add_argument("--gpu", action="store_true", help="run the merged model on the GPU (PyTorch/CUDA)")
    ap.add_argument("--out", default=str(ROOT / "eval" / "ocr_merge" / "merge_test"))
    a = ap.parse_args()
    out = Path(a.out)
    truth = C.load_truth(Path(a.truth)) if a.truth else {}
    items = [(p, p.relative_to(a.folder).as_posix(), truth.get(p.name)) for p in C.images(Path(a.folder))]
    print(f"{len(items)} images, {sum(t is not None for _, _, t in items)} with labels", flush=True)
    summary = {"folder": Path(a.folder).name, "images": len(items), "labelled": sum(t is not None for _, _, t in items)}

    if a.gpu:
        import torch

        from ocr_merge.gpu import use_gpu

        summary["device"] = use_gpu()
        summary["gpu_memory_mb"] = round(torch.cuda.memory_allocated() / 2 ** 20, 1)
        print("merged model on:", summary["device"], f"({summary['gpu_memory_mb']} MB allocated)", flush=True)
    else:
        summary["device"] = "cpu (ONNX Runtime)"
    summary["merged"] = C.run_set("", items, "merged", out)
    summary["separate"] = C.run_set("", items, "separate", out)

    # same scans, deployed ONNX on the CPU: must give the same readings as the GPU run (and its speed for comparison)
    cpu = MergedReader(ROOT / "ocr_merge" / "plate_net.onnx")
    gpu_rows = list(csv.DictReader(open(out / "merged" / "results.csv", encoding="utf-8-sig")))
    sep_rows = list(csv.DictReader(open(out / "separate" / "results.csv", encoding="utf-8-sig")))
    same_cpu = same_sep = 0
    for (p, shown, _), g, s in zip(items, gpu_rows, sep_rows):
        r, _ = C._scans[p]
        rep = C.sep.build_report(r, cpu.candidates([r.rectified, r.restored]))
        same_cpu += rep.digits == g["digits"] and rep.top == g["top"]
        same_sep += s["digits"] == g["digits"]
    summary["agreement"] = {"gpu_vs_cpu_onnx_same_reading": f"{same_cpu}/{len(items)}", "merged_vs_separate_same_digits": f"{same_sep}/{len(items)}"}
    summary["reader_ms_median_in_this_run"] = {"merged": summary["merged"]["reader_ms_median"], "separate_cpu": summary["separate"]["reader_ms_median"]}

    if truth:  # unseen check: a plate whose number is in the training labels is a twin of something the model learned
        seen = set()
        for f in (ROOT / "labels.csv", ROOT / "labels_embossed10.csv"):
            seen |= {normalize_digits(r["digits"]) for r in csv.DictReader(open(f, encoding="utf-8-sig"))}
        lab = [r for r in gpu_rows if "exact" in r]
        twin = [r for r in lab if r["truth"] in seen]
        new = [r for r in lab if r["truth"] not in seen]
        acc = lambda rs: {"plates": len(rs), "digits": f"{sum(int(r['digits_right']) for r in rs)}/{6 * len(rs)}",  # noqa: E731
                          "whole_number": f"{sum(int(r['exact']) for r in rs)}/{len(rs)}"}
        summary["accuracy"] = {"all_labelled": acc(lab), "number_in_training_labels": acc(twin), "truly_unseen": acc(new),
                               "by_verdict": {v: acc([r for r in lab if r["verdict"] == v]) for v in ("readable", "uncertain", "unreadable")}}
        for k in ("city", "letter_shown_right"):
            summary["accuracy"][k] = summary["merged"][k]
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("merged", "separate")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
