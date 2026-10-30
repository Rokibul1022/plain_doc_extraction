# ocr_merge - the four Bengali readers merged into one model

`plate_net.onnx` (1.54 M parameters, 5.9 MB) replaces `ocr/bangla_row.onnx`, `bangla_top.onnx`, `bangla_row_relief.onnx`
and `bangla_top_relief.onnx` (5.32 M, 20.3 MB). Everything else (YOLO, the OpenCV scan, the decision rules) is `ocr/`.

```python
from ocr_merge.pipeline import read_plate, render_report
report = read_plate(cv2.imread("car.jpg"))     # same PlateReport as ocr.pipeline.read_plate
```

| File | Purpose |
|---|---|
| `model.py` | PlateNet (torch, training only): shared CNN -> CTC digit head + city/letter head; ONNX export |
| `reader.py` | run time: one batched ONNX call for both looks (ink, relief_view) and both rows of a plate |
| `pipeline.py` | `read_plate` with the merged reader (settings above) |
| `prior.py`, `priors.json`, `decode.py`, `views.py`, `cv.py`, `gpu.py` | class priors (and their counts), optional six-digit decoder, view sets, grouped folds, GPU session |
| `scripts/train_merged.py` | synthetic pretraining and fine-tuning on the labels (with hold-outs) |
| `scripts/finetune_all.sh` | the deployed model + the held-out models used for the evaluation |
| `scripts/compare.py`, `scripts/benchmark.py` | separate vs. merged: accuracy (eval/ocr_merge) and size / memory / latency |
| `weights/` | .pt weights, held-out models, cached scans of the labelled plates (local only, not pushed) |

## Latest model (v2)
`plate_net.onnx` is trained on all 171 labelled plates. Settings (environment variables, defaults in brackets):
`MERGE_VIEWS` (20) digit-row views per crop, `MERGE_RETRY_VIEWS` (55; 0 = off) denser re-read of plates that are not "readable",
`MERGE_PRIOR` (1) class priors from `priors.json`, `MERGE_DEVICE` (auto: the GPU when PyTorch + CUDA + `weights/plate_net.pt` exist, else
the ONNX model on the CPU; `cpu` forces the CPU), `MERGE_CHUNK` (48) views per model call (memory).
Fast setting: `MERGE_VIEWS=6 MERGE_RETRY_VIEWS=0 MERGE_PRIOR=0`. Cross-validated accuracy: [../eval/ocr_merge/cv171/README.md](../eval/ocr_merge/cv171/README.md).

## What to share
`ocr/`, `opencv/` and `ocr_merge/` (it contains `plate_net.onnx` and `priors.json`) plus `requirements.txt`; the `.onnx` file alone is only the network.
Not needed by others: `ocr_merge/weights/` (training weights and cached crops of the real plates; keep it private).

Results: [../eval/ocr_merge/README.md](../eval/ocr_merge/README.md).
