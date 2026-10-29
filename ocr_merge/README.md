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
| `pipeline.py` | `read_plate` with the merged reader |
| `scripts/train_merged.py` | synthetic pretraining and fine-tuning on the labels (with hold-outs) |
| `scripts/finetune_all.sh` | the deployed model + the held-out models used for the evaluation |
| `scripts/compare.py`, `scripts/benchmark.py` | separate vs. merged: accuracy (eval/ocr_merge) and size / memory / latency |
| `weights/` | .pt weights, held-out models, cached scans of the labelled plates (local only, not pushed) |

Results: [../eval/ocr_merge/README.md](../eval/ocr_merge/README.md).
