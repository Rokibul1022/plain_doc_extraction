# ocr - the learned (OCR) route

**How it works in detail: [ARCHITECTURE.md](ARCHITECTURE.md)** (pipeline, the five models, decision rules, training and evaluation).

Everything in this folder uses trained models. The OpenCV-only scan lives in [`../opencv`](../opencv/README.md);
this route imports the classical pieces it shares (corner geometry, restoration, layout) from there.

| File | Purpose |
|---|---|
| `detector.py` | learned plate detector (YOLO ONNX via `open-image-models`) with tiling + the classical fallback |
| `bangla_reader.py`, `bangla_row.onnx` | CRNN + CTC reader for the 6 Bangla digits of the number row |
| `bangla_top.py`, `bangla_top.onnx` | classifier for the top row (city / series letter) |
| `ocr.py` | Latin plate OCR (`fast-plate-ocr`) |
| `verdict.py`, `verdict_bn.py`, `verdict_*.json` | calibrated "is this reading reliable" score |
| `deblur.py` | OCR-guided deblurring |
| `video.py` | per-track fusion / voting over video frames |
| `scan.py` | `scan_image`: detect -> restore -> read -> verdict for one image |
| `typeset.py`, `textrender.py`, `fonts/` | typeset a recognised reading as a clean page (HarfBuzz + Noto Sans Bengali) |
| `synth_row.py`, `classic.py`, `degrade.py`, `digits.py`, `rowprep.py` | synthetic training data, template-matching experiment, degradations, digit helpers, row preparation |
| `api.py` | HTTP API (`/api/plate/*`) used by the Image and Video tabs of the main app |
| `scripts/` | training (`train_bangla_*.py`, GPU), cross-validation (`cv_*.py`), calibration, labelling page, tier evaluation |

```bash
pip install -r ocr/requirements.txt
python ocr/scripts/run_folder.py <folder of plates> --out results/
python ocr/scripts/eval_tiers.py <tier folder> --labels labels.csv --cv-dir DIR --calib-folder <folder> --out results_tiers/
python ocr/scripts/make_label_page.py <folder of plates> --out results/label.html     # label plates in the browser
```

Training data (hand-verified six-digit labels and the plate photos) is kept locally and is not part of the repository.
Cross-validation groups captures of one physical plate into the same fold, so a model is never tested on a plate it
has seen.

## Run it in Docker (CPU)

```bash
docker build -f Dockerfile.ocr -t plate-ocr .
docker run --rm -p 8001:8001 --cpus=4 -e WORKERS=2 -e ORT_INTRA_THREADS=2 plate-ocr
curl -F file=@car.jpg http://localhost:8001/api/plate/read       # GET /health, GET /api/plate/capabilities
```

The reader ONNX files are copied into the image and the YOLO detector (and the Latin plate OCR model) are downloaded while the image is built, so the running container needs no
internet. CPU use is roughly `WORKERS x ORT_INTRA_THREADS` threads: keep it at or below the CPUs you give the
container. Without Docker: `uvicorn ocr.server:app --port 8001 --workers 2`.

## Reading pipeline and the relief readers

`ocr/pipeline.py` ties the routes together:

```python
import cv2
from ocr.pipeline import read_plate, render_report

report = read_plate(cv2.imread("car.jpg"))     # YOLO (large frames) -> opencv scan -> readers
report.text, report.verdict                      # "ঢাকা মেট্রো-ছ  ১১-২৭৮৯", "readable|uncertain|unreadable" (no confidence numbers)
cv2.imwrite("result.png", render_report(report)) # cleaned plate + the reading + the verdict
```

Two reader pairs run on every plate and the digits of the more confident one win (the first row is named by the ink reader, falling back to the relief reader when it is unsure; `TOP_RULE` in `pipeline.py`): the shipped **ink** readers (`bangla_row.onnx`,
`bangla_top.onnx`) and **relief** readers (`bangla_row_relief.onnx`, `bangla_top_relief.onnx`) for embossed plates with no
ink. The relief readers look at `opencv.relief.relief_view` (the plate's shading with lighting and colour removed) and were
trained only on synthetic embossed plates (`ocr/synth_row.embossify`: a height map lit from a random direction plus wear,
grime and grain):

```bash
python ocr/scripts/train_bangla_row.py --style relief --steps 6000 --lr 1e-3 --samples 20000 --dataset syn_relief.npz
python ocr/scripts/train_bangla_top.py --style relief --steps 3000 --lr 1e-3 --samples 20000 --dataset syn_top_relief.npz
python ocr/scripts/read_plates.py <folder> --out results/ [--truth labels.csv]   # csv: file,digits[,city,letter]
python ocr/scripts/make_plain_label_page.py <folder>      # one HTML page to label the photos in the browser (exports that csv)
```

Training uses the GPU (`torch` CUDA); data generation uses the CPU (`GEN_PROCS` limits its processes). The top-row model knows
19 city words (the plain "ঢাকা" was added); its class-letter head is weak (about 42% on synthetic plates), so the letter is
shown only when it is confident. Thresholds in `pipeline.py` are heuristics, not calibrated on labelled embossed plates.

**Continue tuning:** the training weights are in [`weights/`](weights/README.md) and `scripts/finetune_on_labels.py` retrains the readers on new labelled plates.
