# ocr - the learned (OCR) route

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
curl -F file=@car.jpg http://localhost:8001/api/plate/scan       # GET /health, GET /api/plate/capabilities
```

The YOLO detector and the plate OCR model are downloaded while the image is built, so the running container needs no
internet. CPU use is roughly `WORKERS x ORT_INTRA_THREADS` threads: keep it at or below the CPUs you give the
container. Without Docker: `uvicorn ocr.server:app --port 8001 --workers 2`.
