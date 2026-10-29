# Bangladeshi License-Plate Scanner

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-classical%20CV-orange.svg)](https://opencv.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-web%20app-009688.svg)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU-5C2D91.svg)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Turns real-world CCTV frames and car photos of Bangladeshi license plates into **clean, straightened, plain
black-on-white pages** that are easy to read, with an optional OCR reading on top.

The project has two independent routes that share one web app:

| Route | Folder | Technology | Needs models / training |
|---|---|---|---|
| **OpenCV scan** | [`opencv/`](opencv/README.md) | OpenCV + NumPy only | **No** - classical image processing |
| **OCR route** | [`ocr/`](ocr/README.md) | YOLO plate detector + Bangla digit reader (ONNX) | Yes (models included) |

```
 frame / photo
      │
      ▼
 ┌─────────────────┐   ┌──────────────────────────────────────┐   ┌────────────────┐
 │ 1. YOLO detects │──►│ 2. OpenCV scan                       │──►│ 4. OCR reads   │
 │    the plate    │   │    corners → straighten → clean      │   │    the plate   │
 │  ocr/detector   │   │    → plain page   (opencv/)          │   │   (ocr/)       │
 └─────────────────┘   └──────────────────────────────────────┘   └────────────────┘
```

The two routes are usable on their own: the OpenCV scan works on any plate image (full frame or tight crop) without
any model, and the OCR route can run without the plain-page step.

---

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [The OpenCV scan](#the-opencv-scan-opencv)
- [The OCR route](#the-ocr-route-ocr)
- [Web app and HTTP API](#web-app-and-http-api)
- [Docker and production](#docker-and-production)
- [Configuration reference](#configuration-reference)
- [Testing](#testing)
- [Evaluation and limitations](#evaluation-and-limitations)
- [Data and privacy](#data-and-privacy)
- [Third-party components](#third-party-components)
- [License](#license)

---

## Features

**OpenCV scan** (`opencv/`, no OCR, no neural network, no training)
- Plate **corner detection** from colour clustering, seeded region growing and edge-verified line fitting, with a
  character-containment check; Canny + contour search for wide shots. Deterministic: the same image always gives
  the same corners.
- **Perspective correction** to the plate's true aspect ratio, sub-pixel corner refinement.
- **Restoration**: low-light lift, glare removal, de-blocking, non-local-means denoising, upscaling, contrast stretch.
- **Plain page generation**: colour projection that separates ink from plate body (works on teal-on-teal plates),
  illumination flattening, Wiener **deblurring** from a blur estimate measured on the strokes, hysteresis
  thresholding, frame / bolt / rule removal, per-row deskew and de-slant, smooth stroke rebuild.
- **Dedicated top-row processing** for the small Bangla city / series-letter row (re-extraction, thinning to print
  weight, counter re-opening, gap bridging).
- **3D relief mode** for embossed plates stamped in the plate colour (no ink): reads the light and shadow of the
  raised characters and draws them as clear outlines.

**OCR route** (`ocr/`)
- YOLO-v9 plate detector (ONNX) with tiling for small plates in large frames.
- Bangla number-row reader (CRNN + CTC) and top-row classifier, Latin plate OCR, calibrated reliability verdict
  (`readable` / `uncertain` / `unreadable`).
- OCR-guided deblurring, video tracking with multi-frame voting and fusion.

**Web app and operations**
- One-page UI with **Image**, **Video** and **OpenCV** tabs (including an *Embossed (3D) plate* switch).
- Stand-alone CPU services and Docker images for each route, with thread / worker settings for predictable CPU use.

---

## Quick start

```bash
git clone https://github.com/Rokibul1022/plain_doc_extraction.git
cd plain_doc_extraction
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # everything; or opencv/requirements.txt for the scan only
```

**Web app**

```bash
python run.py                    # opens http://127.0.0.1:8000
```

**OpenCV scan from Python**

```python
import cv2
from opencv import scan

result = scan(cv2.imread("plate.jpg"))      # relief="force" for embossed plates without ink
cv2.imwrite("plain.png", result.document)   # uint8, white paper, black characters, 960 px wide
print(result.corners, result.method)        # (4, 2) corners in the input image, how they were found
```

**OpenCV scan from the command line**

```bash
python -m opencv.cli plate.jpg --out out/                  # writes *_corners.jpg, *_straight.png, *_plain.png
python -m opencv.cli ./plates --out out/ --relief force
```

---

## Repository layout

```
.
├── opencv/                  OpenCV-only plate scan (self-contained, own README / requirements / tests / CLI)
│   ├── scan.py                entry point: scan(img) -> corners, straightened plate, plain page
│   ├── quadfit.py             edge-verified plate quad, seeded region growing, border-line fitting
│   ├── plate_find.py          colour-cluster hull, sub-pixel corners, rectification, letterbox trim
│   ├── document.py            plain-page generation          cleanpage.py   deblur / threshold helpers
│   ├── toprow.py              small Bangla top-row clean-up   relief.py      3D relief reading
│   ├── restore.py layout.py readable.py quality.py strokes.py geometry.py detect.py imgio.py
│   ├── api.py server.py cli.py static/   HTTP route, stand-alone service, CLI, UI tab assets
│   ├── scripts/               folder scans with contact sheets
│   └── tests/
├── ocr/                     YOLO + OCR route (ONNX models, fonts, training / evaluation scripts, README)
├── app/                     web app: server.py + static UI (plate.html, plate.js, plate.css)
├── tests/                   app and OCR-route tests
├── run.py                   starts the web app
├── Dockerfile.opencv        OpenCV-only service image
├── Dockerfile.ocr           YOLO + OCR service image
├── Dockerfile               whole web app in one image
└── requirements.txt         dependencies for the full app
```

---

## The OpenCV scan (`opencv/`)

```
photo ──► trim ──► find 4 corners ──► straighten ──► restore ──► plain page
```

| Stage | Module | Method |
|---|---|---|
| Trim | `plate_find.trim_letterbox` | cut pure-black padding bars |
| Find the plate | `scan.find_corners`, `quadfit.py`, `plate_find.py`, `detect.py` | k-means colour clusters and seeded region growing give candidate boxes; each is scored by whether its four sides lie on a real colour step along their whole length; the four border lines are then fitted under plate-shape constraints; a text-containment check keeps the quad that holds the characters. Wide shots: Canny + contour search |
| Straighten | `plate_find.rectify` | homography to the true aspect ratio (estimated from the quad, snapped to common plate shapes) |
| Restore | `restore.py` | low-light gamma / CLAHE, glare inpainting, bilateral de-blocking, non-local-means denoise, bicubic upscale, contrast stretch, unsharp mask |
| Plain page | `document.py`, `cleanpage.py` | Fisher colour projection with a coherence self-check, illumination flattening, blur estimation + Wiener deconvolution, black-hat stroke mask, hysteresis threshold, connected-component and morphological clean-up, frame / bolt / rule removal, row detection with per-row deskew and de-slant, stroke evening, smooth contour rebuild |
| Top row | `toprow.py` | per-row deblur + threshold (Sauvola and Otsu candidates, the one with fewest broken strokes wins), thinning to print weight, centre-line-safe lump removal, gap bridging guided by the grey image |
| Embossed plates | `relief.py` | see below |

### Embossed ("3D") plates

Some plates are stamped in the plate colour: there is no ink to threshold, the characters only exist as light and
shadow. `relief.py` reads that relief: it flattens the lighting, measures the edge strength at stroke scale, masks the
mounting bolts and long rim lines, and stretches the contrast so each raised character appears as a crisp dark
**outline** on white. Switch it on with `relief="force"`, the CLI flag `--relief force`, or the *Embossed (3D) plate*
checkbox in the UI. `relief="auto"` only switches when the normal ink map is almost empty.

More detail, module reference and the Python API: [`opencv/README.md`](opencv/README.md). **How every stage works: [`opencv/ARCHITECTURE.md`](opencv/ARCHITECTURE.md).**

---

## The OCR route (`ocr/`)

| Component | File | Notes |
|---|---|---|
| Plate detector | `detector.py` | YOLO-v9 (ONNX, via `open-image-models`), tiling for small plates, classical fallback |
| Bangla number row | `bangla_reader.py`, `bangla_row.onnx` | CRNN + CTC over the six digits |
| Top row | `bangla_top.py`, `bangla_top.onnx` | city / series-letter classifier |
| Latin plate OCR | `ocr.py` | `fast-plate-ocr` with multi-variant voting |
| Reliability | `verdict.py`, `verdict_bn.py` | calibrated score → `readable` / `uncertain` / `unreadable` |
| Deblur / video | `deblur.py`, `video.py` | OCR-guided deblur; tracking, frame voting and fusion |
| Typeset page | `typeset.py`, `textrender.py`, `fonts/` | clean typed page for reliable readings (HarfBuzz) |
| Training / evaluation | `scripts/` | GPU training, cross-validation grouped by plate number, calibration, labelling page |

Details: [`ocr/README.md`](ocr/README.md). **Pipeline, models, decision rules, training and evaluation: [`ocr/ARCHITECTURE.md`](ocr/ARCHITECTURE.md).**

---

## Web app and HTTP API

`python run.py` serves the UI at `http://127.0.0.1:8000` (`--port`, `--host`, `--no-browser` are available).
Interactive API docs are at `/docs`.

| Method and path | Purpose |
|---|---|
| `POST /api/opencv/scan` | OpenCV scan. Form fields: `file`, `relief` = `auto` \| `force` \| `off`. Returns `corners`, `method`, `original` (corners drawn), `rectified`, `restored`, `document` (data-URI images), `width`, `height`, `elapsed_s` |
| `POST /api/plate/read` | **The reading pipeline** (used by the Image tab): YOLO / crop -> OpenCV scan -> ink and embossed readers. Form fields: `file`, `mode` = `auto` \| `ink` \| `relief`. Returns `verdict` (readable / uncertain / unreadable), `top` (city word), `letter`, `digits`, `number` (NN-NNNN), `mode`, `corner_method`, `elapsed_s` and result pictures (`result`, `page`, `straight`, `original`). No confidence numbers are exposed |
| `POST /api/plate/scan` | The older YOLO + OCR route on an image. Returns `plates[]` with `detection`, `quality`, `reading`, `verdict`, `document_kind`, `timings_ms` and result image URLs |
| `POST /api/plate/video` | start a video job (`fps`, `max_seconds`); returns `job_id` |
| `GET /api/plate/job/{id}` | job status / progress / result |
| `GET /api/plate/asset/{scan_id}/{name}` | result images |
| `GET /api/plate/capabilities` | whether the detector and OCR models loaded |
| `GET /api/health` | liveness |

```bash
curl -F file=@plate.jpg -F relief=auto http://localhost:8000/api/opencv/scan
curl -F file=@car.jpg http://localhost:8000/api/plate/scan
```

---

## Docker and production

Everything runs on **CPU**; no GPU is needed or used by the OpenCV scan (`opencv-python-headless`).

```bash
# OpenCV-only service (no models)
docker build -f Dockerfile.opencv -t plate-opencv .
docker run --rm -p 8000:8000 --cpus=2 -e WORKERS=2 plate-opencv

# YOLO + OCR service (models are downloaded while the image is built)
docker build -f Dockerfile.ocr -t plate-ocr .
docker run --rm -p 8001:8001 --cpus=4 -e WORKERS=2 -e ORT_INTRA_THREADS=2 plate-ocr

# whole web app
docker build -t plate-app . && docker run --rm -p 8000:8000 plate-app
```

Without Docker: `uvicorn opencv.server:app --workers 4`, `uvicorn ocr.server:app --workers 2`.

**Measured (this repo's images, Docker Desktop, CPU only).** `plate-opencv` is 587 MB and `plate-ocr` 822 MB (models baked in; the OCR image also starts with no network). One container uses about 350 MB of memory and stays near 450 MB after 120 reads. A read takes about 2-3 s per plate crop and 3-4 s for a 1920x1080 frame (YOLO with tiling). 60 reads, 6 at a time, gave answers identical to sequential ones. Corrupt or non-image uploads return 400, blank / noise / 1-pixel / 24-megapixel images return an "unreadable" verdict without errors.

**CPU sizing.** One scan is mostly single-threaded, so throughput comes from worker *processes*: set `WORKERS` to the
CPUs you give the container and keep each worker to one thread (the images do this by default). For the OCR image the
thread budget is about `WORKERS × ORT_INTRA_THREADS`. The scan is deterministic, so results can be cached by image hash.
A typical production pipeline is *YOLO service → crop → OpenCV service (display / storage) → OCR on the straightened
colour crop*. OCR reads the restored colour plate better than the black-and-white page; use the plain page for display.

---

## Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `WORKERS` | `2` | uvicorn worker processes (Docker images) |
| `PORT` | `8000` / `8001` | listen port (Docker images) |
| `OPENCV_THREADS` | `1` | threads OpenCV may use inside one scan |
| `OPENCV_MAX_CONCURRENT` | `2` | scans running at once per worker (OpenCV service) |
| `OPENCV_MAX_UPLOAD_MB` | `25` | upload limit, OpenCV service |
| `OPENCV_MAX_IMAGE_DIM` | `4000` | longest image side kept, OpenCV service |
| `ORT_INTRA_THREADS` / `ORT_INTER_THREADS` | `0` (ONNX default) | ONNX Runtime threads per model session |
| `OCR_MAX_UPLOAD_MB` | `25` | upload limit, OCR service |

---

## Testing

```bash
python -m pytest tests opencv/tests        # app, OCR route and standalone OpenCV tests
```

Some OCR-route tests are skipped when their sample photos are not present locally (see [Data and privacy](#data-and-privacy)).
The OpenCV tests use synthetic plates and need no data. The scan is deterministic, which the tests assert.

---

## Evaluation and limitations

- The OpenCV scan was developed and checked on three local tier sets (easy / medium / extreme CCTV plates), a
  200-plate set and a small set of embossed no-ink plates. Results were **judged by eye** per image, using two
  measures: *corners correct* and *usable output*, and compared against saved contact sheets; no automatic accuracy
  number is claimed for the current version.
- Output quality is limited by the photo. If blur, glare or compression destroyed the strokes, no classical step can
  restore them: very small or very blurry plates stay rough, mostly in the small top row.
- Embossed plates come out as outlines, not solid text; shallow or paint-worn relief gives partial results.
- Settings were tuned on the sets above; other cameras and plate styles may need adjustment. The relief mode was tuned
  on four plates.
- **OCR reading (normal plates).** On the 26 labelled plates of the extreme tier, tested by readers that never saw them: 87% of digits
  (136/156), 17/26 whole numbers, 91% of first-row words (20/22); every plate marked `readable` was fully correct (12/12). The class
  letter is below 50% right and is not printed. Easy and medium tiers were not tested cleanly (their labelled plates were used for
  tuning). See `eval/ocr_extreme/`.
- **OCR reading (embossed plates, no ink).** Only 10 labelled plates exist (9 photos, all Dhaka): leave-out cross-validation gave
  57/60 digits and 10/10 first-row words, but with that little data the number is optimistic. See `eval/ocr_relief3d/`.
- Plates are expected upright; a plate rotated by 90 degrees is not recognised. One plate per image is read.
- Treat the reading as a suggestion unless the verdict says `readable`.

---

## Data and privacy

The photos used for development show real vehicles and their plate numbers, and some come from third-party sources.
They, their labels and the generated result sheets are **not part of this repository**. The scripts take any folder of
images, for example `python opencv/scripts/scan_plain.py <folder with easy/medium/extreme sub-folders> --out results/`.
If you publish results, make sure you have the right to share the images.

---

## Third-party components

| Component | Use | License |
|---|---|---|
| [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/) | the whole OpenCV scan | Apache-2.0 / BSD |
| [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/) | web app and services | MIT / BSD |
| [ONNX Runtime](https://onnxruntime.ai/) | model inference | MIT |
| [open-image-models](https://github.com/ankandrew/open-image-models) | YOLO-v9 plate detector | MIT |
| [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) | Latin plate OCR | MIT |
| [HarfBuzz (uharfbuzz)](https://github.com/harfbuzz/uharfbuzz), [fontTools](https://github.com/fonttools/fonttools) | Bengali text shaping | MIT |
| Noto Sans Bengali and the other fonts in `ocr/fonts/` | typeset pages | SIL OFL 1.1 (license texts alongside the fonts) |

---

## License

[MIT](LICENSE)
