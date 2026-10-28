# opencv - license-plate scan with OpenCV + NumPy only

Takes a photo of a (Bangladeshi) license plate and returns a **plain page**: the plate straightened, cleaned and
re-drawn as black characters on pure white paper.

**No OCR, no neural network, no training, no model files.** Everything here is classical image processing
(`cv2` + `numpy`). The folder is self-contained: copy `opencv/` into any project and import it.

```
photo ──► 4 corners ──► straighten ──► clean-up ──► plain page
```

## Install

```bash
pip install -r opencv/requirements.txt       # opencv-python-headless + numpy (+ Pillow / FastAPI for the web API only)
```

## Use

**Python**

```python
import cv2
from opencv import scan

result = scan(cv2.imread("plate.jpg"))
result.corners      # (4, 2) float32: TL, TR, BR, BL of the plate in the input image
result.method       # how the corners were found
result.rectified    # straightened plate (colour)
result.restored     # denoised / deblocked / upscaled / contrast-stretched plate
result.document     # the plain page: uint8, white paper, black characters, 960 px wide
cv2.imwrite("plain.png", result.document)
```

**Command line**

```bash
python -m opencv.cli plate.jpg --out out/                     # one image
python -m opencv.cli "folder of plates" --out out/            # a folder
python -m opencv.cli plate.jpg --out out/ --relief force      # embossed plate with no ink (see below)
```

**Whole folders with a contact sheet** (pairs of straightened plate | final page, 3 per row)

```bash
python opencv/scripts/scan_plain.py <folder> --out results/        # a folder of images, or sub-folders such as easy/medium/extreme
python opencv/scripts/scan_relief.py <folder> --out results_3d/    # normal vs 3D relief side by side
```

**Web API** (used by the OpenCV tab of the main app): `POST /api/opencv/scan` with a `file` and an optional
`relief` field (`auto` | `force` | `off`) - see `opencv/api.py`.

**Tests**

```bash
python -m pytest opencv/tests
```

## How it works

| Stage | Module | What it does |
|---|---|---|
| Trim | `plate_find.trim_letterbox` | cuts black padding bars |
| Find the plate | `scan.find_corners`, `quadfit.py`, `plate_find.py`, `detect.py` | k-means colour clusters + seeded region growing give candidate boxes; each is scored by whether its four sides sit on a real colour step along their whole length, then the four border lines are fitted; the quad that keeps the most characters inside wins. Wide shots use Canny + contour search (`detect.py`) |
| Straighten | `plate_find.rectify` | homography to the plate's true aspect ratio |
| Restore | `restore.py` | low-light lift, glare removal (inpainting), de-blocking, non-local-means denoise, bicubic upscale, contrast stretch, unsharp |
| Plain page | `document.py`, `cleanpage.py` | Fisher colour projection (best ink/plate separation, even for teal-on-teal), illumination flattening, Wiener deblur from a blur estimate taken off stroke edges, black-hat stroke mask, hysteresis threshold, frame / bolt / rule removal, per-row deskew and de-slant, stroke evening, smooth contour rebuild |
| Top row (city / series letter) | `toprow.py` | the small Bangla row gets its own deblur + threshold, is thinned back to print weight, has its filled counters re-opened and gaps bridged where the grey image shows ink |
| Embossed plates | `relief.py` | see below |

Helpers: `geometry.py` (point ordering, deskew, de-shear), `layout.py` (row / glyph layout, frame removal),
`readable.py` (best grey channel, polarity), `quality.py` (blur / noise / blockiness measures), `strokes.py`
(thin-stroke mask), `imgio.py` (EXIF-aware image loading and encoding).

### Embossed ("3D") plates with no ink

Some plates are stamped in the plate colour: there is no ink to threshold. The characters only show as light and
shadow, so `relief.py` reads the relief instead:

1. high-pass the luminance and normalise the local contrast (colour and paint tone drop out);
2. denoise, measure the edge strength at stroke scale (the lit and shadowed edges of the embossing);
3. mask the mounting bolts and remove long straight rim lines;
4. stretch the contrast so only real glyph contours stay dark, crop to the text.

The page then shows each raised character as a clear dark **outline** on white. Use `relief="force"` (the
"Embossed (3D) plate" checkbox in the web UI, `--relief force` on the CLI). `relief="auto"` switches to it only when
the normal ink map is almost empty. `relief.py` also contains the signed-derivative stroke finder
(`relief_ink`: a ridge behind and a shadow ahead along the estimated light direction) used for experiments.

## Honest limits

- Output quality depends on what is in the photo. If blur, glare or compression destroyed the strokes, no
  classical step can bring them back; very small or very blurry plates stay rough, mostly in the small top row.
- Embossed plates come out as outlines, not solid text, and shallow or paint-worn relief gives partial results.
- Settings were tuned on local tier sets (easy / medium / extreme CCTV plates), a 200-plate set and four embossed plates;
  those photos are not in the repository (real vehicles).
- Results are checked by eye, per image, with two measures: *corners correct* and *usable output*; no automatic accuracy
  number is claimed. Scans are deterministic.

## Run it in Docker (CPU)

```bash
docker build -f Dockerfile.opencv -t plate-opencv .
docker run --rm -p 8000:8000 --cpus=2 -e WORKERS=2 plate-opencv
curl -F file=@plate.jpg -F relief=auto http://localhost:8000/api/opencv/scan      # GET /health for a health check
```

The image copies only `opencv/` (no models, no GPU). One scan is mostly single-threaded, so throughput comes from
`WORKERS` (processes): set it to the CPUs you give the container and leave `OPENCV_THREADS=1`.
`OPENCV_MAX_CONCURRENT` (default 2) limits scans running at once per worker; `OPENCV_MAX_UPLOAD_MB` caps uploads.
Without Docker: `uvicorn opencv.server:app --port 8000 --workers 4`.
