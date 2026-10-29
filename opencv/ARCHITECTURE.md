# OpenCV scan - architecture

`opencv/` turns a photo of a Bangladeshi license plate into a **plain page**: the plate found, straightened, cleaned and
re-drawn as black characters on white. It uses only **OpenCV and NumPy**: no neural network, no OCR, no training, no model files.
It runs on the CPU, gives the same output for the same input, and can be copied into another project as one folder.

```
 photo / CCTV frame / plate crop
          │
          ▼
 ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐   ┌─────────────────────────────┐
 │ 0. load + trim     │──►│ 1. find 4 corners  │──►│ 2. straighten      │──►│ 3. restore (only what the   │
 │ EXIF, size cap,    │   │ quadfit.py         │   │ homography to the  │   │    crop needs)              │
 │ black bars removed │   │ plate_find.py      │   │ plate's true shape │   │ restore.py                  │
 └────────────────────┘   │ detect.py (wide)   │   └────────────────────┘   └──────────────┬──────────────┘
                          └────────────────────┘                                           │
                                                                                           ▼
                          ┌────────────────────────────────────────────────────────────────────────────┐
                          │ 4. plain page   document.py + cleanpage.py + toprow.py + layout.py          │
                          │    ink plate  ──► colour projection, deblur, threshold, clean, rows,        │
                          │                   dedicated top-row processing, smooth stroke rebuild       │
                          │    no-ink plate ► relief.py: read the light and shadow of the raised letters │
                          └────────────────────────────────────────────────────────────────────────────┘
                                                                                           │
                                                          CvScan(corners, method, rectified, restored, document)
```

## 1. Design principles

| Principle | What it means in the code |
|---|---|
| **Classical only** | Only `cv2` and `numpy` (plus Pillow for EXIF-aware loading and FastAPI for the optional API). Nothing is learned, so there is nothing to retrain when a camera changes: you change a constant. |
| **Deterministic** | The same image gives the same corners and page. The one random step, the k-means seed, is fixed with `cv2.setRNGSeed(1234)` before every clustering. |
| **Thread-safe** | Per-image fitting state lives in a `threading.local` object (`quadfit._FIT`), so several scans can run in threads of one process. A test runs 4 threads against sequential results. |
| **Measure first** | Each stage returns something inspectable (`rectified`, `restored`, `document`, `method`), and every stage was checked image by image on the tier sets before the next was tuned. |
| **Do no harm** | Restoration steps fire only when the measured quality calls for them; the thinning of the first row never deletes a stroke (its centre line is always put back); the relief mode is used only for plates that have no ink. |
| **CPU, one thread per scan** | A scan is mostly single-threaded; throughput comes from worker processes. GPU was evaluated and rejected (see section 10). |

## 2. Module map

| File | Role |
|---|---|
| `scan.py` | Entry point. `scan(img, relief="auto", make_page=True) -> CvScan`; `find_corners(img)`: routes between the crop path and the wide-shot path |
| `quadfit.py` | The corner finder for plate crops: candidate boxes, edge-verified scoring, border-line fitting |
| `plate_find.py` | `trim_letterbox`, the older colour-cluster hull (`crop_mode_quad`), `subpixel_corners`, `true_aspect`, `rectify` |
| `detect.py` | Wide-shot localiser (`detect_document(img, 600, "plate")`): masks, contours, plate-likeness scoring. Also contains the document paths of the original scanner |
| `restore.py`, `quality.py` | Quality measures and the restoration steps |
| `document.py` | `make_document`: the plain-page pipeline for ink plates |
| `cleanpage.py` | Blur estimate, Wiener deblur, Sauvola threshold, Fisher colour projection, smooth stroke rebuild |
| `toprow.py` | The small Bangla first row: re-extraction, thinning, gap bridging |
| `layout.py`, `readable.py`, `strokes.py`, `geometry.py` | Frame / bolt removal and glyph layout; best grey channel and polarity; thin-stroke mask; ordering, deskew, de-shear |
| `relief.py` | 3D (embossed) plates: `relief_lines`, `relief_page`, `relief_view`, `relief_ink` |
| `imgio.py` | `load_image_from_bytes` (EXIF orientation, alpha on white, size cap) and `encode_image` |
| `api.py`, `server.py`, `cli.py` | `POST /api/opencv/scan`, a stand-alone FastAPI service, a command line tool |
| `scripts/`, `tests/` | Folder scans with contact sheets; self-contained tests |

## 3. Stage 0 - input

* `imgio.load_image_from_bytes` fixes EXIF rotation, composites transparency on white, converts to BGR and caps the longest side at
  4000 px (`OPENCV_MAX_IMAGE_DIM`).
* `plate_find.trim_letterbox` removes pure-black padding bars (pixels ≤ 6 on more than 3 % of a row or column), so resized crops
  with black borders are not mistaken for dark plates. The offset is added back to the corners at the end.

## 4. Stage 1 - find the four corners

`scan.find_corners` chooses a path by the picture's size:

```
 longest side ≤ 720 px and width/height in 0.55 … 3.5 ?
        │ yes: a tight crop, the plate body fills the frame            │ no: a wide shot (a car photo)
        ▼                                                              ▼
  quadfit.best_quad  +  crop_mode_quad  → "edge-verified quad"   detect_document(…, "plate") → "edge/contour search"
        │                                  or "colour-cluster hull"              │ nothing found
        │ nothing usable                                                         ▼
        ▼                                                                   "full frame"
     "full frame"
```

### 4.1 Tight crops: `quadfit.py`

The earlier method (`plate_find.crop_mode_quad`) takes the plate colour blob's convex hull. It fails when that blob merges with a
car body or with padding of a similar colour: the quad then spills onto the background. `quadfit` checks every candidate against
the picture instead of trusting a blob. All work runs at 320 px on the longest side, in Lab colour, after a 3×3 blur.

**a. Candidate quads**

| Source | How |
|---|---|
| Colour clusters | k-means with k = 2, 3, 4, 5 on Lab pixels; every blob covering 15 %–98 % of the image is closed with a 5×5 kernel, turned into a convex hull, then into a minimum-area rectangle and (when it exists) a 4-point polygon (`approxPolyDP`, ε = 2 %, 4 %, 7 % of the perimeter) |
| Seeded region | The plate body colour is taken from the **centre** of the crop: the majority class (Otsu on L) of the central patch, decided by which class owns the very centre. Pixels within Lab distance 16 / 24 / 34 of that colour, connected to the centre, closed and opened, give a region whose hull becomes a quad. Unlike free clustering it cannot wander into a background that merely shares a cluster |
| The whole frame | Always a candidate, for plates that fill the picture |

If a seeded quad covers ≥ 12 % of the image and ≥ 50 % of its pixels match the body colour, the seeded candidates alone are used
and the box is only fitted within ±12 % of its size ("tight" mode). Otherwise all candidates are used with a ±55 % allowance.

**b. Scoring a quad: `edge_support`.** Four terms, each computed along the **whole** side, not at a few points:

1. *Colour step*: the Lab distance between a point just **outside** the side and one just **inside** (offset 1.2 % of the image size);
   a real plate border is a long, consistent step, background and letters are not. The score uses the mean step (capped at 40) and the
   share of the side where the step exceeds 10.
2. *Leak*: how much of the band 2, 3.5 and 5 offsets outside the side still has the plate-body colour (distance < 24). A side that
   cuts through the plate has body colour outside it and is penalised (weight 0.9).
3. *Body share*: the share of the quad's pixels within distance 26 of the body colour (weight 0.5): background spill lowers it.
4. *Frame sides*: a side lying on the picture's border (more than 60 % of its outside points off-image) is judged by whether the
   inside still looks like plate body, so plates cut off by the frame keep their last digits. Corners may lie up to 25 % of the image
   size beyond the frame.

**c. Refinement: `fit_lines`.** The best six candidates are refined. Each of the four border lines (angle + offset) is moved in
steps from 12 % down to 0.8 % of the quad size, keeping the shape plate-like: opposite sides parallel within 0.30 rad, adjacent sides
perpendicular within 0.55 rad, convex, area at least half the start. The best refined quad wins.

**d. Cross-check.** `scan._text_inside` measures how many of the thin strokes (black-hat response, i.e. the characters) lie inside
the old hull and inside the new quad. If the old hull holds more than 4 % more of them, it is kept: a quad that cuts through the
digits is worse than a loose one. Returned methods: `"edge-verified quad"`, `"colour-cluster hull"` (corners then snapped with
`cornerSubPix`), or `"full frame"`.

### 4.2 Wide shots: `detect.py`

`_find_plate_quad` builds six foreground masks (Otsu both polarities, two adaptive thresholds, locally-brighter-than-background,
Canny edges), closes each with three rectangular kernels, finds contours (`RETR_LIST`, so a plate inside a bumper blob is seen),
keeps compact rectangles (aspect 1–6.5, at least 72 % rectangular, area ≥ 0.3 % of the frame), scores each with
`_plate_likeness × _plate_prior`, penalises quads that leave text outside them, and finally prefers the **innermost** strong candidate
(a plate sits inside the car front that also scores well). If nothing passes, a text-context finder is tried. This path is used on
about 4 of the 155 tier images; most plates arrive as crops.

## 5. Stage 2 - straighten (`plate_find.rectify`)

* The plate's true width/height ratio is estimated from the quad's homography, assuming a camera of roughly 50° field of view
  (focal length 1.1 × the longest image side), falling back to the measured edge ratio if the estimate is implausible, and snapped to
  a common plate shape (1.3, 2.0, 2.5, 3.2, 4.0, 4.7, 5.2) when within 10 %.
* `cv2.getPerspectiveTransform` + `warpPerspective` (cubic, border replicate) produce the front view with a 5 % margin (8 % vertically).

## 6. Stage 3 - restore (`restore.py`)

`quality.assess_quality` measures resolution, sharpness, contrast, mean level, share of very dark / very bright pixels, noise and
JPEG blockiness on a height-normalised grey copy, and lists issues (`low_resolution` < 70 px wide, `blurry` sharpness < 0.40,
`low_contrast`, `dark`, `glare`, `noisy`, `compressed`). `restore_plate` then applies **only what is needed**, in a safe order
(brighten first, so noise and blocks are measured at their true level):

| Step | Fires when | What it does |
|---|---|---|
| Low-light lift | `dark` | gamma towards mid-grey on luminance, CLAHE (2×4 tiles), blend with a single-scale retinex |
| Glare | `glare` | small saturated blobs (≤ 4 % of the area) inpainted (Telea) |
| De-block | `compressed` or blockiness > 1.18 | bilateral filter blended 70/30 with the original |
| Denoise | noise > 4.5 | non-local means, strength from the measured noise |
| Upscale | height < 96 px | bicubic, at most 4× |
| Contrast | always | percentile stretch (1 %–99 %) |
| Sharpen | `blurry` / `low_resolution` | unsharp mask |

On the 155 tier images: contrast 155, upscale 117, low-light 15, de-block 10, denoise 7, glare 1. It takes about 0.01 s.

## 7. Stage 4 - the plain page for ink plates (`document.py`)

`make_document(restored)` works on the restored crop and returns a uint8 page, 960 px wide, white background, black characters.

**Sizing.** Crops taller than 200 px are first downscaled to 128 px: the page step is tuned for plates about 100-130 px high and
upscaling a bigger one 3× only made it 7× slower (a 500 px crop took 14.6 s, now 2.4 s, with identical output on the tuned sets).
The image is deskewed (text-line projection profile, Hough as a fallback), scaled 4× (< 500 px), 3× (< 700 px) or 1×, and smoothed with
an edge-preserving bilateral filter.

**Finding the ink.** For several grey projections of the plate the same chain runs, and the one with the most *coherent* ink wins:

1. *Candidates*: a Fisher colour projection (`cleanpage.fisher_channel`: ink and body are split by Otsu, then the direction
   Σ⁻¹(μ_body − μ_ink) separates them best across B, G, R, which recovers teal-on-teal plates) and the best plain channel
   (`readable.best_channel`: grey, B, G, R or PCA, only those where the characters are the darker minority).
2. *Polarity*: dark text is the default; it flips only for a clearly dark plate body with brighter characters.
3. *Flatten*: divide by a large morphological closing blurred (kernel 30 % of the height) to remove shadows and vignetting.
4. *Deblur*: the blur is estimated from stroke edges (a step edge of contrast C blurred by σ has peak gradient C/(σ√2π), robust
   percentiles); if σ ≥ 1.2 a frequency-domain **Wiener deconvolution** (NumPy FFT, edge-reflected padding) with strength 0.8σ is applied.
5. *Soft ink*: black and white points come from the plate's own tones (Otsu split, medians), plus a local-contrast term (a stroke must
   be clearly darker than the paper around it).
6. *Near-stroke mask*: the black-hat response (thin dark structures, not big shadows) grown by half a stroke keeps bold strokes solid.
7. *Hysteresis*: pixels above 0.42 that connect to a pixel above 0.60.
8. *Coherence*: the share of ink in character-sized pieces. The Fisher candidate wins unless the grey one is better by more than 0.02.

**Is there any ink?** If the winning ink map is nearly empty (coverage < 8 %, or coherence < 30 % and the relief reading finds more),
the plate has no ink and the relief page of section 8 is returned instead (`relief="auto"`). `relief="force"` skips the ink pipeline
entirely; `relief="off"` never uses it.

**Cleaning.**

| Step | Purpose |
|---|---|
| Inset the plate edge (8.5 % vertical, 6.5 % horizontal) | the rectified frame includes the rim |
| `layout.clean_ink` | crop to the inner frame, remove long frame lines, round bolts in the top corners, slivers and speckle relative to the plate's own character height |
| Small-piece filter, `_drop_bars` | remove pieces far below a character's ink, and stand-alone hair-thin bars (plate edge, shadow lines) |
| Row bands | the ink-row profile is split at its deepest valley into at most two rows (first row and number) |
| Number row | stroke thickness evened (thin broken strokes thickened to print weight, with a guard against merging), outline smoothing |
| First row | `toprow.thin_row`, section 7.1 |
| `_align_rows` | each row levelled and de-slanted on its own, rules stripped, rows centred on a common width |
| `cleanpage.rebuild_strokes` | every outline redrawn as a smooth closed curve at 2× resolution (print-like edges, not pixel staircases) |
| Crop, scale, crisp | crop with margins, resize to 960 px wide, smoothstep threshold, values above 245 set to pure white |

### 7.1 The first row (`toprow.py`)

The first row is small thin Bangla script with conjuncts (the "ট্রো" of "মেট্রো"). The generic stroke clean-up, tuned for bold digits,
closes counters and fattens strokes until the words become blobs, so this row is processed alone:

1. *Re-extract* from the grey image with its own deblur and local contrast (CLAHE), by two methods: a stroke-sized Sauvola window and
   a denoised per-row Otsu. The candidate with the fewest broken pieces that is not a blob (ink coverage ≤ 30 %) wins; the result is
   pulled back to the original stroke width so the page weight does not change.
2. *Thin to print weight*: a bold or clogged row is eroded until its ink coverage is a print level (≈ 27 % of its box) and its stroke is
   about 9.5 % of the row height, but **the Zhang-Suen centre lines of the original are always put back**, so no stroke can disappear.
3. *Re-open counters*: pixels inside the mask that are clearly paper-like in the grey image are removed.
4. *Bridge gaps*: stroke ends and stray dots are joined to the nearest other piece when the grey image really is dark along the line
   between them (reach 14 % of the row height). Nothing is redrawn: this only fills what the picture shows.
5. *Remove what is not text*: long rules at the top and bottom, tall bars at the ends, thick compact lumps (rivets, shadow corners) that
   sit at the row's border; lumps in the middle are letters and stay.

A "redraw the row as uniform centre-line strokes" mode (`MONO`) was built and **turned off**: it produced cleaner-looking but less
faithful letters.

## 8. Embossed plates with no ink (`relief.py`)

Some plates are stamped in the plate colour. There is nothing to threshold; the characters exist only as light and shadow:
the face turned toward the light has a bright edge, the face turned away a dark one (brightness ≈ the derivative of the height along
the light direction).

`relief_lines(img, work_h=360)` produces the page's line map:

```
 luminance (Lab L) → denoise (non-local means) → high-pass (remove lighting, paint tone) and normalise the local contrast
   → edge strength of the blurred result at stroke scale → mask the bolt corners and a 9 % border
   → remove long straight runs (rim, frame) → keep only the strongest contours (clip, gamma)
```

`relief_page` crops that map to the text, scales it to 960 px and returns a page where each raised character is a clear dark
**outline** on white. Solid fills were tried (thresholding the relief, and a ridge-and-shadow pairing along the estimated light
direction, `relief_ink`) and gave blobs; outlines are readable.

`relief_view(img, work_h=160)` returns the signed shading itself (flat plate = 128, lit ridges bright, shadows dark) and is the input of
the embossed readers in the `ocr/` package: it is the same representation at training and reading time.

## 9. Interfaces

* **Python**: `from opencv import scan, find_corners, CvScan`. `CvScan` has `corners` (4×2, TL TR BR BL in the *input* image),
  `method`, `rectified`, `restored`, `document` (`None` when `make_page=False`).
* **CLI**: `python -m opencv.cli plate.jpg --out out/ [--relief auto|force|off]`.
* **HTTP**: `POST /api/opencv/scan` (form: `file`, `relief`). Runs the scan off the event loop with at most `OPENCV_MAX_CONCURRENT`
  (default 2) scans per process. Environment: `OPENCV_THREADS` (default 1), `OPENCV_MAX_UPLOAD_MB` (25), `OPENCV_MAX_IMAGE_DIM` (4000).
  `opencv/server.py` serves it stand-alone (`uvicorn opencv.server:app --workers N`); `Dockerfile.opencv` packages it.

## 10. Performance and decisions

Measured on one CPU thread (this PC, 12 logical CPUs):

| Step | Time per plate |
|---|---|
| Find corners | ~1.0 s (was 4.1 s) |
| Restore | ~0.01 s |
| Plain page | ~1.5 s |
| Whole scan, small CCTV crop | ~2.5 s |
| Whole scan, 500 px crop (page step downscaled) | ~2.4 s (was 14.6 s) |
| `relief="force"` | ~1.4 s |

* The corner search was made 4× faster **with identical output** (batched colour sampling, a per-image body-colour mask, caches for repeated
  quads, closed-form line intersection), checked on 158 images (maximum corner difference 0.0 px).
* The plain-page step was **not** sped up by approximate shortcuts: running the flatten step at one third resolution was only 1.3×
  faster and changed every page, with up to 10 % different heights.
* **GPU was evaluated and rejected**: the time is in thousands of small steps on tiny crops (100-700 px), the pip OpenCV has no CUDA,
  and CPU↔GPU copies would cost as much as the work. Scale by adding worker processes.
* Docker: `plate-opencv` is about 590 MB and uses about 350 MB of memory per container.

## 11. Limitations

* Output quality is bounded by the photo: strokes destroyed by blur, glare or compression cannot be restored by classical steps; very
  small or blurry plates stay rough, mostly in the first row.
* Embossed plates come out as outlines, not solid text, and shallow or paint-worn relief gives partial results. The relief settings
  were tuned on four plates.
* One plate per crop; a plate rotated by 90° is not handled.
* Settings were tuned on easy / medium / extreme CCTV tiers, a 200-plate set and a few embossed plates (not in the repository). The
  results were judged per image by eye, with two measures: *corners correct* and *usable output*; no automatic accuracy number is claimed.

## 12. Tests

`python -m pytest opencv/tests`: a scan of a synthetic plate returns four corners and a white page with dark characters, is
deterministic, runs in relief mode, gives the same corners from a 4-thread pool as sequentially, and handles large crops.
