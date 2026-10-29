# OCR pipeline - architecture

`ocr/` reads a Bangladeshi license plate: the **first-row words** (city, e.g. "ঢাকা মেট্রো", plus the class letter), the **six digits**
of the number (`NN-NNNN`), and a **verdict** saying how far to trust the reading. It finds the plate with YOLO, uses the
[`opencv/`](../opencv/ARCHITECTURE.md) package to straighten and clean it, and reads it with small neural networks run through ONNX
Runtime on the CPU.

```
 photo / CCTV frame / plate crop
        │
        ▼
 ┌──────────────────┐   ┌───────────────────────────┐   ┌───────────────────────────────────────────────┐
 │ 1. find the plate│──►│ 2. OpenCV scan            │──►│ 3. read the plate (4 small ONNX models)       │
 │ YOLO-v9 tiny     │   │ corners → straighten →    │   │  ink readers      digits  + first row         │
 │ (large frames    │   │ restore                   │   │  embossed readers digits  + first row         │
 │  only)           │   │ opencv/                   │   └──────────────────────┬────────────────────────┘
 └──────────────────┘   └───────────────────────────┘                          ▼
                                                        ┌───────────────────────────────────────────────┐
                                                        │ 4. decide: best digit per position, first-row │
                                                        │    word, verdict    5. result picture + JSON  │
                                                        └───────────────────────────────────────────────┘
```

The whole thing is `ocr/pipeline.py::read_plate(img)`; the web app's Image tab and `POST /api/plate/read` call it.

## 1. Design principles

| Principle | What it means |
|---|---|
| **Small models, CPU only** | Four readers of 0.86 M and 1.8 M parameters plus a 1.9 M-parameter detector (7.2 M in total) run on the CPU in about 0.1-0.2 s per plate. No GPU is needed to *run* it; the GPU is used only to *train*. |
| **Two specialists, not one generalist** | Plates with dark ink and plates stamped in the plate colour (no ink contrast) look different. Each has its own pair of readers, and the more convincing one wins. |
| **Models are replaceable files** | The readers are ONNX files in `ocr/`; the PyTorch weights to continue training are in `ocr/weights/`; the synthetic training plates are in `synthetic data/`. |
| **A verdict instead of confidence numbers** | The result shows words and digits, and one flag (readable / uncertain / unreadable). The flag was checked against held-out plates. |
| **Honest evaluation** | Plates used to tune a model are never used to score it: held-out folds, and no second photo of a test plate in training. |

## 2. Module map

| File | Role |
|---|---|
| `pipeline.py` | **`read_plate`**, `render_report`, the decision rules, `use_models` (swap ONNX files for testing) |
| `detector.py` | `LearnedDetector`: the YOLO plate detector with tiling (the `PlateDetector` classical fallback is used by the older route) |
| `bangla_reader.py` | `BanglaRowReader`: the six-digit reader (CRNN, CTC) with multi-view voting |
| `bangla_top.py` | `BanglaTopReader`: the first-row reader (CNN with a city head and a letter head) |
| `ortutil.py` | ONNX Runtime session options (thread limits) |
| `synth_row.py` | Synthetic plate generator: `render_plate`, `embossify`, `degraded_crop`, `CITY_WORDS`, `LETTERS` |
| `textrender.py`, `typeset.py`, `fonts/` | HarfBuzz text shaping, so Bengali words are drawn correctly (result picture, synthetic plates) |
| `digits.py` | `DIGITS = "০১২৩৪৫৬৭৮৯"`, `normalize_digits` |
| `api.py`, `server.py` | HTTP routes (`/api/plate/*`) and the stand-alone service |
| `bangla_row.onnx`, `bangla_row_relief.onnx`, `bangla_top.onnx`, `bangla_top_relief.onnx` | the four deployed readers |
| `weights/` | `.pt` training weights of those four (and their synthetic-only starting points) |
| `scripts/` | training, fine-tuning, evaluation, labelling (section 9) |
| `scan.py`, `video.py`, `verdict*.py`, `deblur.py`, `ocr.py`, `classic.py`, `rowprep.py`, `degrade.py` | the **older route** (section 10); `read_plate` does not call them |

## 3. Stage 1 - find the plate (`pipeline.locate_plate`, `detector.py`)

* An image of at most **720 px** on its longest side with a width/height ratio of 0.55-3.5 is treated as a plate crop and goes
  straight to stage 2.
* Anything larger goes through the **YOLO-v9 tiny, 640 px, license-plate end-to-end** detector (via the `open-image-models` package,
  ONNX Runtime, CPU, confidence 0.25). To keep small, distant plates from vanishing at the model's fixed 640 px input, a frame
  larger than 900 px is also run as **nine overlapping tiles** (each 55 % of the frame, positions 0 / centre / end) after the full frame;
  the ten answers are merged with non-maximum suppression (IoU 0.4). One inference takes about 70 ms, a 1920×1080 frame about 0.7 s.
* The highest-confidence box is taken with an 8 % margin (12.8 % vertically) so the corner finder sees the whole plate edge. If YOLO
  finds nothing, the whole image goes on. **One plate per image** is read.

## 4. Stage 2 - OpenCV scan

`opencv.scan.scan(crop, relief="off", make_page=False)` returns the **rectified** (straightened) and **restored** plate. Both are used
as inputs to the readers (two views of the same plate). The plain page is made later, only for display (section 7). See
[`opencv/ARCHITECTURE.md`](../opencv/ARCHITECTURE.md).

## 5. Stage 3 - the readers

Four ONNX models, in two pairs. A pair reads **digits** and the **first row**.

| Model | File | Net | Params | Input | Output |
|---|---|---|---|---|---|
| Ink digit row | `bangla_row.onnx` | CRNN | 856,907 | 48×128 grey digit region | 6 digits |
| Embossed digit row | `bangla_row_relief.onnx` | same CRNN | 856,907 | 48×128 digit region of the relief view | 6 digits |
| Ink first row | `bangla_top.onnx` | `TopNet` | 1,801,971 | 48×160 grey top region | city word (19) + class letter (32) |
| Embossed first row | `bangla_top_relief.onnx` | same `TopNet` | 1,801,971 | 48×160 top region of the relief view | city + letter |

The detector is the fifth model (1,904,073 parameters, 7.8 MB). Total with the detector: **7,221,829 parameters**.

### 5.1 Digit reader: CRNN + CTC

```
 48×128 grey ─ per-image mean/std normalisation
   ─ conv 16 ─ pool 2×2 ─ conv 32 ─ pool 2×2 ─ conv 64 ─ conv 64 ─ pool 2×1 ─ conv 96 ─ pool 2×1   (3×3, BatchNorm, ReLU)
   ─ 32 time steps × (96·3 features) ─ Linear 160 ─ ReLU
   ─ 2-layer bidirectional LSTM (128) ─ Linear → 11 classes (10 digits + CTC blank) ─ log-softmax
```

* **Views.** The reader does not trust one crop. For each of the two plate images (rectified, restored) it cuts the digit region at
  six slightly different positions (`VIEWS`: start at 40 %, 34 %, 46 %, 40 %, 37 %, 43 % of the plate's height, with 0-4 % trimmed from
  each side), resizes each to 48×128, and runs the **12 views as one batch**.
* **Decode.** Greedy CTC decode of each view gives a digit string and a per-digit confidence (the highest probability of that
  character's frames). Only views that spell exactly six digits vote.
* **Vote.** For each position the digits are tallied, each weighted by its confidence; the winner's confidence is the mean of its
  supporters' confidences × the share of views that agreed. `agreement` is the share of views whose whole string equals the result.
  The per-position runner-ups (`alts`) are kept for the pipeline.
* **Embossed input.** The embossed reader first applies `opencv.relief.relief_view` (the plate's shading with lighting and colour
  removed, flat plate = 128) to each plate image. The same function makes the training images.

### 5.2 First-row reader: `TopNet`

```
 48×160 grey ─ per-image normalisation ─ 4 × (conv 3×3, BatchNorm, ReLU, max-pool): 32, 64, 128, 192 channels
   ─ flatten (192·3·10 = 5760) ─ Dropout 0.3 ─ Linear 256 ─ ReLU ─ Dropout 0.2
   ├─ city head   Linear → 19  (চট্ট মেট্রো, ঢাকা মেট্রো, ঝালকাঠি, নোয়াখালী, কুমিল্লা, রাঙ্গামাটি, চট্টগ্রাম, খুলনা মেট্রো, রাজশাহী মেট্রো,
   │                          সিলেট মেট্রো, বরিশাল মেট্রো, রংপুর মেট্রো, গাজীপুর, ময়মনসিংহ, নারায়ণগঞ্জ, বগুড়া, ফেনী, কক্সবাজার, ঢাকা)
   └─ letter head Linear → 32  (the class letters কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ)
```

The input is the **top 52 % of the plate**. It is a classifier over a fixed vocabulary, so it can only name those 19 words. The plain
"ঢাকা" was added last (appended, so the older word indices still work). One layer (5760 → 256) holds 1.47 M of the model's 1.8 M parameters.
Probabilities are averaged over the plate images.

### 5.3 ONNX Runtime

All sessions are created with the CPU provider and `ortutil.session_options()`: `ORT_INTRA_THREADS` / `ORT_INTER_THREADS` cap the threads
per model so several worker processes do not oversubscribe the machine (total ≈ workers × intra-op threads). A session is created
lazily on first use and shared across threads.

## 6. Stage 4 - the decision (`pipeline.read_plate`)

Each pair of readers produces a **candidate** (digits, per-digit confidence, agreement, per-position alternatives, city, letter). Both
pairs run on every plate; then:

1. **Best candidate.** The one with the higher `score = mean(digit confidence) × (0.5 + 0.5 × agreement)` (ties go to the ink pair).
   Its mode (`ink` or `relief`) is reported, and decides which page is drawn.
2. **Digits.** Every position takes the digit with the most support summed over **both** pairs (share × confidence). So a digit one
   pair misreads can be outvoted by the other; nothing is shown as "?" unless no reader produced six digits.
3. **First-row word** (`TOP_RULE = "ink_first"`). The ink first-row reader names the city. The embossed one takes over only if the ink one
   is unsure (confidence < 0.5) and the embossed one is more sure. The reason is measured, not assumed: on normal plates the ink reader
   named 20 of 22 words right and the embossed one 15 of 22 (it says "খুলনা মেট্রো" for Chattogram plates), because the ink one is
   the one trained on real normal plates. `relief_first` and `winner` rules remain selectable.
4. **Class letter.** It comes from the same reader as the city word, and is shown **only if its confidence is at least 0.60**. It is the
   weakest output (below 50 % right), so it is usually left out; the cleaned plate picture shows it.
5. **Verdict.** On the best candidate:

| Verdict | Condition |
|---|---|
| `readable` | six digits, **every** digit's confidence ≥ 0.80 and agreement ≥ 0.6 (and the voted digits equal the candidate's digits) |
| `uncertain` | mean confidence ≥ 0.55 and at least four digits ≥ 0.55 |
| `unreadable` | otherwise, or no six-digit reading |

If the two pairs disagree about a digit, a `readable` result is downgraded to `uncertain`. On the 26 held-out extreme plates every
`readable` result was a fully correct number (12/12), 5 of 12 `uncertain` ones were, and 0 of 2 `unreadable`.

The thresholds are heuristics: there is no large labelled set to calibrate them on.

## 7. Stage 5 - output

* **Page.** For the embossed pair, `opencv.relief.relief_page(restored)`; for the ink pair, `make_document` on the restored plate downscaled
  to about 128 px high (the size the page step is tuned for).
* **`render_report`**: one picture: the page on the left, on the right the verdict banner, the first-row words and the number, drawn with
  HarfBuzz-shaped text so conjuncts are correct. No confidence numbers are drawn.
* **API** (`api.py`):

| Endpoint | Purpose |
|---|---|
| `POST /api/plate/read` | **the pipeline.** Form: `file`, `mode` = `auto` / `ink` / `relief`. Returns `verdict`, `top`, `letter`, `digits`, `number`, `mode`, `corner_method`, `elapsed_s`, `images{result, page, straight, original}` (data URIs). At most `OCR_MAX_CONCURRENT` (2) reads per process; uploads capped at 25 MB (`OCR_MAX_UPLOAD_MB`); corrupt or empty files give 400 |
| `GET /api/plate/capabilities` | whether the models loaded |
| `POST /api/plate/scan`, `POST /api/plate/video`, `GET /api/plate/job/{id}`, `GET /api/plate/asset/...` | the older route (section 10) |

## 8. Training the readers

### 8.1 Synthetic plates (`synth_row.py`)

Real labelled plates are few, so the readers are first trained on generated ones:

* **`render_plate`**: a 300 px-high plate (aspect 1.25-1.95) with one of 12 body colours, uneven lighting, a frame line and bolts; six
  digits in a random Bangla font (ten font files in `ocr/fonts/` and, when present, five Nirmala UI faces from Windows), sometimes with an emboss shadow; a dash or a dot; the
  top row ("city-letter", or the city alone with the letter in front of the number) shaped with HarfBuzz.
* **`degraded_crop`**: how a real crop looks: perspective jitter, rotation ±7°, a strip of surroundings, downscale to 85-170 px wide, blur
  (65 %), motion blur (12 %), exposure and noise, JPEG (70 %).
* **`embossify`** (embossed style): the ink mask of the plate becomes a **height map**; light from a random direction shades its slopes
  (lit ridge, shadowed face), then paint-wear patches, grime and grain are added, with a `difficulty` knob. The characters now exist only
  as shading, as on a real stamped plate. 25 % of "relief" samples stay ink plates so the embossed readers cope with those too.
* Style `ink` feeds the grey crop to the model; style `relief` feeds `relief_view(crop)`.

Four sets of 20,000 plates (digit row and first row, ink and embossed; seeds 1 and 2) are saved in `synthetic data/`.

### 8.2 Training (`scripts/train_bangla_row.py`, `train_bangla_top.py`)

Digit row: CTC loss, AdamW (weight decay 1e-4), one-cycle learning rate (peak 2e-3), batch 64, gradient clip 5, brightness / offset /
noise / ±3 px shift augmentation applied on the GPU. First row: cross-entropy on both heads (an unknown label is ignored). A run of
6,000 steps takes 5-17 minutes on an RTX 2070 (longer when the GPU is shared or the CPU is busy generating data); generating 20,000 plates
takes 4-8 minutes on 10 CPU processes.

| Model | Result on synthetic plates |
|---|---|
| Ink digit row (retrained base) | 100 % exact |
| Embossed digit row | 91.7 % exact, 97.9 % of digits |
| Ink first row (retrained base) | city 100 %, letter 72 % |
| Embossed first row | city 96.7 %, letter 42 % |

Synthetic accuracy says nothing about real plates: it is the starting point.

### 8.3 Fine-tuning on real plates (`scripts/finetune_on_labels.py`)

A labelled plate (`file,digits,city,letter`) is run through the OpenCV scan; its views (12 for the digit row, 6 for the first row,
with the plate shifted ±3 %) become training samples, mixed with synthetic ones (35 % real per batch, heavy geometric augmentation for the
digit row), 800 steps at a learning rate of 3e-4. `--holdout` keeps test plates out, together with any other photo with the same plate
number (a second capture of the same plate would leak). `--init-dir ocr/weights/base_synthetic` restarts from the synthetic-only models.

### 8.4 What the deployed models are

| File | Trained on |
|---|---|
| `bangla_row.onnx` (ink digits) | the model shipped from earlier sessions (`weights/current/ink_row.pt`); which real plates it saw was not recorded |
| `bangla_row_relief.onnx`, `bangla_top_relief.onnx` | 20,000 synthetic embossed plates, then tuned on the 10 labelled embossed plates (9 photos, all Dhaka) |
| `bangla_top.onnx` (ink first row) | 20,000 synthetic ink plates, then tuned on all the labelled plates in `labels.csv` + the 10 embossed-set plates |

## 9. Evaluation (what is claimed, and how)

**Normal plates** (`eval/ocr_extreme/`): the 26 labelled plates of the extreme tier were excluded from fine-tuning (with same-number twins) and
read with those held-out models:

| | Digits | Whole number | First-row word | Class letter |
|---|---|---|---|---|
| Synthetic only, no real plate | 95/156 | 8/26 | 13/22 | 5/20 |
| **Full pipeline, tuned on real labels** | **136/156 (87 %)** | **17/26** | **20/22 (91 %)** | not printed |

**Embossed plates** (`eval/ocr_relief3d/`): five-fold leave-out cross-validation on the 10 labelled plates, embossed readers alone:
digits 57/60 (50/60 untuned), whole number 8/10 (6/10), city 10/10 (5/10), letter 7/10 (0/10). With only 10 labels, all Dhaka, 1-3 plates per
fold, this is optimistic. A model tuned on them reads those same 10 plates perfectly (memorised, not an accuracy). Checks that tuning did
no harm: on 40 other labelled plates the embossed digit reader improved from 172 to 181 of 240 digits, and on 38 tier plates the first-row
model still names "চট্ট মেট্রো" 14 times (15 before), i.e. it did not collapse to "ঢাকা".

Other scripts: `read_plates.py` (run a folder, save pictures, score against labels, swap models with `--ink-row` etc.), `make_plain_label_page.py`
(one HTML page to label photos in a browser), `cv_bangla.py` / `cv_top.py` (grouped cross-validation), `calibrate_*.py` (the older
reliability models), `eval_tiers.py`, `run_folder.py`, `plate_bench.py`.

## 10. The older route (kept, not used by `read_plate`)

`scan.py` (`scan_image`), `video.py`, `verdict.py` / `verdict_bn.py` (+ `*.json`), `deblur.py`, `ocr.py`, `rowprep.py`, `classic.py`, `degrade.py`. It
detects, restores, runs the Latin plate OCR (`fast-plate-ocr`) and the Bangla readers, deblurs with OCR guidance, scores reliability with a
logistic model, and (`video.py`) tracks a plate over frames, picks its sharpest frames, votes and fuses them. It serves the **Video** tab and the
older `/api/plate/scan` endpoint. `classic.py` is a template-matching experiment that did not reach the neural readers' accuracy.

## 11. Deployment and cost

* **Docker**: `Dockerfile.ocr` copies `ocr/` (with the four ONNX files) and `opencv/`, installs the CPU requirements and bakes the YOLO model into
  the image, so a running container needs no network (verified with `--network none`). The image is about 820 MB; a container uses about
  330-460 MB. `Dockerfile` packages the whole web app.
* **Settings**: `WORKERS`, `ORT_INTRA_THREADS`, `ORT_INTER_THREADS`, `OPENCV_THREADS`, `OCR_MAX_CONCURRENT`, `OCR_MAX_UPLOAD_MB`.
* **Cost per plate** (CPU): the four readers about 0.1 s together, YOLO 70 ms per 640 px inference (0.7 s for a 1080p frame with tiling), the OpenCV scan
  and page 1.5-3 s. A full read takes 2-3 s per plate crop and 3-4 s for a 1080p frame; the networks are about 5 % of that.
* **Load test**: 60 reads, 6 at a time, twice: every answer equal to the sequential one, memory 335 → 444 → 464 MB (levelling off).
* **Smaller models**: int8 quantisation shrinks the four readers 4× (21 → 5.4 MB) with almost no accuracy change but made them 5× *slower* on
  this CPU. Merging the ink and embossed pairs (5 → 3 models) and shrinking the 5760 → 256 layer are possible and need retraining and the
  same held-out test.

## 12. Limitations

* Accuracy is bounded by the **amount of real labelled data**: 121 normal plates and 10 embossed ones. Easy and medium tiers have no clean test
  (their labelled plates were used for tuning). More plates, from more cities and cameras, is the way to improve it.
* The first-row reader knows 19 words; any other city is named as the nearest one. The class letter is unreliable and mostly not printed.
* The embossed readers were tuned on 10 plates, all Dhaka; unseen embossed plates are the weakest case.
* One plate per image; a plate rotated by 90° is not recognised.
* The verdict thresholds are heuristics checked on 26 plates.
