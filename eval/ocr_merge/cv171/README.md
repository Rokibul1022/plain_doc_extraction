# Merged model v2: cross-validated accuracy on all 171 labelled plates

Every plate is read by a model that never saw it **or any other photo with the same plate number** (5 grouped folds, `ocr_merge/cv.py`;
each fold model is trained on about 137 plates). Same folds for every row, so rows are comparable; with 171 plates one plate is
0.6 points, so differences of 1-2 plates are noise. Reproduce: `ocr_merge/scripts/cv_train.sh` + `cv_eval.py`.

| Step (each row adds to the one above) | Digits | Whole number | City word | Class letter |
|---|---|---|---|---|
| v1 recipe (6 views, letter-label bug) | 89.7 % (920/1026) | 66.1 % (113/171) | 87.5 % (140/160) | 46.6 % (69/148) |
| fix the letter labels (a blank letter was trained as "ক") | 90.3 % | 65.5 % | 85.0 % | 53.4 % (79/148) |
| + class priors (city-conditioned letter prior) | 90.3 % | 65.5 % | 86.9 % | 56.1 % (83/148) |
| + 20 digit-row views per crop instead of 6 | 91.2 % (936) | 70.2 % (120) | 87.5 % (140) | 56.1 % |
| **+ re-read plates that are not "readable" with 55 views = v2** | **91.9 % (943)** | **72.5 % (124)** | 87.5 % | 56.1 % (shown letters: 82/130 right) |

Verdict of v2 (whole number right / plates): readable 86/97, uncertain 38/61, unreadable 0/13.

Tried and not adopted (all within noise of the row above): forcing exactly six digits per view (`ocr_merge/decode.py`; 89.9 % / 64.9 % on the v1
models), training on 20 views per crop (69.0 % whole), averaging the weights of two fine-tunes (68.4 %), 55 views for every plate
(72.5 %, same as the adaptive re-read but slower), class prior on the city word (no gain).

The deployed `ocr_merge/plate_net.onnx` is trained on all 171 plates, so its own score on them is memorised; the table above is the
estimate for new plates (a model trained on all 171 should be a little better than the 137-plate fold models).
What is left are mostly plates that are tiny, blurry, strongly tilted or have a second plate in the crop; more labelled plates is the next lever.

Files: `base_buggy_labels.json` (v1 recipe), `base.json`, `base_views20.json`, `base_views55.json`, `base_adaptive.json` (v2),
`v20_views20.json`, `soup_bv_views20.json`; each has the summary and every plate's reading.
