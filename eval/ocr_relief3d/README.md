# OCR on the 10 labelled plates (`new samples+3d`, 9 distinct photos)

`labels_embossed10.csv` are the labels typed by the project owner (file, digits, city, letter).

## What was done
The two **relief readers** (digit row `ocr/bangla_row_relief.onnx`, top row `ocr/bangla_top_relief.onnx`) were first trained on
20,000 synthetic embossed plates, then fine-tuned on these 10 plates (96 digit-row and 48 top-row views, with synthetic plates
mixed in, 800 steps). The shipped models in `ocr/` are the ones tuned on **all 10**.

## Honest numbers: leave-out cross-validation
5 folds; each fold's plates (both copies of one photo stay together) were tested by a model tuned on the other plates only.
Relief readers alone:

| | Digits right | Whole number | City word | Class letter |
|---|---|---|---|---|
| Untuned | 50/60 | 6/10 | 5/10 | 0/10 |
| **Tuned, held-out** | **57/60** | **8/10** | **10/10** | **7/10** |

`results.csv`, `sheet.jpg` and `results/` show the **final pipeline** on the same 10 plates: all read exactly as labelled. That
is **memorised, not accuracy** (the model trained on them); the held-out table above is the estimate.

## Checks that tuning did no harm
- 40 *other* labelled plates (LPR_200): relief digit reader 172/240 digits (14 exact) untuned -> 181/240 (18 exact) tuned.
- 38 tier plates (mostly Chattogram): the top-row model still names "চট্ট মেট্রো" 14 times (untuned 15), so it did not collapse to "ঢাকা".

## Read this before trusting it
- Only 10 labels, 9 photos, and **all are Dhaka plates**: the 10/10 city and 7/10 letter are optimistic, and the held-out sets
  were 1-3 plates each. A real accuracy needs more photos of other embossed plates (cities, angles, wear).
- The verdict "readable" is not calibrated for the tuned models: it says "readable" for all 10 (they were memorised).
- Files: `table.md` per-plate comparison, `tuning_results.json` raw per-plate results, `labels_embossed10.csv` the labels.
