# Merged reader (`ocr_merge/`) vs. the four separate readers

> **Update:** the sections below measure the first merged model (v1). The latest model (v2: label fix, class priors, 20 views + a denser re-read of doubtful plates, trained on 171 plates) is evaluated in [cv171/README.md](cv171/README.md): 91.9 % of digits and 72.5 % of whole numbers on plates it never saw (v1 recipe on the same folds: 89.7 % / 66.1 %).

The four Bengali readers (ink digit row, ink top row, embossed digit row, embossed top row) were merged into one network,
**PlateNet** (`ocr_merge/plate_net.onnx`): one shared CNN, a CTC digit head and a city/letter head, reading both the ink look
and the embossed (`relief_view`) look. One ONNX call per plate instead of four. YOLO and the OpenCV stage are unchanged; the
decision rules are the same code (`ocr.pipeline.build_report`). Every image was scanned once and both pipelines read that same
scan, so differences come from the readers alone.

## Size, memory, speed (12-thread PC, CPU, 30 images: 10 extreme + 10 easy + 10 from new samples+3d)
| | Separate (4 models) | Merged (1 model) |
|---|---|---|
| Model files | 20.3 MB | **5.9 MB** (-71 %) |
| Parameters | 5,317,756 | **1,537,790** (-71 %) |
| ONNX sessions / calls per plate | 4 / 4 | 1 / 1 |
| Load time (sessions + first call) | 0.44 s | 0.25 s |
| Memory added by the readers | 77 MB | 76 MB |
| Process memory after reading / peak | 166 / 190 MB | 160 / 192 MB |
| Reader step per plate, default ONNX threads | 156 ms | **74 ms** |
| Reader step, `ORT_ALLOW_SPINNING=0` | 48 ms | 57 ms |
| Read without the display page (mean) | 1.40 s | 1.25 s |
| Full `read_plate` incl. display page (mean) | 2.23 s | 2.39 s |

Notes: the memory is ONNX Runtime + OpenCV, not the weights, so it hardly changes. With default threads the merged reader
is 2x faster (fewer calls to wake 12 threads for); with spinning off the separate readers are as fast or faster (the merged
backbone is wider on the digit row). The full read is ~0.1 s slower only because the merged model picks the "ink" look more
often on embossed plates, and the ink plain page costs 1.6 s vs 0.2 s for the relief page (display only, not the reading).
Raw numbers: `benchmark_default.json`, `benchmark_nospin.json`.

## Accuracy
### Honest (held-out: the tested plates and their twins were not in training)
| | Digits | Whole number | City word | Letter shown right | "readable" and right |
|---|---|---|---|---|---|
| 26 extreme plates - separate | 136/156 | 17/26 | 20/22 | - | 12/12 |
| 26 extreme plates - **merged** | **137/156** | 17/26 | 20/22 | - | 14/14 |
| 10 embossed plates, 5 folds - separate | 57/60 | 8/10 | 9/10 | 3/10 | |
| 10 embossed plates, 5 folds - **merged** | 56/60 | 8/10 | 9/10 | 2/10 | |

The separate embossed baseline uses the relief readers of each fold and an ink top-row reader re-tuned without that fold
(`ocr_merge/scripts/separate_ink_top_folds.py`): the deployed ink top-row reader was tuned on these 10 plates.

### All images with the deployed models (not a fair accuracy number: see below)
| | Images | Labelled | Digits | Whole | City | Verdicts readable / uncertain / unreadable |
|---|---|---|---|---|---|---|
| LPR catagory - separate | 152 | 81 | 451/486 | 58/81 | 75/75 | 95 / 54 / 3 |
| LPR catagory - merged | 152 | 81 | 486/486 | 81/81 | 75/75 | 131 / 20 / 1 |
| new samples+3d - separate | 11 | 10 | 60/60 | 10/10 | 10/10 | 10 / 0 / 0 |
| new samples+3d - merged | 11 | 10 | 60/60 | 10/10 | 10/10 | 10 / 0 / 0 |

The deployed merged model was tuned on all these labels, so its 100 % is memorised. The shipped separate digit reader was
not tuned on them, so the two rows are not comparable; use the held-out table.

## Verdict
Same accuracy (within one digit) at 29 % of the size and parameters. Faster reader step with default threads, equal with
spinning off; memory unchanged. Known change: on embossed plates the merged model's "ink" look often wins, so the cleaned
picture is the ink plain page instead of the relief page (the reading itself is right).

## Files
`lpr/`, `new3d/`, `heldout_extreme/`, `heldout_embossed/fold*/` each have `separate/` and `merged/` with `results.csv`,
`results/` (one picture per plate) and `sheet.jpg`; `summary.json` has the totals.
Reproduce: `ocr_merge/scripts/train_merged.py`, `finetune_all.sh`, `separate_ink_top_folds.py`, `compare.py`, `benchmark.py`.
