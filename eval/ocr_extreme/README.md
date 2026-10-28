# Normal plates: OCR pipeline on `LPR catagory/extreme` (51 images)

## What was trained
Fresh **ink readers** (digit row + a 19-class top row that includes plain "ঢাকা") were trained from scratch on 20,000 synthetic
plates (they had never seen a real plate), then fine-tuned on the real labels in `labels.csv`.

## The test is held out
All 121 labels are photos from `LPR_200 1st`, which are the same files as the tier folders; 26 of the extreme plates are labelled
(`labels_extreme_test.csv`; 6 more have no complete number). To keep the test honest:
- the 26 extreme plates were **excluded from training**, and so was any other photo with the same plate number (no twins);
- fine-tuned on the remaining plates (82 labelled plates + the 10 embossed-set plates for the top reader), 800 steps.

`results.csv`, `results/` and `sheet.jpg` were produced by the pipeline **using these held-out models** (`--ink-row/--ink-top`),
so they are a fair picture. 25 of the 51 images have no label and are shown but not scored.

## Results on the 26 labelled extreme plates (held out)
| | Digits right | Whole number | City word (first row) | Class letter |
|---|---|---|---|---|
| Fresh readers, synthetic only (no real plates) | 95/156 | 8/26 | 13/22 | 5/20 |
| **Fine-tuned on the labels, extreme excluded** | 130/156 | 16/26 | 20/22 | 8/20 |
| **Full pipeline (this folder's `results.csv`)** | **136/156 (87%)** | **17/26** | **20/22 (91%)** | 0/20 (not printed, see below) |
| Shipped readers before this work (may have seen some of these plates, so not a clean number) | 142/156 | 19/26 | 21/22 | 14/20 |

Whole number correct **by verdict** (held-out): readable 12/12, uncertain 5/12, unreadable 0/2 - the verdict is a usable flag.

## Why the first row was wrong before
The first-row rule introduced for embossed plates preferred the embossed reader's word, and that reader says things like
"খুলনা মেট্রো" for Chattogram plates (15/22 right). The default rule is now `ink_first`: the real-data-trained ink reader names
the words; the embossed reader takes over only when the ink reader is unsure. The shipped top-row model also had no plain
"ঢাকা" class; the new one has.

## Deployed vs. tested
Deployed: `ocr/bangla_top.onnx` = the new 19-class reader **fine-tuned on all labels** (its score on these plates would be
memorised, so it is not reported). The ink **digit-row** model was **not** replaced: the retrained one scored below the shipped one
on this test (130 vs 142), and I cannot tell how much of the shipped model's score is from having seen these plates, so I kept it.
The class letter is shown only when confident (never, here: letter accuracy is below 50%); the cleaned plate image shows it.
`tuning_results.json` holds the per-plate results of every model compared.
