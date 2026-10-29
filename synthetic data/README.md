# Synthetic training plates

The 20,000-plate datasets the current readers were trained on (before any real plate), kept so training can be repeated or continued
without waiting for them to be generated (about 4 minutes each, on 10 CPU processes). Together 555 MB; **not tracked by git**.

| File | Size | Model it trains | Arrays |
|---|---|---|---|
| `ink_row_20k.npz` | 124 MB | ink digit row (`ocr/weights/*/ink_row.pt`) | `x` uint8 (20000, 48, 128) grey digit region; `y` int64 (20000, 6) digit indices (0-9 = ০-৯) |
| `relief_row_20k.npz` | 124 MB | embossed digit row | same, plates embossed in their own colour and shown through `opencv.relief.relief_view` |
| `ink_top_20k.npz` | 154 MB | ink first row | `x` uint8 (20000, 48, 160); `c` int64 city index into `ocr/synth_row.CITY_WORDS` (19 words); `l` int64 class-letter index into `LETTERS` (-1 = no separate letter) |
| `relief_top_20k.npz` | 154 MB | embossed first row | same, embossed |

Made with `ocr/synth_row.py` (`render_plate`, `embossify`, `degraded_crop`) through `ocr/scripts/train_bangla_row.py` and
`train_bangla_top.py` (`--style ink|relief --samples 20000 --dataset <file>`; seeds 1 for the row and 2 for the first row). The plates
use the fonts in `ocr/fonts/` plus Windows' Nirmala UI when present, so regenerating on another machine gives similar but not
identical plates. If you change `CITY_WORDS` (the model's output size) or the rendering, regenerate them.

Use: `python ocr/scripts/finetune_on_labels.py labels.csv <folder> --style ink --out tuned_ink/` picks these files up automatically;
the training scripts take them with `--dataset "synthetic data/ink_row_20k.npz"`.
Older, larger datasets from the first training sessions (`syn_row_v2`, `syn_row_60k`, `syn_top`; about 1.2 GB) are not included.
