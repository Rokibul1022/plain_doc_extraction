# Training weights (`.pt`) of the readers - for further tuning

The deployed models are the ONNX files in `ocr/`; these PyTorch weights are what you need to **continue training**. Each file was
checked to reproduce its ONNX model exactly (same outputs on random input).

| File | Reproduces | Architecture | What it was trained on |
|---|---|---|---|
| `current/ink_row.pt` | `ocr/bangla_row.onnx` | CRNN (`train_bangla_row.CRNN`, 857k params) | the shipped ink digit model from earlier training sessions; which real plates it saw was not recorded |
| `current/relief_row.pt` | `ocr/bangla_row_relief.onnx` | same CRNN | 20,000 synthetic embossed plates, then tuned on the 10 labelled embossed plates |
| `current/ink_top.pt` | `ocr/bangla_top.onnx` | `train_bangla_top.TopNet` (1.8M params, 19 city words + 32 letters) | 20,000 synthetic ink plates, then tuned on all 121 labels + the 10 embossed-set plates |
| `current/relief_top.pt` | `ocr/bangla_top_relief.onnx` | same TopNet | 20,000 synthetic embossed plates, then tuned on the 10 embossed plates |
| `base_synthetic/*.pt` | - | same | the same four models **before** any real plate (synthetic only). Start from these to re-tune without carrying over memorised plates, e.g. for an honest held-out test |

The `.pt` files are about 41 MB and are **ignored by git** (see `.gitignore`); keep this folder, or remove the ignore line to track it.
The synthetic datasets they were trained on are in `../../synthetic data/` (555 MB, also not tracked by git).

## Continue tuning
```bash
# more labelled plates (file,digits,city,letter): start from what is deployed ...
python ocr/scripts/finetune_on_labels.py labels.csv <image folder> --style ink    --out tuned_ink/
python ocr/scripts/finetune_on_labels.py labels.csv <image folder> --style relief --out tuned_relief/
# ... or from the clean synthetic start, keeping some plates out for an honest test
python ocr/scripts/finetune_on_labels.py labels.csv <image folder> --style ink --out tuned_ink/ \
       --init-dir ocr/weights/base_synthetic --holdout a.png,b.png
python ocr/scripts/read_plates.py <folder with the held-out plates> --out results/ --truth heldout.csv \
       --ink-row tuned_ink/ink_row.onnx --ink-top tuned_ink/ink_top.onnx
```
To deploy: copy the new `.onnx` files over `ocr/bangla_row.onnx`, `bangla_top.onnx`, `bangla_row_relief.onnx`, `bangla_top_relief.onnx`
and the `.pt` files into `current/`, rebuild the Docker image. Re-run the held-out test before trusting the numbers: the first-row rule
(`TOP_RULE`) and the verdict thresholds in `ocr/pipeline.py` were chosen on the current models.
