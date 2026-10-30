# 63 new plates (`new_merge-63`), inference only

Read by the latest merged model (`ocr_merge/plate_net.onnx`, v2) on the GPU (RTX 2070), with the deployed settings: 20 views per crop,
class priors, plates that are not "readable" re-read with 55 views.

The plates are not in the training data (no file name or content match with any earlier photo) but have **no labels yet**, so no accuracy
is computed. `merged/results.csv` has the reading and verdict of each plate, `merged/results/` one picture per plate (cleaned plate + reading),
`merged/sheet.jpg` all of them on one sheet. Verdicts: 32 readable, 27 uncertain, 4 unreadable.

To score: label them (`python ocr/scripts/make_plain_label_page.py new_merge-63 --out new_merge-63/label_page.html`) and run
`python ocr_merge/scripts/test_folder.py new_merge-63 --gpu --truth <labels.csv>`.
