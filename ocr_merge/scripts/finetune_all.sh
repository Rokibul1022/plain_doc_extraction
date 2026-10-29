#!/usr/bin/env bash
# Fine-tune the merged reader from the synthetic start: the deployed model (all labels) and the held-out models used by
# compare.py (the 26 extreme test plates out; each of the 5 embossed folds out). Run from the repository root.
set -uo pipefail  # not -e: python can return 127 at exit after CUDA training although the model was saved
W=ocr_merge/weights
STEPS=${STEPS:-1500}
tune() {  # name, extra args...
  local name=$1; shift
  python -u -W ignore ocr_merge/scripts/train_merged.py --init $W/base_synthetic.pt --real --steps "$STEPS" --lr 3e-4 \
    --save "$W/$name.pt" --onnx-out "$W/$name.onnx" "$@" > "$W/train_$name.log" 2>&1
  grep -q "^exported" "$W/train_$name.log" || { echo "training $name failed: see $W/train_$name.log"; exit 1; }
  tail -2 "$W/train_$name.log"
}
[ -n "${SKIP_MAIN:-}" ] || tune plate_net
cp $W/plate_net.onnx ocr_merge/plate_net.onnx
tune plate_net_extreme_out --holdout-csv labels_extreme_test.csv
FOLDS=("46d4aeaf-f0ce-4362-82e4-4d1495067caa.jpeg,Screenshot_5.png" "Screenshot_6.png,Screenshot_7.png"
       "Screenshot_10.png,Screenshot_11.png,Screenshot_11 (1).png" "Screenshot_12.png,Screenshot_13.png" "Screenshot_14.png")
for k in 0 1 2 3 4; do tune "plate_net_emb_fold${k}_out" --holdout "${FOLDS[$k]}"; done
echo ALL_FINETUNES_DONE
