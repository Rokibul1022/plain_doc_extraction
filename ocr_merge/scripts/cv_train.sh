#!/usr/bin/env bash
# Train one variant on the 5 grouped folds, one at a time (JOBS=1, the default) on one GPU. Run from the repository root.
#   bash ocr_merge/scripts/cv_train.sh NAME [train_merged.py options ...]      -> ocr_merge/weights/cv/NAME/fold{0..4}.pt
# Defaults: start from the synthetic weights, fine-tune on the real plates for 1500 steps at lr 3e-4 (the deployed recipe).
# (Parallel jobs do not help on one 8 GB GPU: 3 at once were ~5x slower each, 5 at once thrash.)
set -uo pipefail
NAME=$1; shift
D=ocr_merge/weights/cv/$NAME
JOBS=${JOBS:-1}
mkdir -p "$D"
ARGS=(--init ocr_merge/weights/base_synthetic.pt --real --steps 1500 --lr 3e-4)
for k in 0 1 2 3 4; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
  python -u -W ignore ocr_merge/scripts/train_merged.py "${ARGS[@]}" "$@" --fold $k --save "$D/fold$k.pt" > "$D/train_fold$k.log" 2>&1 &
done
wait
ok=0; for k in 0 1 2 3 4; do [ -f "$D/fold$k.pt" ] && ok=$((ok+1)); done
echo "$NAME: $ok/5 folds trained"
