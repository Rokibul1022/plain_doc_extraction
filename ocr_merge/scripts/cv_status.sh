#!/usr/bin/env bash
# Print one status line whenever a cross-validation training run advances:   bash ocr_merge/scripts/cv_status.sh NAME [steps]
# (fold being trained, step, time per fold, folds finished, GPU use). Exits when all 5 folds are saved.
NAME=$1; STEPS=${2:-1500}; D=ocr_merge/weights/cv/$NAME; prev=""; t0=$(date +%s)
while true; do
  done_n=0; cur=""; step=""
  for k in 0 1 2 3 4; do
    if [ -f "$D/fold$k.pt" ]; then done_n=$((done_n+1)); elif [ -f "$D/train_fold$k.log" ] && [ -z "$cur" ]; then
      cur=$k; step=$(grep -o "^step *[0-9]*" "$D/train_fold$k.log" | tail -1 | grep -o "[0-9]*$"); fi
  done
  now="fold=$cur step=${step:-0} done=$done_n"
  if [ "$now" != "$prev" ]; then
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr -d ' ')
    el=$(( $(date +%s) - t0 ))
    if [ "$done_n" -ge 5 ]; then echo "[$(date +%H:%M:%S)] $NAME: all 5 folds trained (${el}s)"; exit 0; fi
    echo "[$(date +%H:%M:%S)] $NAME: training fold ${cur:-?} of 5, step ${step:-0}/$STEPS | folds finished $done_n/5 | GPU $gpu | elapsed ${el}s"
    prev="$now"
  fi
  sleep 10
done
