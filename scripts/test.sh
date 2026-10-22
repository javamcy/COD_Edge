#!/usr/bin/env bash
# Inference + evaluation on CAMO / COD10K / NC4K at 768 (headline setting).
# For the matched-input 896 study, change --trainsize to 896.
set -e
python test.py \
  --ckpt weights/v8uCGR_best.pth \
  --data_root data \
  --out_root results/pred \
  --trainsize 768 \
  --dino3_weight "" \
  --use_sad 1 --use_detail 1 --use_pc 0 --use_afd 1 --use_refine 2
