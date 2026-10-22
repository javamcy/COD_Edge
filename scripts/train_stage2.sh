#!/usr/bin/env bash
# Stage 2: band-gated CGR refinement at 768 (12 epochs, batch 8, refine-only).
set -e
python train.py \
  --arch dino7 --dino_weight weights/dinov3_vitl16.pth \
  --use_sad 1 --use_detail 1 --use_pc 0 --use_afd 1 --use_refine 2 --refine_only 1 \
  --load results/stage1/Net_epoch_best.pth --load_strict 0 \
  --epoch 12 --batchsize 8 --trainsize 768 --lr 1e-4 --bl_mult 0.25 \
  --w_ot 0 --w_side 0 --w_edge 0 --w_sc 0 --p_dual 0 \
  --val_every 1 --decay_end 12 --decay_pow 0.9 \
  --train_root data/train --val_img data/camo_test/Imgs --val_gt data/camo_test/GT \
  --save_path results/stage2
