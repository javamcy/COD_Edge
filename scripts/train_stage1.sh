#!/usr/bin/env bash
# Stage 1: partial-unfreeze training at 448 (50 epochs, batch 8).
# Paper protocol: last 8 of 24 DINOv3 blocks unfrozen; SAD + DetailStem + AFD on.
set -e
python train.py \
  --arch dino3 --dino_weight weights/dinov3_vitl16.pth \
  --use_sad 1 --use_detail 1 --use_pc 0 --use_afd 1 \
  --freeze_enc 1 --unfreeze_blocks 8 \
  --epoch 50 --batchsize 8 --trainsize 448 --lr 1e-4 --bl_mult 0.25 \
  --w_ot 0 --w_side 0.35 --w_edge 0.5 --w_sc 0.25 --p_dual 0.3 --scale2 384 \
  --val_every 2 --decay_end 50 --decay_pow 0.9 \
  --train_root data/train --val_img data/camo_test/Imgs --val_gt data/camo_test/GT \
  --save_path results/stage1
