# -*- coding: utf-8 -*-
"""Inference + evaluation for the proposed model.

Model: DINOv3 ViT-L/16 (last 8 of 24 blocks unfrozen) + SAD + DetailStem + AFD + band-gated CGR.
Usage:
  python test.py --ckpt weights/v8uCGR_best.pth --data_root data --out_root results/pred
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from ournet_v7 import OursNetV7
from dataset_own import ValSet

SETS = {'CAMO': 'camo_test', 'COD10K': 'cod10k_test', 'NC4K': 'nc4k'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='weights/v8uCGR_best.pth')
    ap.add_argument('--data_root', default='data', help='contains camo_test/, cod10k_test/, nc4k/')
    ap.add_argument('--out_root', default='results/pred')
    ap.add_argument('--trainsize', type=int, default=768, help='768 for the paper headline; 896 for the matched-input study')
    ap.add_argument('--dino3_weight', default='',
                    help='DINOv3 ViT-L/16 weights. Can be empty for inference: the checkpoint contains the backbone.')
    ap.add_argument('--use_sad', type=int, default=1)
    ap.add_argument('--use_detail', type=int, default=1)
    ap.add_argument('--use_pc', type=int, default=0)
    ap.add_argument('--use_afd', type=int, default=1)
    ap.add_argument('--use_refine', type=int, default=2)
    ap.add_argument('--sets', default='CAMO,COD10K,NC4K')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--eval_workers', type=int, default=8)
    opt = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OursNetV7(opt.dino3_weight,
                      use_sad=bool(opt.use_sad), use_detail=bool(opt.use_detail),
                      use_pc=bool(opt.use_pc), use_afd=bool(opt.use_afd),
                      use_refine=int(opt.use_refine))
    sd = torch.load(opt.ckpt, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print('checkpoint loaded: missing=%d unexpected=%d' % (len(missing), len(unexpected)))
    if missing:
        print('  missing (first 5):', missing[:5])
    if unexpected:
        print('  unexpected (first 5):', unexpected[:5])
    model.to(dev).eval()

    pairs = {}
    for name in [s.strip() for s in opt.sets.split(',') if s.strip()]:
        folder = SETS[name]
        img_dir = os.path.join(opt.data_root, folder, 'Imgs')
        gt_dir = os.path.join(opt.data_root, folder, 'GT')
        out_dir = os.path.join(opt.out_root, name)
        os.makedirs(out_dir, exist_ok=True)

        ds = ValSet(img_dir, gt_dir, opt.trainsize)
        dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=opt.workers, pin_memory=True)
        for imgs, _, stems in tqdm(dl, desc=name):
            imgs = imgs.to(dev)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(dev.type == 'cuda')):
                    out = model(imgs)
                    p = torch.sigmoid(out['main_full'] if 'main_full' in out else out['main']).float()
                    p = F.interpolate(p, size=imgs.shape[-2:], mode='bilinear', align_corners=False)
            p = p[:, 0].cpu().numpy()
            for b, stem in enumerate(stems):
                im = Image.fromarray((p[b] * 255).astype(np.uint8))
                gt_path = os.path.join(gt_dir, stem + '.png')
                if os.path.exists(gt_path):
                    with Image.open(gt_path) as g:
                        im = im.resize(g.size, Image.BILINEAR)
                im.save(os.path.join(out_dir, stem + '.png'))
        key = name + '-Test' if name != 'NC4K' else name
        pairs[key] = {'pred': out_dir + os.sep, 'gt': gt_dir + os.sep}
        print('saved %s (%d images)' % (name, len(ds)))

    os.makedirs(os.path.dirname(os.path.abspath(opt.out_root)) or '.', exist_ok=True)
    pj = os.path.join(os.path.dirname(os.path.abspath(opt.out_root)) or '.', 'pairs.json')
    with open(pj, 'w') as f:
        json.dump(pairs, f, indent=2)
    print('pairs ->', pj)

    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_local.py'),
                    '--pairs', pj, '--workers', str(opt.eval_workers)], check=True)


if __name__ == '__main__':
    main()
