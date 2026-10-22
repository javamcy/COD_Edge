# -*- coding: utf-8 -*-
"""Evaluate a local prediction directory with the same evaluator as the paper.

Usage:
  python eval_local.py --pred local_run/pred_ds768/camo_test --gt local_run/datasets/camo_test/GT --name DS768-CAMO
  python eval_local.py --pairs pairs_local.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaltools.metrics import (Emeasure, Fmeasure_and_FNR, MAE, Smeasure,  # noqa: E402
                               WeightedFmeasure)

IMG_EXT = ('.png', '.jpg', '.bmp', '.jpeg')


def find_gt(gt_dir, name):
    stem = os.path.splitext(name)[0]
    for ext in IMG_EXT:
        p = os.path.join(gt_dir, stem + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError('no GT for %s in %s' % (name, gt_dir))


def one(job):
    pred_dir, gt_dir, name = job
    gt = np.array(Image.open(find_gt(gt_dir, name)).convert('L'))
    pred = Image.open(os.path.join(pred_dir, name)).convert('L')
    if pred.size != (gt.shape[1], gt.shape[0]):
        pred = pred.resize((gt.shape[1], gt.shape[0]), Image.BILINEAR)
    pred = np.array(pred, dtype=np.uint8)
    sm, em, fm, wfm, mae = (Smeasure(), Emeasure(), Fmeasure_and_FNR(),
                            WeightedFmeasure(), MAE())
    for m in (sm, em, fm, wfm, mae):
        m.step(pred, gt)
    return (float(sm.get_results()['sm']), float(em.get_results()['em']['adp']),
            float(wfm.get_results()['wfm']), float(mae.get_results()['mae']))


def evaluate(pred_dir, gt_dir, workers):
    names = sorted(f for f in os.listdir(pred_dir) if f.lower().endswith(IMG_EXT))
    with Pool(workers) as pool:
        res = pool.map(one, [(pred_dir, gt_dir, n) for n in names], chunksize=16)
    arr = np.array(res, dtype=np.float64)
    return {'n': len(names), 'Salpha': float(arr[:, 0].mean()),
            'Ephi': float(arr[:, 1].mean()), 'Fbetaw': float(arr[:, 2].mean()),
            'MAE': float(arr[:, 3].mean())}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred')
    ap.add_argument('--gt')
    ap.add_argument('--name', default='')
    ap.add_argument('--pairs')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    if args.pairs:
        pairs = json.load(open(args.pairs))
        out = {}
        for tag, d in pairs.items():
            out[tag] = evaluate(d['pred'], d['gt'], args.workers)
            print(tag, out[tag], flush=True)
        json.dump(out, open(args.pairs.replace('.json', '_result.json'), 'w'), indent=1)
    else:
        tag = args.name or os.path.basename(os.path.normpath(args.pred))
        r = evaluate(args.pred, args.gt, args.workers)
        print(tag, r)
