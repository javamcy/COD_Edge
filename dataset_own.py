# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp')


def _list_dir(d):
    fs = [f for f in os.listdir(d) if f.lower().endswith(IMG_EXT)]
    return sorted(fs)


class TrainSet(data.Dataset):
    """Imgs + GT + Edge, same stem. Aug: random h-flip, light color jitter (CPU-cheap).
    Optional depth prior (grayscale, precomputed by Depth-Anything-V2) for depth-privileged training."""

    def __init__(self, root, trainsize=384, with_depth=False):
        self.size = trainsize
        self.img_dir = os.path.join(root, 'Imgs')
        self.gt_dir = os.path.join(root, 'GT')
        self.edge_dir = os.path.join(root, 'Edge')
        self.depth_dir = os.path.join(root, 'depth')
        self.with_depth = with_depth
        self.names = _list_dir(self.img_dir)
        self.img_tf = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ColorJitter(0.15, 0.15, 0.1, 0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.mask_tf = transforms.Compose([transforms.Resize((trainsize, trainsize)), transforms.ToTensor()])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        stem = os.path.splitext(n)[0]
        img = Image.open(os.path.join(self.img_dir, n)).convert('RGB')
        gt = Image.open(os.path.join(self.gt_dir, stem + '.png')).convert('L')
        ep = os.path.join(self.edge_dir, stem + '.png')
        edge = Image.open(ep).convert('L') if os.path.exists(ep) else gt
        if self.with_depth:
            dp = None
            for ext in ('.jpg', '.png', '.bmp'):
                cand = os.path.join(self.depth_dir, stem + ext)
                if os.path.exists(cand):
                    dp = cand
                    break
            depth = Image.open(dp).convert('L') if dp else Image.new('L', gt.size, 0)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
            edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
            if self.with_depth:
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        img = self.img_tf(img)
        gt = self.mask_tf(gt)
        gt = (gt > 0.5).float()
        edge = self.mask_tf(edge)
        edge = (edge > 0.5).float()
        if self.with_depth:
            depth = self.mask_tf(depth)
            return img, gt, edge, depth, stem
        return img, gt, edge, stem


class ValSet(data.Dataset):
    """Deterministic resize, no aug."""

    def __init__(self, img_dir, gt_dir, size=384):
        self.img_dir, self.gt_dir, self.size = img_dir, gt_dir, size
        self.names = _list_dir(img_dir)
        self.img_tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.mask_tf = transforms.Compose([transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
                                           transforms.ToTensor()])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        stem = os.path.splitext(n)[0]
        img = self.img_tf(Image.open(os.path.join(self.img_dir, n)).convert('RGB'))
        gt = self.mask_tf(Image.open(os.path.join(self.gt_dir, stem + '.png')).convert('L'))
        return img, (gt > 0.5).float(), stem


def get_loader(root, batchsize, trainsize=384, workers=6, with_depth=False):
    ds = TrainSet(root, trainsize, with_depth=with_depth)
    return data.DataLoader(ds, batch_size=batchsize, shuffle=True, num_workers=workers,
                           pin_memory=True, drop_last=True, persistent_workers=True)


def get_val_loader(img_dir, gt_dir, size=384, workers=4):
    ds = ValSet(img_dir, gt_dir, size)
    return data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=workers, pin_memory=True)
