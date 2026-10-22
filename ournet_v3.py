# -*- coding: utf-8 -*-
"""Ours v3/v5: DINOv3 ViT-L/16 (timm-verified foundation encoder) + spectral/physics-inspired head.

Innovation story ("spectral camouflage detection"): camouflage defeats intensity-amplitude cues.
We attack it with three complementary, ablation-defensible, physics/signal-processing-grounded
modules:
  (1) SAD  - Spectral Amplitude Decoupling: adaptive radial-band gain on the amplitude spectrum
             (texture statistics) with the phase kept intact (object structure).
  (2) PCB  - Phase-Congruency Boundary prior: contrast-invariant boundary map from a *fixed*
             quadrature (Gabor) filter bank applied in feature space; recovers boundaries that
             amplitude/gradient detectors miss on low-contrast camouflage.
  (3) AFD  - Anisotropic Feature Diffusion: unrolled Perona-Malik nonlinear heat equation on
             features; suppresses camouflage texture while keeping boundaries sharp.
Plus a lightweight image-side DetailStem that injects high-frequency detail at 1/16, 1/8, 1/4.
All modules are individually toggleable for ablation (use_sad / use_pc / use_afd / use_detail).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ournet import SAD


def load_dinov3_vit(timm_name, weight_path):
    """Build the timm Eva DINOv3 ViT and load the raw facebook checkpoint (verified: 0 missing/unexpected).

    The raw ckpt uses 'storage_tokens'/'ls1.gamma'/'ls2.gamma' naming; qkv.bias is all-zero for the
    distilled b/l/h models (discarded); rope periods are config-derived (not persisted).
    """
    import timm
    m = timm.create_model(timm_name, pretrained=False, num_classes=0, dynamic_img_size=True)
    if not weight_path or not os.path.exists(weight_path):
        print('[DINOv3] no pretrained weights at %r; using random init (load a checkpoint afterwards)' % (weight_path,))
        return m
    sd = torch.load(weight_path, map_location='cpu')
    new = {}
    for k, v in sd.items():
        if k.endswith('.periods') or k.endswith('.bias_mask') or k == 'mask_token':
            continue
        if k.startswith('projectors.') or k.startswith('local_cls_norm'):
            continue
        if k.endswith('qkv.bias'):
            continue
        k = k.replace('ls1.gamma', 'gamma_1').replace('ls2.gamma', 'gamma_2').replace('storage_tokens', 'reg_token')
        # DINOv3 ViT-H+ (and any SwiGLU variant) names its MLP w1/w2/w3; timm's SwiGLU uses fc1_g/fc1_x/fc2.
        # Mapping verified tensor-by-tensor against timm/vit_huge_plus_patch16_dinov3.lvd1689m (maxdiff = 0).
        k = k.replace('.mlp.w1.', '.mlp.fc1_g.').replace('.mlp.w2.', '.mlp.fc1_x.').replace('.mlp.w3.', '.mlp.fc2.')
        new[k] = v
    missing, unexpected = m.load_state_dict(new, strict=False)
    assert len(unexpected) == 0, 'unexpected keys: %s' % unexpected[:5]
    if missing:
        print('[DINOv3] missing (init) keys:', missing[:5], 'n=%d' % len(missing))
    return m


class DetailStem(nn.Module):
    """Lightweight trainable CNN on the input image -> high-frequency detail at 1/4, 1/8, 1/16."""

    def __init__(self, c2=96, c4=128, c8=192, c16=192):
        super().__init__()
        self.b2 = nn.Sequential(nn.Conv2d(3, c2, 3, 2, 1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.b4 = nn.Sequential(nn.Conv2d(c2, c4, 3, 2, 1), nn.BatchNorm2d(c4), nn.ReLU(inplace=True))
        self.b8 = nn.Sequential(nn.Conv2d(c4, c8, 3, 2, 1), nn.BatchNorm2d(c8), nn.ReLU(inplace=True))
        self.b16 = nn.Sequential(nn.Conv2d(c8, c16, 3, 2, 1), nn.BatchNorm2d(c16), nn.ReLU(inplace=True))

    def forward(self, x):
        f2 = self.b2(x)
        f4 = self.b4(f2)
        f8 = self.b8(f4)
        f16 = self.b16(f8)
        return f4, f8, f16


class ConvBNR(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.c = nn.Conv2d(cin, cout, k, padding=k // 2)
        self.b = nn.BatchNorm2d(cout)
        self.r = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.r(self.b(self.c(x)))


def _gabor_pair_bank(n_ori=6, n_scale=3, ksize=9,
                     sigmas=(1.8, 2.8, 4.0), freqs=(0.30, 0.20, 0.13)):
    """Fixed even/odd Gabor quadrature filter bank, ordered [orientation][scale].

    Justification (phase theory, Kovesi 1999): phase congruency is a *contrast-invariant* edge
    measure, unlike amplitude/gradient detectors that fail on low-contrast camouflage. Zero-DC and
    L1-normalised, hence scale/illumination agnostic.
    """
    import numpy as np
    c = (ksize - 1) / 2.0
    yy, xx = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    evens, odds = [], []
    for oi in range(n_ori):
        th = np.pi * oi / n_ori
        xr = xx * np.cos(th) + yy * np.sin(th)
        yr = -xx * np.sin(th) + yy * np.cos(th)
        for s in range(n_scale):
            env = np.exp(-(xr ** 2 + yr ** 2) / (2.0 * sigmas[s] ** 2))
            ev = env * np.cos(2 * np.pi * freqs[s] * xr)
            od = env * np.sin(2 * np.pi * freqs[s] * xr)
            ev -= ev.mean()
            od -= od.mean()
            ev /= (np.abs(ev).sum() + 1e-6)
            od /= (np.abs(od).sum() + 1e-6)
            evens.append(ev)
            odds.append(od)
    return (torch.from_numpy(np.stack(evens))[:, None],
            torch.from_numpy(np.stack(odds))[:, None])


class PhaseCongruency(nn.Module):
    """Feature-space phase congruency (PCB): a contrast-invariant boundary prior.

    Quadrature (even/odd) responses -> local energy E_theta over scales -> PC = sum_theta E_theta /
    sum_{theta,n} A_{theta,n}. The filter bank is fixed (physics-derived, parameter-free); only a
    1x1 channel reduction is learned, so the prior cannot be "learned away".
    """

    def __init__(self, C, red=8, n_ori=6, n_scale=3, ksize=9, noise=0.15):
        super().__init__()
        self.red = nn.Conv2d(C, red, 1)
        ev, od = _gabor_pair_bank(n_ori, n_scale, ksize)
        self.n_ori, self.n_scale, self.ksize, self.red_ch = n_ori, n_scale, ksize, red
        self.register_buffer('even', ev)
        self.register_buffer('odd', od)
        self.thr = nn.Parameter(torch.tensor(float(noise)))

    def forward(self, x):
        B, C, H, W = x.shape
        R = self.red_ch
        r = self.red(x)                                   # (B, R, H, W)
        ev = self.even.repeat(R, 1, 1, 1)                 # (R*M, 1, k, k)
        od = self.odd.repeat(R, 1, 1, 1)
        p = self.ksize // 2
        e = F.conv2d(r, ev, padding=p, groups=R).reshape(B, R, self.n_ori, self.n_scale, H, W)
        o = F.conv2d(r, od, padding=p, groups=R).reshape(B, R, self.n_ori, self.n_scale, H, W)
        A = torch.sqrt(e * e + o * o + 1e-6)
        E = torch.sqrt(e.sum(3) ** 2 + o.sum(3) ** 2 + 1e-6)
        pc = E.sum(2) / (A.sum((2, 3)) + 1e-6)
        t = self.thr.abs().clamp(max=0.9)
        pc = ((pc - t) / (1.0 - t + 1e-6)).clamp(0.0, 1.0)
        return pc.mean(1, keepdim=True)


class AnisotropicDiffusion(nn.Module):
    """Anisotropic feature diffusion (AFD): unrolled Perona-Malik nonlinear heat equation.

    Physics: dI/dt = div( g(||grad I||) grad I ), g = exp(-(||grad||/k)^2). Diffusion acts strongly
    in flat/background regions and is blocked across edges, so camouflage texture is suppressed
    while object boundaries stay sharp. Gradients are made scale-free by the per-channel spatial std.
    """

    def __init__(self, C, steps=4, k=0.5, dt=0.1):
        super().__init__()
        self.steps = steps
        self.k = nn.Parameter(torch.tensor(float(k)))
        self.dt = nn.Parameter(torch.tensor(float(dt)))

    def forward(self, f):
        k = self.k.abs() + 1e-3
        dt = self.dt.clamp(0.0, 0.24)
        s = f.std(dim=(2, 3), keepdim=True).detach() + 1e-6
        x = f
        for _ in range(self.steps):
            p = F.pad(x, (1, 1, 1, 1), mode='replicate')
            c = p[:, :, 1:-1, 1:-1]
            gE = (p[:, :, 2:, 1:-1] - c) / s
            gW = (p[:, :, :-2, 1:-1] - c) / s
            gS = (p[:, :, 1:-1, 2:] - c) / s
            gN = (p[:, :, 1:-1, :-2] - c) / s
            x = x + dt * ((torch.exp(-(gE / k) ** 2) * gE + torch.exp(-(gW / k) ** 2) * gW
                           + torch.exp(-(gS / k) ** 2) * gS + torch.exp(-(gN / k) ** 2) * gN) * s)
        return x


class OursNetV3(nn.Module):
    def __init__(self, dino_weight='/2022900015/COD/weights/dinov3_vitl16.pth',
                 timm_name='vit_large_patch16_dinov3', out_indices=(7, 15, 23),
                 use_sad=True, C=192, use_detail=True, use_pc=False, use_afd=False,
                 pc_red=8, afd_steps=4):
        super().__init__()
        self.backbone = load_dinov3_vit(timm_name, dino_weight)
        d = self.backbone.embed_dim
        self.out_indices = list(out_indices)
        self.use_sad, self.use_detail = use_sad, use_detail
        self.use_pc, self.use_afd = use_pc, use_afd
        if use_sad:
            self.sad = SAD(d)
        self.p3 = nn.Conv2d(d, C, 1)
        self.p2 = nn.Conv2d(d, C, 1)
        self.p1 = nn.Conv2d(d, C, 1)
        self.fuse = ConvBNR(C * 3, C)
        if use_afd:
            self.afd = AnisotropicDiffusion(C, steps=afd_steps)
        self.c4 = ConvBNR(C, C)
        self.c3 = ConvBNR(C, C)
        self.c2 = ConvBNR(C, C)
        self.c1 = ConvBNR(C, C)
        if use_pc:
            self.pc = PhaseCongruency(C, red=pc_red)
            self.pc_alpha = nn.Parameter(torch.zeros(1))
        if use_detail:
            self.detail = DetailStem(c2=96, c4=128, c8=192, c16=192)
            self.proj4 = nn.Conv2d(128, C, 1)
            self.proj8 = nn.Conv2d(192, C, 1)
            self.proj16 = nn.Conv2d(192, C, 1)
        self.side4 = nn.Conv2d(C, 1, 1)
        self.side3 = nn.Conv2d(C, 1, 1)
        self.side2 = nn.Conv2d(C, 1, 1)
        self.main = nn.Sequential(ConvBNR(C, 64), nn.Conv2d(64, 1, 1))
        self.edge = nn.Sequential(ConvBNR(C, 64), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        inter = self.backbone.forward_intermediates(x, indices=self.out_indices, norm=True)[1]
        f_low, f_mid, f_high = inter[0], inter[1], inter[2]
        # Base-model-level adaptation (v10): spectral whitening adapters inside the frozen encoder.
        # None for v3/v6/v7/v8/v9 -> behaviour unchanged.
        if getattr(self, 'adapters', None) is not None:
            f_low = self.adapters[0](f_low)
            f_mid = self.adapters[1](f_mid)
            f_high = self.adapters[2](f_high)
        if self.use_sad:
            f_high = self.sad(f_high)
        fused = self.fuse(torch.cat([self.p3(f_high), self.p2(f_mid), self.p1(f_low)], dim=1))
        if self.use_afd:
            fused = self.afd(fused)
        if self.use_detail:
            d4, d8, d16 = self.detail(x)
            if d16.shape[-2:] != fused.shape[-2:]:
                d16 = F.interpolate(d16, size=fused.shape[-2:], mode='bilinear', align_corners=False)
            fused = fused + self.proj16(d16)
        c4 = self.c4(fused)
        up3 = F.interpolate(c4, scale_factor=2, mode='bilinear', align_corners=False)
        if self.use_detail:
            if d8.shape[-2:] != up3.shape[-2:]:
                d8 = F.interpolate(d8, size=up3.shape[-2:], mode='bilinear', align_corners=False)
            up3 = up3 + self.proj8(d8)
        c3 = self.c3(up3)
        up2 = F.interpolate(c3, scale_factor=2, mode='bilinear', align_corners=False)
        if self.use_detail:
            if d4.shape[-2:] != up2.shape[-2:]:
                d4 = F.interpolate(d4, size=up2.shape[-2:], mode='bilinear', align_corners=False)
            up2 = up2 + self.proj4(d4)
        c2 = self.c2(up2)
        c1 = self.c1(F.interpolate(c2, scale_factor=2, mode='bilinear', align_corners=False))
        if self.use_pc:
            pc = self.pc(c1)
            c1 = c1 * (1.0 + self.pc_alpha * pc)
        return {'main': self.main(c1), 'edge': self.edge(c1),
                's2': self.side2(c2), 's3': self.side3(c3), 's4': self.side4(c4), 'feat': c1}
