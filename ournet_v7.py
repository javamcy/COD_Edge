# -*- coding: utf-8 -*-
"""Ours v7: v3/v6 + CGR (Camouflage-Guided Refinement).

Motivation (from our own error analysis): the DINOv3 ViT encoder emits *all* levels at stride 16,
so the decoder has NO genuine high-resolution skip connection; the coarse 1/2-resolution mask is
upsampled bilinearly, which blurs exactly the boundary band that carries ~50-57% of the MAE.

CGR adds a trainable, image-guided full-resolution residual refinement head on top of the coarse
mask. It mirrors the classical guided-filter / joint-bilateral upsampling principle from image
processing (edge-preserving upsampling guided by a high-resolution image), made learnable and
conditioned on an illumination-invariant structure map. The final conv is zero-initialised, so the
model starts exactly at the baseline and the head can only add a correction (guaranteed no harm),
which makes its effect cleanly ablative (use_refine on/off).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ournet_v3 import OursNetV3, ConvBNR


class SpectralBandGate(nn.Module):
    """Learnable spectral boundary refinement (LSBG).

    Motivation: our fixed phase-consistency gate (PCG) failed because its response had a
    near-constant mean (~0.75) and was therefore *not discriminative*. We instead learn a
    bank of K radial band-pass filters directly in the Fourier domain (centre frequency and
    bandwidth are trainable parameters) and gate each band with a *spatially varying*,
    boundary-band-conditioned attention map. This makes the spectral filtering both learnable
    and boundary-aware, targeting exactly the boundary band that carries ~50-57% of the MAE.

    The output conv is zero-initialised, so the module starts as a no-op on top of the existing
    band-gated CGR head (guaranteed no harm) and is cleanly ablative.
    """

    def __init__(self, K=6, ch=16, n_prior=4):
        super().__init__()
        self.K = K
        init_ctr = torch.logspace(math.log10(0.03), math.log10(0.45), K)
        self.ctr = nn.Parameter(init_ctr)                       # (K,) centre freq (cycles/px)
        self.log_bw = nn.Parameter(torch.full((K,), math.log(0.12)))
        # gate: [prob, gmag, n_prior boundary rings] -> K per-band, per-pixel gates
        self.gn = nn.Sequential(
            nn.Conv2d(2 + n_prior, ch, 3, 1, 1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, K, 3, 1, 1),
        )
        # band fusion -> residual mask
        self.refine_sp = nn.Sequential(
            nn.Conv2d(K, ch, 3, 1, 1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 3, 1, 1),
        )
        nn.init.zeros_(self.refine_sp[-1].weight)
        nn.init.zeros_(self.refine_sp[-1].bias)

    def _masks(self, H, W, device):
        # Recomputed every forward (cheap: K gaussians over H x W//2) so that the centre
        # frequencies / bandwidths actually receive gradients (they are the learnable part).
        fy = torch.fft.fftfreq(H, device=device).view(H, 1)
        fx = torch.fft.rfftfreq(W, device=device).view(1, -1)
        r = torch.sqrt(fy * fy + fx * fx)                   # (H, W//2+1)
        c = torch.clamp(self.ctr, 1e-3, 0.6).view(1, 1, -1)
        bw = torch.exp(self.log_bw).clamp(1e-3, 0.5).view(1, 1, -1)
        m = torch.exp(-0.5 * ((r.unsqueeze(-1) - c) / bw) ** 2)   # (H, W2, K)
        m = m / (m.amax(dim=(0, 1), keepdim=True) + 1e-6)
        return m.permute(2, 0, 1)                           # (K, H, W2)

    def forward(self, gray, prob, gmag, rings):
        # gray/prob/gmag: (B,1,H,W); rings: (B,n_prior,H,W)
        H, W = gray.shape[-2:]
        Fg = torch.fft.rfft2(gray, norm='ortho')                # (B,1,H,W2)
        m = self._masks(H, W, gray.device)                      # (H,W2,K)
        bands = torch.fft.irfft2(Fg * m.unsqueeze(0), s=(H, W), norm='ortho')   # (B,K,H,W)
        g = torch.sigmoid(self.gn(torch.cat([prob, gmag, rings], 1)))           # (B,K,H,W)
        return self.refine_sp(bands * g)                        # (B,1,H,W)


def _sobel_mag(gray):
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                      dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, kx.transpose(2, 3), padding=1)
    g = (gx * gx + gy * gy).sqrt()
    return g / (g.amax(dim=(2, 3), keepdim=True) + 1e-6)


class OursNetV7(OursNetV3):
    def __init__(self, dino_weight='/2022900015/COD/weights/dinov3_vitl16.pth',
                 use_sad=True, use_detail=True, use_pc=False, use_afd=False,
                 use_refine=True, rf_ch=24, band_k=6,
                 timm_name='vit_large_patch16_dinov3', out_indices=(7, 15, 23)):
        super().__init__(dino_weight, timm_name=timm_name, out_indices=out_indices,
                         use_sad=use_sad, use_detail=use_detail,
                         use_pc=use_pc, use_afd=use_afd)
        # use_refine: 0=off, 1=plain CGR, 2=band-gated CGR (residual restricted to the
        # error-dominant boundary band so the interior/exterior stay exactly the base output)
        #             3=band-gated CGR with a *learnable* boundary attention and *adaptive*
        #               bandwidth (multi-scale rings combined by a learned gate instead of a
        #               fixed k): the gate decides per-pixel how much correction to trust and
        #               how wide the boundary band should be.
        #             6=band-gated CGR + LSBG (learnable spectral band-pass refinement on top;
        #               zero-init output so it starts exactly at the band-gate baseline).
        _r = int(use_refine)
        self.gate_mode = 1 if _r == 2 else (2 if _r == 3 else (3 if _r >= 6 else 0))
        self.band_k = band_k
        self.use_refine = (_r > 0)
        self.gate = None
        if self.use_refine:
            self.refine = nn.Sequential(
                nn.Conv2d(6, rf_ch, 3, 1, 1), nn.BatchNorm2d(rf_ch), nn.ReLU(inplace=True),
                nn.Conv2d(rf_ch, rf_ch, 3, 1, 1), nn.BatchNorm2d(rf_ch), nn.ReLU(inplace=True),
                nn.Conv2d(rf_ch, 1, 3, 1, 1),
            )
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)
            if self.gate_mode == 2:
                # in: prob, gmag, ring(k=2), ring(k=4), ring(k=8)  ->  soft attention in (0,1)
                self.gate = nn.Sequential(
                    nn.Conv2d(5, rf_ch, 3, 1, 1), nn.BatchNorm2d(rf_ch), nn.ReLU(inplace=True),
                    nn.Conv2d(rf_ch, 1, 3, 1, 1),
                )
            elif self.gate_mode == 3:
                # LSBG: learnable multi-band Fourier gate, boundary-band conditioned.
                self.gate = SpectralBandGate(K=max(band_k, 4), ch=rf_ch, n_prior=4)

    @staticmethod
    def _ring(prob, k):
        kw = 2 * k + 1
        m = (prob > 0.5).float()
        dil = F.max_pool2d(m, kw, 1, k)
        ero = -F.max_pool2d(-m, kw, 1, k)
        return (dil - ero).clamp(0, 1)

    def forward(self, x):
        return self._apply_cgr(x, super().forward(x))

    def _apply_cgr(self, x, out):
        if not self.use_refine:
            return out
        H, W = x.shape[-2:]
        base = F.interpolate(out['main'], size=(H, W), mode='bilinear', align_corners=False)
        prob = torch.sigmoid(base).float()
        gray = x.mean(1, keepdim=True).float()   # fp32 for the Fourier branch under autocast
        gmag = _sobel_mag(gray)
        inp = torch.cat([prob, x, gray, gmag], 1)
        res = self.refine(inp)
        if self.gate_mode == 1:
            with torch.no_grad():
                band = F.avg_pool2d(self._ring(prob, self.band_k), 5, 1, 2).clamp(0, 1)
            res = res * band
        elif self.gate_mode == 2:
            with torch.no_grad():
                rings = [self._ring(prob, k) for k in (2, 4, 8)]
            g = torch.sigmoid(self.gate(torch.cat([prob, gmag] + rings, 1)))
            res = res * g
        elif self.gate_mode == 3:
            with torch.no_grad():
                band = F.avg_pool2d(self._ring(prob, self.band_k), 5, 1, 2).clamp(0, 1)
                rings = torch.cat([self._ring(prob, k) for k in (2, 4, 6, 8)], 1)
            # band-gated CGR  +  learnable spectral boundary refinement (zero-init -> no harm)
            res = res * band + self.gate(gray, prob, gmag, rings)
        out['main_full'] = base + res
        return out
