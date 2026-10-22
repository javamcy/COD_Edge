# -*- coding: utf-8 -*-
"""Ours v0: PVTv2-B4 + SAD (spectral amplitude decoupling) + scale-consistency-ready decoder + OT boundary loss utilities.
Innovation story: camouflage is a scale-and-texture illusion; we break it with
 (1) SAD: adaptive amplitude-band gain (texture statistics) with phase preservation (object structure);
 (2) cross-scale prediction consistency (dual-resolution forward, loss-side);
 (3) OT-based soft assignment of ambiguous boundary pixels between fg/bg prototypes (loss-side).
"""
import sys, types, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')


def _load_pvtv2(py_path):
    mod = types.ModuleType('pvtv2_ext')
    sys.modules['pvtv2_ext'] = mod
    with open(py_path, 'r') as f:
        exec(compile(f.read(), py_path, 'exec'), mod.__dict__)
    return mod


class PVTv2Backbone(nn.Module):
    def __init__(self, code_path, weight_path, arch='b4'):
        super().__init__()
        m = _load_pvtv2(code_path)
        self.pvt = getattr(m, f'pvt_v2_{arch}')()
        sd = torch.load(weight_path, map_location='cpu')
        if 'state_dict' in sd:
            sd = sd['state_dict']
        sd = {k: v for k, v in sd.items() if not k.startswith('head.')}
        self.pvt.load_state_dict(sd, strict=False)

    def forward(self, x):
        B = x.shape[0]
        feats = []
        for i in range(4):
            pe = getattr(self.pvt, f'patch_embed{i + 1}')
            blk = getattr(self.pvt, f'block{i + 1}')
            nrm = getattr(self.pvt, f'norm{i + 1}')
            x, H, W = pe(x)
            for b in blk:
                x = b(x, H, W)
            x = nrm(x)
            feats.append(x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())
            if i != 3:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return feats


class SAD(nn.Module):
    """Spectral Amplitude Decoupling: radial-band learnable gain on amplitude (context-gated), phase kept intact."""

    def __init__(self, channels, bands=8, rmin=0.25, rmax=1.6):
        super().__init__()
        self.bands = bands
        self.mlp = nn.Sequential(nn.Linear(channels + bands, channels // 2), nn.ReLU(inplace=True),
                                 nn.Linear(channels // 2, bands))
        self.rmin, self.rmax = rmin, rmax
        self.fuse = nn.Sequential(nn.Conv2d(channels * 2, channels, 3, padding=1), nn.BatchNorm2d(channels),
                                  nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x.float(), norm='ortho')
        amp = xf.abs()
        ph = torch.angle(xf)
        # radial band ids over half-spectrum grid
        fy = torch.fft.fftfreq(H, device=x.device).abs().unsqueeze(1)  # H x 1
        fx = torch.fft.rfftfreq(W, device=x.device).abs().unsqueeze(0)  # 1 x W'
        r = torch.sqrt(fy ** 2 + fx ** 2) / (fy.max() + fx.max() + 1e-8)  # normalized 0..~0.7
        edges = torch.linspace(0, r.max() + 1e-6, self.bands + 1, device=x.device)
        band_id = torch.bucketize(r, edges[1:-1])  # H x W' in [0, bands-1]
        # per-band amplitude energy (texture statistics)
        amp_n = amp / (amp.mean(dim=(2, 3), keepdim=True) + 1e-6)
        stats = []
        for b in range(self.bands):
            m = (band_id == b)
            if m.sum() == 0:
                stats.append(amp.new_zeros(()))
            else:
                stats.append(amp_n[:, :, m].mean(dim=(1, 2)))
        ctx = torch.cat([x.mean(dim=(2, 3)).float(), torch.log1p(torch.stack(stats, dim=1))], dim=1)
        g = torch.sigmoid(self.mlp(ctx)) * (self.rmax - self.rmin) + self.rmin  # B x bands in [rmin, rmax]
        gain = g[:, band_id]  # B x H x W'
        amp2 = amp * gain.unsqueeze(1).expand(-1, C, -1, -1)
        xr = torch.fft.irfft2(amp2 * torch.exp(1j * ph), s=(H, W), dim=(2, 3), norm='ortho').to(x.dtype)
        out = x + self.fuse(torch.cat([x, xr], dim=1))
        return out


class DecBlock(nn.Module):
    def __init__(self, cin, cpv, cout):
        super().__init__()
        self.proj = nn.Conv2d(cin, cout, 1)
        self.conv = nn.Sequential(nn.Conv2d(cout + cpv, cout, 3, padding=1), nn.BatchNorm2d(cout),
                                  nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1),
                                  nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x_low, x_skip):
        x_low = self.proj(x_low)
        x_low = F.interpolate(x_low, size=x_skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x_low, x_skip], dim=1))


class OursNet(nn.Module):
    def __init__(self, pvt_code='/2022900015/COD/own_method/pvtv2.py',
                 pvt_weight='/2022900015/COD/weights/pvt_v2_b4.pth', arch='b4', use_sad=True):
        super().__init__()
        self.backbone = PVTv2Backbone(pvt_code, pvt_weight, arch)
        ch = [64, 128, 320, 512]
        self.use_sad = use_sad
        if use_sad:
            self.sad1 = SAD(ch[0])
            self.sad2 = SAD(ch[1])
        C = 128
        self.b4 = nn.Sequential(nn.Conv2d(ch[3], C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.d3 = DecBlock(C, ch[2], C)
        self.d2 = DecBlock(C, ch[1], C)
        self.d1 = DecBlock(C, ch[0], C)
        self.side4 = nn.Conv2d(C, 1, 1)
        self.side3 = nn.Conv2d(C, 1, 1)
        self.side2 = nn.Conv2d(C, 1, 1)
        self.main = nn.Sequential(nn.Conv2d(C, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 1, 1))
        self.edge = nn.Sequential(nn.Conv2d(C, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 1, 1))

    def forward(self, x):
        f1, f2, f3, f4 = self.backbone(x)
        if self.use_sad:
            f1 = self.sad1(f1)
            f2 = self.sad2(f2)
        c4 = self.b4(f4)
        c3 = self.d3(c4, f3)
        c2 = self.d2(c3, f2)
        c1 = self.d1(c2, f1)
        s4 = self.side4(c4)
        s3 = self.side3(c3)
        s2 = self.side2(c2)
        main = self.main(c1)
        edge = self.edge(c1)
        return {'main': main, 'edge': edge, 's2': s2, 's3': s3, 's4': s4,
                'feat': c1}  # feat: stride-4 map for OT loss


# ---------------- OT boundary soft assignment (parameter-free) ----------------
def ot_boundary_softlabel(feat, prob, tau_lo=0.15, tau_hi=0.85, eps=0.1, iters=5):
    """feat: B x C x h x w (stride-4), prob: B x 1 x h x w in [0,1] (sigmoid of main logits).
    Returns soft fg labels for ambiguous ring pixels (B x 1 x h x w, values in [0,1]; 0.5=neutral for non-ring)."""
    B, C, h, w = feat.shape
    out = torch.full((B, 1, h, w), 0.5, device=feat.device, dtype=feat.dtype)
    for b in range(B):
        p = prob[b, 0]
        ring = (p > tau_lo) & (p < tau_hi)
        n = int(ring.sum())
        if n < 8:
            continue
        fg = (p > tau_hi)
        bg = (p < tau_lo)
        if fg.sum() < 4 or bg.sum() < 4:
            continue
        fb = feat[b].reshape(C, -1).t()  # HW x C
        v_fg = F.normalize(fb[fg.reshape(-1)], dim=1).mean(dim=0)
        v_bg = F.normalize(fb[bg.reshape(-1)], dim=1).mean(dim=0)
        fr = F.normalize(fb[ring.reshape(-1)], dim=1)  # n x C
        cost = 1.0 - torch.stack([fr @ v_fg, fr @ v_bg], dim=1)  # n x 2
        K = torch.exp(-(cost - cost.min()) / eps)  # n x 2
        a = torch.full((n,), 1.0 / n, device=feat.device)
        tgt = torch.stack([(fg.sum() / (fg.sum() + bg.sum())).expand(n),
                           (bg.sum() / (fg.sum() + bg.sum())).expand(n)], dim=1) / n
        u = torch.ones(n, device=feat.device)
        v = torch.ones(2, device=feat.device)
        for _ in range(iters):
            v = tgt.sum(0) / (K.t() @ u + 1e-12)
            u = a / (K @ v + 1e-12)
        T = (u.unsqueeze(1) * K * v.unsqueeze(0))  # n x 2
        s = T[:, 0] / (T.sum(dim=1) + 1e-12)
        out[b, 0][ring] = s.to(out.dtype)
    return out
