# -*- coding: utf-8 -*-
"""Train Ours v0. Protocol mirrors baselines: train 4040 (COD10K-train+CAMO-train), val = CAMO test 250 @384 MAE -> best ckpt.
Losses: BCE+IoU (main + sides), edge BCE, OT soft-label (ring pixels), cross-scale consistency (dual forward).
"""
import argparse, os, time, math
import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd

from ournet import OursNet, ot_boundary_softlabel
from dataset_own import get_loader, get_val_loader

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--bl_mult', type=float, default=0.5)
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--trainsize', type=int, default=384)
parser.add_argument('--scale2', type=int, default=320)
parser.add_argument('--p_dual', type=float, default=0.3)
parser.add_argument('--w_edge', type=float, default=0.5)
parser.add_argument('--w_side', type=float, default=0.35)
parser.add_argument('--w_ot', type=float, default=0.5)
parser.add_argument('--w_sc', type=float, default=0.25)
parser.add_argument('--ot_start', type=int, default=15)
parser.add_argument('--val_every', type=int, default=2)
parser.add_argument('--workers', type=int, default=6)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--train_root', default='/2022900015/COD/datasets/train/')
parser.add_argument('--val_img', default='/2022900015/COD/datasets/camo_test/Imgs/')
parser.add_argument('--val_gt', default='/2022900015/COD/datasets/camo_test/GT/')
parser.add_argument('--pvt_code', default='/2022900015/COD/own_method/pvtv2.py')
parser.add_argument('--pvt_weight', default='/2022900015/COD/weights/pvt_v2_b4.pth')
parser.add_argument('--arch', default='pvt', choices=['pvt', 'dino', 'dino3', 'dino6', 'dino7', 'dino8', 'dino9', 'dino10', 'dino11', 'dino12', 'dino13', 'dino14', 'dino15', 'dino16'])
parser.add_argument('--dino_weight', default='/2022900015/COD/weights/dinov3_vitl16.pth')  # DINOv3 ViT-L/16 (our backbone); legacy fallback was dinov2_vitb14
parser.add_argument('--dino_h_weight', default='/2022900015/COD/weights/dinov3_vith16plus.pth')  # DINOv3 ViT-H+/16 (base-model upgrade, arch dino12)
parser.add_argument('--sam2_enc', type=int, default=0)   # arch dino14: run SAM2 at its native input size (>0)
parser.add_argument('--use_depth', type=int, default=1)  # arch dino15: second (geometric) frozen foundation
parser.add_argument('--da_size', type=int, default=448)  # arch dino15: DA-V2 internal input size (0 = follow input)
parser.add_argument('--use_sad', type=int, default=1)
parser.add_argument('--use_detail', type=int, default=1)
parser.add_argument('--use_pc', type=int, default=0)
parser.add_argument('--use_afd', type=int, default=0)
parser.add_argument('--use_fullres', type=int, default=1)
parser.add_argument('--use_refine', type=int, default=0)   # v7: image-guided full-res residual refinement
parser.add_argument('--use_swa', type=int, default=1)   # v10: spectral whitening adapters inside the encoder
parser.add_argument('--use_mslc', type=int, default=1)  # v11: multi-scale local-contrast adapters
parser.add_argument('--refine_only', type=int, default=0)   # stage-2: freeze everything but the refinement head
parser.add_argument('--load_strict', type=int, default=1)
parser.add_argument('--freeze_enc', type=int, default=0)
parser.add_argument('--unfreeze_blocks', type=int, default=0)  # unfreeze last N DINOv3 blocks (partial finetune)
parser.add_argument('--w_depth', type=float, default=0.0)   # depth-privileged boundary supervision
parser.add_argument('--save_path', default='/2022900015/COD/results/ours/')
parser.add_argument('--load', default=None)
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--best_mae', type=float, default=1.0)
parser.add_argument('--decay_end', type=int, default=0)   # >0 enables poly decay until this epoch
parser.add_argument('--decay_pow', type=float, default=0.9)
parser.add_argument('--seed', type=int, default=0)   # >0: fix python/numpy/torch RNG for multi-seed replication
opt = parser.parse_args()

if opt.seed > 0:
    import random as _random
    _random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    print('[SEED] fixed RNG seed=%d' % opt.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
torch.backends.cudnn.benchmark = True
os.makedirs(opt.save_path, exist_ok=True)

if opt.arch == 'dino':
    # Legacy v2 path (DINOv2 ViT-B/14), kept only for reproducing early runs. It hard-codes its own
    # DINOv2 checkpoint so that --dino_weight (DINOv3) stays aligned for the dino3+ architectures.
    from ournet_v2 import OursNetV2
    model = OursNetV2('/2022900015/COD/weights/dinov2_vitb14_pretrain.pth').cuda()
elif opt.arch == 'dino3':
    from ournet_v3 import OursNetV3
    model = OursNetV3(opt.dino_weight, use_sad=bool(opt.use_sad), use_detail=bool(opt.use_detail),
                      use_pc=bool(opt.use_pc), use_afd=bool(opt.use_afd)).cuda()
elif opt.arch == 'dino6':
    from ournet_v6 import OursNetV6
    model = OursNetV6(opt.dino_weight, use_sad=bool(opt.use_sad), use_detail=bool(opt.use_detail),
                      use_pc=bool(opt.use_pc), use_afd=bool(opt.use_afd),
                      use_fullres=bool(opt.use_fullres)).cuda()
elif opt.arch == 'dino7':
    from ournet_v7 import OursNetV7
    model = OursNetV7(opt.dino_weight, use_sad=bool(opt.use_sad), use_detail=bool(opt.use_detail),
                      use_pc=bool(opt.use_pc), use_afd=bool(opt.use_afd),
                      use_refine=int(opt.use_refine)).cuda()
elif opt.arch == 'dino8':
    from ournet_v8 import OursNetV8
    model = OursNetV8(opt.dino_weight, use_sad=bool(opt.use_sad), use_detail=bool(opt.use_detail),
                      use_pc=bool(opt.use_pc), use_afd=bool(opt.use_afd),
                      use_refine=int(opt.use_refine)).cuda()
elif opt.arch == 'dino9':
    from ournet_v9 import OursNetV9
    model = OursNetV9(opt.dino_weight, use_sad=bool(opt.use_sad), use_detail=bool(opt.use_detail),
                      use_pc=bool(opt.use_pc), use_afd=bool(opt.use_afd),
                      use_refine=int(opt.use_refine)).cuda()
elif opt.arch == 'dino10':
    from ournet_v10 import OursNetV10
    model = OursNetV10(opt.dino_weight, use_sad=bool(opt.use_sad), use_detail=bool(opt.use_detail),
                       use_pc=bool(opt.use_pc), use_afd=bool(opt.use_afd),
                       use_refine=int(opt.use_refine), use_swa=bool(opt.use_swa)).cuda()
elif opt.arch == 'dino11':
    from ournet_v11 import OursNetV11
    model = OursNetV11(opt.dino_weight, use_sad=bool(opt.use_sad), use_detail=bool(opt.use_detail),
                       use_pc=bool(opt.use_pc), use_afd=bool(opt.use_afd),
                       use_refine=int(opt.use_refine), use_mslc=bool(opt.use_mslc)).cuda()
elif opt.arch == 'dino12':
    # Base-model upgrade: DINOv3 ViT-H+/16 (depth 32, embed 1280) replaces ViT-L/16 (depth 24, embed 1024).
    # Same decoder/heads; only the foundation encoder + its feature-tap indices change.
    from ournet_v3 import OursNetV3
    model = OursNetV3(opt.dino_h_weight, timm_name='vit_huge_plus_patch16_dinov3',
                      out_indices=(9, 21, 31), use_sad=bool(opt.use_sad), use_detail=bool(opt.use_detail),
                      use_pc=bool(opt.use_pc), use_afd=bool(opt.use_afd)).cuda()
elif opt.arch == 'dino13':
    # dino12 backbone (DINOv3 ViT-H+/16) + CGR refinement head (stage-2 on top of a dino12 base ckpt).
    from ournet_v7 import OursNetV7
    model = OursNetV7(opt.dino_h_weight, timm_name='vit_huge_plus_patch16_dinov3',
                      out_indices=(9, 21, 31), use_sad=bool(opt.use_sad), use_detail=bool(opt.use_detail),
                      use_pc=bool(opt.use_pc), use_afd=bool(opt.use_afd),
                      use_refine=int(opt.use_refine)).cuda()
elif opt.arch == 'dino14':
    # Base-model swap: SAM2.1 Hiera-Large image encoder (segmentation-pretrained, real 4/8/16
    # feature pyramid) replaces the DINOv3 ViT; decoder/SAD/DetailStem/CGR remain identical.
    from ournet_sam2 import OursNetSAM2
    model = OursNetSAM2(use_sad=bool(opt.use_sad), use_detail=bool(opt.use_detail),
                        use_afd=bool(opt.use_afd), use_refine=int(opt.use_refine),
                        enc=int(opt.sam2_enc)).cuda()
elif opt.arch == 'dino15':
    # Second foundation model: frozen Depth-Anything-V2 ViT-L (geometric, RGB-only) is fused into the
    # decoder alongside the semantic DINOv3 ViT-L; every other component stays identical.
    from ournet_v13 import OursNetV13
    model = OursNetV13(opt.dino_weight, use_sad=bool(opt.use_sad), use_detail=bool(opt.use_detail),
                       use_pc=bool(opt.use_pc), use_afd=bool(opt.use_afd),
                       use_refine=int(opt.use_refine), use_depth=bool(opt.use_depth),
                       da_size=int(opt.da_size)).cuda()
elif opt.arch == 'dino16':
    # v14: depth-discontinuity-GATED boundary refinement. The semantic encoder stays depth-free (the
    # v13 dual-encoder variant measured as neutral); the frozen DA-V2 discontinuity is injected only
    # into the boundary-band refinement head + its gate -- i.e. exactly where the RGB cue is blind.
    from ournet_v13 import OursNetV14
    model = OursNetV14(opt.dino_weight, use_sad=bool(opt.use_sad), use_detail=bool(opt.use_detail),
                       use_pc=bool(opt.use_pc), use_afd=bool(opt.use_afd),
                       da_size=int(opt.da_size)).cuda()
else:
    model = OursNet(opt.pvt_code, opt.pvt_weight).cuda()
if opt.load and os.path.exists(opt.load):
    sd = torch.load(opt.load, map_location='cpu')
    model.load_state_dict(sd, strict=bool(opt.load_strict))
    print('RESUMED from', opt.load, 'strict=%d' % opt.load_strict)

bl_params, other_params = [], []
if opt.freeze_enc:
    for n, p in model.named_parameters():
        if n.startswith('backbone'):
            p.requires_grad = False
if opt.unfreeze_blocks > 0:
    import re as _re
    _depth = 12
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'blocks'):
        _depth = len(model.backbone.blocks)
    _keep = set(range(_depth - opt.unfreeze_blocks, _depth))
    _n_unfrozen = 0
    for n, p in model.named_parameters():
        m = _re.match(r'backbone\.blocks\.(\d+)\.', n)
        if (m and int(m.group(1)) in _keep) or n.startswith('backbone.norm'):
            p.requires_grad = True
            _n_unfrozen += 1
    print('[UNFREEZE] last %d blocks + norm, %d tensors trainable' % (opt.unfreeze_blocks, _n_unfrozen))
if opt.refine_only:
    _n = 0
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith('refine') or n.startswith('gate') or n.startswith('adapters')
        _n += int(p.requires_grad)
    print('[REFINE-ONLY] %d tensors trainable (refinement head + adapters)' % _n)
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (bl_params if n.startswith('backbone') else other_params).append(p)
groups = []
if bl_params:
    groups.append({'params': bl_params, 'lr': opt.lr * opt.bl_mult})
groups.append({'params': other_params, 'lr': opt.lr})
optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler()
base_lrs = [g['lr'] for g in optimizer.param_groups]

train_loader = get_loader(opt.train_root, opt.batchsize, opt.trainsize, opt.workers,
                          with_depth=(opt.w_depth > 0))
val_loader = get_val_loader(opt.val_img, opt.val_gt, opt.trainsize)
total_step = len(train_loader)
print('epochs %d, steps/epoch %d' % (opt.epoch, total_step))

best_mae = opt.best_mae
start_epoch = opt.start_epoch


def bce_iou(pred, gt, eps=1.0):
    bce = F.binary_cross_entropy_with_logits(pred, gt)
    p = torch.sigmoid(pred)
    inter = (p * gt).sum(dim=(2, 3))
    union = p.sum(dim=(2, 3)) + gt.sum(dim=(2, 3)) - inter
    iou = 1 - (inter + eps) / (union + eps)
    return bce + iou.mean()


def depth_grad(depth):
    """Sobel gradient magnitude of a normalized depth map -> object boundary prior in [0,1]."""
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                      dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
    gx = F.conv2d(depth, kx, padding=1)
    gy = F.conv2d(depth, kx.transpose(2, 3), padding=1)
    g = (gx * gx + gy * gy).sqrt()
    return g / (g.amax(dim=(2, 3), keepdim=True) + 1e-6)


def run_batch(scale):
    if opt.w_depth > 0:
        imgs, gts, edges, depth, _ = batch_main
    else:
        imgs, gts, edges, _ = batch_main
        depth = None
    imgs = imgs.cuda(non_blocking=True)
    gts = gts.cuda(non_blocking=True)
    edges = edges.cuda(non_blocking=True)
    if scale != opt.trainsize:
        imgs = F.interpolate(imgs, size=(scale, scale), mode='bilinear', align_corners=False)
        gts_s = F.interpolate(gts, size=(scale, scale), mode='nearest')
        edges_s = F.interpolate(edges, size=(scale, scale), mode='nearest')
    else:
        gts_s, edges_s = gts, edges
    out = model(imgs)
    if 'main_full' in out:
        main = out['main_full']
    else:
        main = F.interpolate(out['main'], size=gts_s.shape[-2:], mode='bilinear', align_corners=False)
    loss = bce_iou(main, gts_s)
    for k, w in (('s2', opt.w_side), ('s3', opt.w_side), ('s4', opt.w_side)):
        s = F.interpolate(out[k], size=gts_s.shape[-2:], mode='bilinear', align_corners=False)
        loss = loss + w * bce_iou(s, gts_s)
    e = F.interpolate(out['edge'], size=edges_s.shape[-2:], mode='bilinear', align_corners=False)
    loss = loss + opt.w_edge * F.binary_cross_entropy_with_logits(e, edges_s)
    if opt.w_depth > 0 and depth is not None:
        dsr = depth.cuda(non_blocking=True).float()
        if dsr.shape[-1] != gts_s.shape[-1]:
            dsr = F.interpolate(dsr, size=gts_s.shape[-2:], mode='bilinear', align_corners=False)
        G = depth_grad(dsr)
        Gb = (G > 0.1).float()        # depth-discontinuity zone = boundary prior
        lo = F.binary_cross_entropy_with_logits(main, gts_s, reduction='none')
        loss = loss + opt.w_depth * (lo * Gb).sum() / (Gb.sum() + 1.0)
    return out, main, loss, imgs


log = open(os.path.join(opt.save_path, 'train.log'), 'a')
print(('%s' % time.strftime('%Y-%m-%d %H:%M:%S')), 'start epoch', start_epoch, 'best_mae %.4f' % best_mae)
log.write('start epoch %d best %.6f\n' % (start_epoch, best_mae))
log.flush()

for epoch in range(start_epoch, opt.epoch + 1):
    if opt.decay_end > 0:
        p = min(0.95, max(0.0, (epoch - start_epoch) / max(1, opt.decay_end - start_epoch)))
        f = (1.0 - p) ** opt.decay_pow
        for g, bl in zip(optimizer.param_groups, base_lrs):
            g['lr'] = bl * f
        print('[LR] epoch %d lr %.3e' % (epoch, optimizer.param_groups[-1]['lr']))
    model.train()
    if opt.refine_only:
        # keep the frozen base in eval mode (BN must use its learned running stats, not batch stats)
        for n, m in model.named_modules():
            if not (n.startswith('refine') or n.startswith('gate')):
                m.training = False
    t0 = time.time()
    loss_sum, nstep = 0.0, 0
    for it, batch_main in enumerate(train_loader, 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            out, main, loss, _ = run_batch(opt.trainsize)
            # OT boundary soft-label on ring pixels (parameter-free, uses stride-4 feat)
            if epoch >= opt.ot_start and opt.w_ot > 0:
                prob = torch.sigmoid(out['main'].detach())
                feat = out['feat'].float()
                soft = ot_boundary_softlabel(feat, prob.float())
                m96 = out['main']
                if soft.shape[-1] != m96.shape[-1]:
                    soft = F.interpolate(soft, size=m96.shape[-2:], mode='bilinear', align_corners=False)
                ring_m = ((soft > 0.5 + 1e-4) | (soft < 0.5 - 1e-4)).float()
                if ring_m.sum() > 16:
                    lo = F.binary_cross_entropy_with_logits(m96, soft, reduction='none')
                    loss = loss + opt.w_ot * (lo * ring_m).sum() / ring_m.sum().clamp(min=1)
            # cross-scale consistency
            sc_loss = None
            if opt.p_dual > 0 and np.random.rand() < opt.p_dual:
                out2, main2, loss2, _ = run_batch(opt.scale2)
                sc_loss = F.mse_loss(torch.sigmoid(main2), torch.sigmoid(
                    F.interpolate(main, size=main2.shape[-2:], mode='bilinear', align_corners=False)))
                loss = loss + opt.w_sc * sc_loss + loss2
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.detach())
        nstep += 1
        if it % 40 == 0:
            print('E%03d %04d/%04d loss %.4f (%.1fs)' % (epoch, it, total_step, loss_sum / nstep, time.time() - t0))
    print('[Train Info]: Epoch %03d Loss_AVG %.4f time %.1fs' % (epoch, loss_sum / max(nstep, 1), time.time() - t0))
    log.write('[Train Info]: Epoch %03d Loss_AVG %.4f time %.1fs\n' % (epoch, loss_sum / max(nstep, 1), time.time() - t0))
    log.flush()

    # val on CAMO every val_every epochs
    if epoch % opt.val_every == 0 or epoch == opt.epoch:
        model.eval()
        mae_sum, n = 0.0, 0
        with torch.no_grad():
            for vi, (vimg, vgt, _) in enumerate(val_loader):
                vimg, vgt = vimg.cuda(), vgt.cuda()
                with torch.cuda.amp.autocast():
                    vo = model(vimg)
                vm = vo['main_full'] if 'main_full' in vo else vo['main']
                vp = torch.sigmoid(F.interpolate(vm, size=vgt.shape[-2:], mode='bilinear', align_corners=False)).float()
                mae_sum += float((vp - vgt).abs().mean()) * vimg.size(0)
                n += vimg.size(0)
        vmae = mae_sum / max(n, 1)
        print('[Val Info]:Epoch %d MAE %.6f best %.6f' % (epoch, vmae, best_mae))
        log.write('[Val Info]:Epoch %d MAE %.6f best %.6f\n' % (epoch, vmae, best_mae))
        log.flush()
        if vmae < best_mae:
            best_mae = vmae
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'Net_epoch_best.pth'))
            print('saved best @%d' % epoch)
            log.write('saved best @%d\n' % epoch)
    if epoch % 25 == 0:
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'Net_epoch_%d.pth' % epoch))
    torch.save(model.state_dict(), os.path.join(opt.save_path, 'Net_epoch_last.pth'))

print('DONE best MAE %.6f' % best_mae)
log.write('DONE best MAE %.6f\n' % best_mae)
log.close()
