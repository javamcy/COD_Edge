# 边界感知的伪装目标检测 / Boundary-Aware Camouflaged Object Detection

本仓库提供论文 **“A Boundary-Aware Detection Algorithm for Camouflaged Object Detection”** 中**本文模型**的最小复现材料：模型代码、训练/推理脚本、统一评测器，以及最终检查点。仓库**不包含**任何基线方法的代码或权重。

**最终检查点（约 1.31 GB）不放在 Git 历史中，请从 Releases 下载：**
<https://github.com/javamcy/COD_Edge/releases/download/v1.0.0/v8uCGR_best.pth>

This repository provides the **minimal reproduction package for the proposed model only** of the paper *“A Boundary-Aware Detection Algorithm for Camouflaged Object Detection”*: model code, training/inference scripts, the unified evaluator, and the final checkpoint. No baseline code or weights are included.

**The final checkpoint (~1.31 GB) is not stored in the Git history; download it from Releases:**
<https://github.com/javamcy/COD_Edge/releases/download/v1.0.0/v8uCGR_best.pth>

---

## 中文

### 1. 方法概述

模型以 **DINOv3 ViT-L/16** 为骨干，除最后 **8/24** 个 transformer 块外保持冻结，并组合四个模块：

- **SAD**：对最深特征图的幅度谱做径向频带重加权，保持相位不变；
- **DetailStem**：浅层卷积支路，补回 patch-16 分词丢失的高频细节；
- **AFD**：解码器中的各向异性特征扩散（边缘停止平滑）；
- **CGR**：第二阶段边界带门控精修头，零初始化，只修正轮廓带。

推理**不使用深度图**；深度仅在论文的离线误差分析中作为诊断工具出现。

### 2. 目录结构

```
github/
├── README.md
├── requirements.txt
├── train.py                  # 训练（stage 1 / stage 2 共用）
├── test.py                   # 推理 + 评测
├── ournet.py                 # 基础模块（SAD、解码块、OT 软标签）
├── ournet_v3.py              # DINOv3 + SAD + DetailStem + AFD
├── ournet_v7.py              # 在 v3 上加 CGR 精修头（最终模型）
├── dataset_own.py            # 数据集与数据增强
├── eval_local.py             # 统一评测器入口
├── evaltools/metrics.py      # Sα / Eϕ / Fωβ / MAE（来自 CamoFormer 评测工具）
├── weights/
│   └── v8uCGR_best.pth       # 本文最终模型（约 1.31 GB，从 Releases 下载）
└── scripts/
    ├── train_stage1.sh
    ├── train_stage2.sh
    └── test.sh
```

### 3. 环境

- Python 3.11
- PyTorch 2.6.0 + CUDA 12.4（其他 2.x 版本一般也可）
- `timm >= 1.0`（DINOv3 通过 timm 构建）
- 其余见 `requirements.txt`

```bash
pip install -r requirements.txt
```

**显存说明**：stage 1（448、batch 8）在原论文中用 A100 80 GB 训练；在 16 GB 笔记本上 batch 8 的峰值约 12.2 GB（未计桌面占用），请勿在显存接近占满时运行。若显存不足，可减小 `--batchsize`，但这会改变 BatchNorm 统计，结果不再与论文严格可比。stage 2（768、batch 8、refine-only）峰值约 6.1 GB。

### 4. 数据准备

训练使用 **CAMO train + COD10K train 的并集（4040 张）**，测试使用 CAMO（250）、COD10K（2026）、NC4K（4121）。目录约定：

```
data/
├── train/
│   ├── Imgs/     # 训练图像
│   ├── GT/       # 二值掩膜
│   └── Edge/     # 边缘图
├── camo_test/
│   ├── Imgs/
│   └── GT/
├── cod10k_test/
│   ├── Imgs/
│   └── GT/
└── nc4k/
    ├── Imgs/
    └── GT/
```

图像与 GT 按**同名 stem** 对应（GT 为 `.png`）。数据集请从各自官方页面下载；本仓库不重新分发数据。

### 5. 模型权重

| 文件 | 说明 |
|---|---|
| `weights/v8uCGR_best.pth` | 最终模型（stage 1 + stage 2），可直接用于推理。从 [Releases](https://github.com/javamcy/COD_Edge/releases/download/v1.0.0/v8uCGR_best.pth) 下载后放到该路径 |
| `weights/dinov3_vitl16.pth` | **仅训练需要**。DINOv3 ViT-L/16 官方权重，请从 Meta/timm 官方渠道下载后放到该路径 |

推理时 `--dino3_weight ""` 即可：检查点已包含完整骨干权重，无需再加载 DINOv3 预训练文件。

### 6. 复现步骤

**Stage 1（50 epoch，448，batch 8，解冻最后 8 块）**

```bash
bash scripts/train_stage1.sh
# 输出：results/stage1/Net_epoch_best.pth
```

**Stage 2（12 epoch，768，batch 8，只训练精修头）**

```bash
bash scripts/train_stage2.sh
# 输出：results/stage2/Net_epoch_best.pth
```

**推理 + 评测（论文主设置 768）**

```bash
bash scripts/test.sh
# 预测：results/pred/{CAMO,COD10K,NC4K}/
# 指标：results/eval_metrics.json（同时打印到终端）
```

**896 同输入设置**：把 `scripts/test.sh` 中的 `--trainsize 768` 改为 `896`；使用**同一个**检查点，不重训练。

### 7. 预期结果

论文主设置（768，单次前向，无 TTA）：

| 数据集 | Sα ↑ | Eϕ ↑ | Fωβ ↑ | MAE ↓ |
|---|---|---|---|---|
| CAMO (250) | 0.9236 | 0.9700 | 0.9084 | 0.0237 |
| COD10K (2026) | 0.9308 | 0.9754 | 0.8959 | 0.0121 |
| NC4K (4121) | 0.9281 | 0.9686 | 0.9096 | 0.0200 |

> **复算说明**：用本包自带的 `eval_local.py` 对同一检查点复算，CAMO 的 Sα 为 **0.9249**，比上表（论文主表）的 0.9236 高 0.0013；这是早期远程评测文件与本地评测器在 CAMO Sα 上的已知差异，Eϕ / Fωβ / MAE 与论文四位一致。论文的 896 表与消融均使用本地评测器。

896 同输入设置（同一检查点）：

| 数据集 | Sα ↑ | Eϕ ↑ | Fωβ ↑ | MAE ↓ |
|---|---|---|---|---|
| CAMO | 0.9257 | 0.9704 | 0.9105 | 0.0234 |
| COD10K | 0.9325 | 0.9757 | 0.8994 | 0.0120 |
| NC4K | 0.9282 | 0.9689 | 0.9104 | 0.0202 |

复现时的小数末位可能因 GPU/驱动/库版本产生 ±0.0001–0.0003 的差异。

### 8. 重要说明

- **统一评测器**：所有指标由 `eval_local.py` + `evaltools/metrics.py` 计算；真值在 128 处二值化，预测按逐图最小–最大值归一化，再逐图平均。这与论文的评测约定一致。
- **单种子**：论文的主表与消融均为单种子结果；论文另用 3 个种子重跑 CODiff 以标定噪声（Sα 极差 0.0008）。本仓库的检查点是论文所用种子。
- **机器差异**：训练在 A100 80 GB 上完成；896 对比与开销测量在 RTX 3080 Laptop 16 GB 上完成。精度与硬件无关，但耗时只在同一台机器内可比。
- **评测器来源**：`evaltools/metrics.py` 取自 CamoFormer 公开评测工具（其本身基于 `lartpang` 的实现），仅用于保证与论文同一口径。

### 9. 引用

若本仓库对你有帮助，请引用论文（bib 待补）：

```bibtex
@article{anonymous2026boundary,
  title   = {A Boundary-Aware Detection Algorithm for Camouflaged Object Detection},
  author  = {Anonymous},
  journal = {Journal of King Saud University Computer and Information Sciences},
  year    = {2026}
}
```

### 10. 许可

代码仅用于学术研究。`evaltools/` 遵循其原始开源许可；DINOv3 权重遵循 Meta 的许可；数据集遵循各自官方许可。

---

## English

### 1. Method

The model uses a **DINOv3 ViT-L/16** backbone, frozen except for the last **8 of 24** transformer blocks, and combines four modules:

- **SAD**: radial-band re-weighting of the amplitude spectrum of the deepest feature map, with the phase preserved;
- **DetailStem**: a shallow convolutional stem that restores the high-frequency detail removed by patch-16 tokenisation;
- **AFD**: anisotropic feature diffusion (edge-stopping smoothing) in the decoder;
- **CGR**: a zero-initialised, band-gated refinement head in stage 2 that edits only the contour band.

Inference is **RGB-only**; depth is used only as an offline diagnostic in the paper's error analysis.

### 2. Layout

```
github/
├── README.md
├── requirements.txt
├── train.py                  # training (stage 1 / stage 2)
├── test.py                   # inference + evaluation
├── ournet.py                 # basic modules (SAD, decoder blocks, OT soft labels)
├── ournet_v3.py              # DINOv3 + SAD + DetailStem + AFD
├── ournet_v7.py              # adds the CGR head (final model)
├── dataset_own.py            # datasets and augmentation
├── eval_local.py             # unified evaluator entry point
├── evaltools/metrics.py      # Sα / Eϕ / Fωβ / MAE (CamoFormer evaluation tools)
├── weights/
│   └── v8uCGR_best.pth       # final model (~1.31 GB, from Releases)
└── scripts/
    ├── train_stage1.sh
    ├── train_stage2.sh
    └── test.sh
```

### 3. Environment

- Python 3.11
- PyTorch 2.6.0 + CUDA 12.4 (other 2.x versions should work)
- `timm >= 1.0` (DINOv3 is built through timm)
- See `requirements.txt`

```bash
pip install -r requirements.txt
```

**Memory note.** Stage 1 (448, batch 8) was trained on an A100 80 GB. On a 16 GB laptop the peak is about 12.2 GB at batch 8 (excluding the desktop), so do not run it with the GPU nearly full. Reducing `--batchsize` changes the BatchNorm statistics and makes the results not strictly comparable. Stage 2 (768, batch 8, refine-only) peaks at about 6.1 GB.

### 4. Data

Training uses the **union of CAMO train and COD10K train (4040 images)**; testing uses CAMO (250), COD10K (2026) and NC4K (4121). Layout:

```
data/
├── train/
│   ├── Imgs/
│   ├── GT/
│   └── Edge/
├── camo_test/{Imgs,GT}/
├── cod10k_test/{Imgs,GT}/
└── nc4k/{Imgs,GT}/
```

Images and masks are matched by **stem** (masks are `.png`). Please download the datasets from their official pages; this repository does not redistribute them.

### 5. Checkpoints

| File | Note |
|---|---|
| `weights/v8uCGR_best.pth` | final model (stage 1 + stage 2), ready for inference. Download from [Releases](https://github.com/javamcy/COD_Edge/releases/download/v1.0.0/v8uCGR_best.pth) and place it here |
| `weights/dinov3_vitl16.pth` | **training only**. Official DINOv3 ViT-L/16 weights; download from Meta/timm and place it here |

For inference, pass `--dino3_weight ""`: the checkpoint already contains the full backbone.

### 6. Reproduction

**Stage 1 (50 epochs, 448, batch 8, last 8 blocks unfrozen)**

```bash
bash scripts/train_stage1.sh
# -> results/stage1/Net_epoch_best.pth
```

**Stage 2 (12 epochs, 768, batch 8, refinement head only)**

```bash
bash scripts/train_stage2.sh
# -> results/stage2/Net_epoch_best.pth
```

**Inference + evaluation (headline setting, 768)**

```bash
bash scripts/test.sh
# predictions: results/pred/{CAMO,COD10K,NC4K}/
# metrics:     results/eval_metrics.json
```

**Matched-input 896 setting**: change `--trainsize 768` to `896` in `scripts/test.sh`; the **same** checkpoint is used and nothing is retrained.

### 7. Expected results

Headline setting (768, single forward, no TTA):

| Dataset | Sα ↑ | Eϕ ↑ | Fωβ ↑ | MAE ↓ |
|---|---|---|---|---|
| CAMO (250) | 0.9236 | 0.9700 | 0.9084 | 0.0237 |
| COD10K (2026) | 0.9308 | 0.9754 | 0.8959 | 0.0121 |
| NC4K (4121) | 0.9281 | 0.9686 | 0.9096 | 0.0200 |

> **Reproduction note.** Re-scoring the same checkpoint with the bundled `eval_local.py` gives **0.9249** for CAMO Sα, 0.0013 above the paper's 0.9236. This is the known CAMO-Sα difference between an earlier remote evaluation file and the local evaluator; Eϕ / Fωβ / MAE match the paper to four decimals. The paper's 896 table and ablations already use the local evaluator.

Matched-input 896 setting (same checkpoint):

| Dataset | Sα ↑ | Eϕ ↑ | Fωβ ↑ | MAE ↓ |
|---|---|---|---|---|
| CAMO | 0.9257 | 0.9704 | 0.9105 | 0.0234 |
| COD10K | 0.9325 | 0.9757 | 0.8994 | 0.0120 |
| NC4K | 0.9282 | 0.9689 | 0.9104 | 0.0202 |

The last decimal may vary by ±0.0001–0.0003 across GPUs, drivers and library versions.

### 8. Notes

- **Unified evaluator.** All metrics are computed by `eval_local.py` + `evaltools/metrics.py`: the ground truth is binarised at 128, each prediction is min–max normalised per image, and quantities are averaged per image. This matches the paper.
- **Single seed.** The main table and ablations in the paper are single-seed; the paper additionally re-ran CODiff with three seeds to calibrate noise (Sα spread 0.0008). The checkpoint here is the seed used in the paper.
- **Machines.** Training was done on an A100 80 GB; the 896 comparison and cost measurements were done on an RTX 3080 Laptop 16 GB. Accuracy is hardware-independent; timings are comparable only within the same machine.
- **Evaluator credit.** `evaltools/metrics.py` is taken from the public CamoFormer evaluation tools (itself based on `lartpang`'s implementation) so that the metric convention matches the paper.

### 9. Citation

```bibtex
@article{anonymous2026boundary,
  title   = {A Boundary-Aware Detection Algorithm for Camouflaged Object Detection},
  author  = {Anonymous},
  journal = {Journal of King Saud University Computer and Information Sciences},
  year    = {2026}
}
```

### 10. License

Code is for academic research only. `evaltools/` follows its original open-source license; DINOv3 weights follow Meta's license; datasets follow their official licenses.
