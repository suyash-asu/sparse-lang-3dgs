# Geometry-Guided Sparse-View 3D Gaussian Splatting via Monocular Depth Priors

**Aadya Bubber · Divyaraj Nakum · Gerald Bowers · Suyash Dhir**  
Arizona State University - EEE 515 / RAS Graduate Coursework · Spring 2026

---

## Overview

3D Gaussian Splatting (3DGS) produces high-quality novel views but requires 100+ images for reliable reconstruction. With only 3–6 images, COLMAP's Structure-from-Motion (SfM) yields too few reliable 3D points. Gaussians degenerate into needle-like artifacts and renders are unusable.

This project introduces a **depth-lifted initialization** strategy that replaces the sparse SfM point cloud with a dense, COLMAP-anchored back-projection from monocular depth estimates. No changes to the 3DGS training loop are required, only initialization changes.

---

## Method

The pipeline has four stages:

```
Sparse RGB images
      │
      ▼
① MiDaS DPT-Large → relative inverse-depth map per image
      │
      ▼
② Per-image least-squares scale alignment to COLMAP anchor points
   (3,000–6,000 anchors/image; scale factors range 7.97× – 40.02×)
      │
      ▼
③ Back-projection at stride s=4 → metric 3D point cloud (PLY)
      │
      ▼
④ Vanilla 3DGS initialized from dense PLY (no training changes)
```

**Scale alignment (step 2)** is the enabling component. Without it, performance drops below the vanilla baseline (8.16 dB vs 9.54 dB at 6 views). With per-image alignment, performance rises to 11.57 dB, which is a +2.03 dB improvement.

**Preprocessing overhead:** ~3 minutes added to a standard 30-minute 3DGS training run on an A100.

---

## Results

### Truck scene - Tanks and Temples

| Method | Views | Init. points | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|
| 3DGS dense | 251 | 136,029 | 25.42 | 0.885 | 0.142 |
| 3DGS vanilla | 3 | 136,029 | 9.85 | 0.178 | 0.647 |
| **Ours (depth)** | **3** | **100,657** | **9.73** | **0.178** | **0.647** |
| 3DGS vanilla | 6 | 136,029 | 9.54 | 0.147 | 0.678 |
| **Ours (depth)** | **6** | **199,225** | **11.57 (+2.03 dB)** | **0.313 (2.1×)** | **0.543** |
| 3DGS vanilla | 12 | 136,029 | 15.56 | 0.553 | 0.392 |
| Ours (depth) | 12 | 402,734 | 12.65 ↓ | 0.321 ↓ | 0.575 ↑ |

### Train scene - generalization check

| Method | Views | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| 3DGS vanilla | 6 | 10.31 | 0.374 | 0.458 |
| **Ours** | **6** | **11.49 (+1.17 dB)** | **0.427** | **0.456** |
| 3DGS vanilla | 12 | 11.75 | 0.446 | 0.470 |

Our 6-view result (11.49 dB) nearly matches vanilla 3DGS at 12 views.

### Operating envelope

| View count | COLMAP quality | Depth prior effect |
|---|---|---|
| 3 views | Very poor but depth cloud is actually smaller (100k vs 136k SfM points) | Neutral / marginal |
| **6 views** | **Nearly degenerate** | **Strongest improvement (+2.03 dB)** |
| 12 views | Already reliable | **Counterproductive**: over-dense init destabilizes adaptive densification |

**Key practical guideline:** use depth-lifted initialization when COLMAP is expected to fail (≤6 images). At ≥12 images, use vanilla 3DGS directly.

### Open-vocabulary semantic alignment (Hypothesis 2 - null result)

LangSplat (OpenCLIP ViT-B-16 + SAM segmentation + per-scene autoencoder) was run on top of both vanilla and depth-init 6-view models for 5 text queries: `truck`, `wheel`, `wood`, `road`, `tree`.

| Query | Vanilla entropy (nats) | Ours (nats) | Δ |
|---|---|---|---|
| truck | 13.146 | 13.183 | +0.037 |
| wheel | 13.168 | 13.177 | +0.009 |
| wood | 13.146 | 13.143 | −0.003 |
| road | 13.178 | 13.181 | +0.003 |
| tree | 13.158 | 13.109 | −0.049 |

Entropy differences are <0.05 nats across all queries, no measurable improvement in semantic localization. At 6 views the language field is fundamentally underconstrained regardless of initialization quality. The +2.03 dB geometric improvement does not translate into better semantic grounding at this view count.

---

## Repository Structure

```
sparse-lang-3dgs/
├── our_method/        # Core pipeline: depth estimation, scale alignment, back-projection
├── scripts/           # Evaluation scripts, metrics computation (PSNR, SSIM, LPIPS)
├── notebooks/         # Exploratory analysis, visualization, LangSplat evaluation
├── results/           # Quantitative results, figures
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/suyash-asu/sparse-lang-3dgs
cd sparse-lang-3dgs
pip install -r requirements.txt
```

**Requirements:** `numpy`, `plyfile`, `opencv-python`, `torch`

**Additional dependencies (not in requirements.txt):**
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - vanilla 3DGS codebase
- [MiDaS DPT-Large](https://github.com/isl-org/MiDaS) - monocular depth estimation
- [COLMAP](https://colmap.github.io/) - for camera poses and SfM anchor points
- [LangSplat](https://github.com/minghanqin/LangSplat) - for the semantic alignment experiments

**Hardware used:** NVIDIA A100-SXM4-80GB via SLURM on ASU Sol supercomputer. 30,000 training iterations per run; preprocessing adds ~3 minutes.

---

## Running the Pipeline

> **Note:** Full end-to-end instructions assume you have COLMAP output (camera poses + sparse point cloud) for your scene. The pipeline replaces only the initialization step. 3DGS training runs unchanged.

**Step 1 - Depth estimation and scale alignment**
```bash
python our_method/depth_lift.py \
  --image_dir /path/to/images \
  --colmap_dir /path/to/colmap_output \
  --output_ply /path/to/output/depth_init.ply \
  --stride 4
```

**Step 2 - Run 3DGS with depth-lifted initialization**

Replace the SfM `.ply` path in your 3DGS config with the output from step 1. No other changes to the training pipeline are required.

**Step 3 - Evaluate**
```bash
python scripts/evaluate.py \
  --renders /path/to/rendered_views \
  --gt /path/to/ground_truth \
  --output results/metrics.json
```

> Script names above are illustrative. Refer to the actual files in `our_method/` and `scripts/` for exact entry points.

---

## Citation

If you build on this work:

```bibtex
@article{bubber2026sparsedepth3dgs,
  title     = {Geometry-Guided Sparse-View {3D} Gaussian Splatting via Monocular Depth Priors},
  author    = {Bubber, Aadya and Nakum, Divyaraj and Bowers, Gerald and Dhir, Suyash},
  institution = {Arizona State University},
  year      = {2026}
}
```

---

## Team Contributions

| Member | Primary role |
|---|---|
| **Suyash Dhir** | Depth-lifting pipeline, scale alignment implementation, back-projection, PLY export, integration with 3DGS training |
| Aadya Bubber | Literature review, introduction, LangSplat pipeline (Hypothesis 2), citations |
| Divyaraj Nakum | View subsampling pipeline, method section, equation derivations, figure preparation |
| Gerald Bowers | Experimental setup design, preliminary results section, 12-view degradation analysis, paper editing |
