# Unsupervised Anomaly Detection for Industrial Inspection

Detecting and localizing manufacturing defects **without a single labeled defect**. All models are trained only on images of good parts and flag anything that doesn't look normal.

![Demo](docs/demo.gif)

*Auto-detects the product category, then highlights the defect. Runs locally with `python app.py`.*

📄 **[Technical documentation](docs/TECHNICAL.md)**: architecture, model details, scoring, full results and failure analysis.

**Results on MVTec AD (mean over 15 categories):**

| Model | Image AUROC | Pixel AUROC | Size per category | Inference |
|---|---|---|---|---|
| Pixel autoencoder (baseline) | 0.749 | 0.841 | | |
| Feature autoencoder | 0.923 ± 0.005 | 0.971 | ~14 MB | ~25 ms |
| **PatchCore (1% coreset)** | **0.982** | **0.980** | **3–21 MB** | **~11 ms** |

---

## Why this problem is hard

Real production lines produce very few defects, and the defects they do produce are varied and unpredictable. Collecting and pixel-labeling enough examples to train a supervised classifier is expensive, and a supervised model only recognizes the defect types it was shown.

This project takes the unsupervised route used in industrial visual inspection: **learn what "normal" looks like, and treat deviation from it as an anomaly.** No anomaly labels are used for training *or* for choosing the decision threshold. Test labels and masks are used only to report metrics.

## Dataset

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad): about 5,350 high-resolution images across 15 categories (5 textures, 10 objects). Each category has 60–400 defect-free training images and a test set of good and defective images with pixel-level ground-truth masks. One model is built per category.

## Approach

The project compares three ways of modelling "normal", built in order:

### 1. Pixel-reconstruction autoencoder (baseline)
A convolutional autoencoder (256×256 input, 16×16 bottleneck) trained to reconstruct normal images. Defective regions can't be reconstructed well, so reconstruction error highlights them.

- **Loss:** 0.5 · MSE + 0.5 · (1 − SSIM). Pure MSE produces blurry reconstructions whose edge errors cause false positives; SSIM focuses on structure, where most defects appear.
- **Decoder:** bilinear upsampling + convolution instead of transposed convolutions, which removed checkerboard artifacts that showed up as false anomalies.
- **Anomaly map:** per-pixel L2 error and SSIM error, each z-normalized using statistics from held-out normal images, combined and Gaussian-smoothed (σ = 4).

### 2. Feature-reconstruction autoencoder
Instead of reconstructing pixels, the autoencoder reconstructs **features from a frozen ImageNet-pretrained WideResNet-50** (layers 2 + 3, 1536 channels at 32×32, with 3×3 local averaging for neighborhood context). A small 1×1-convolution autoencoder (latent size 200) learns to reconstruct normal feature vectors. This follows the idea of deep feature reconstruction (DFR).

Pretrained features already encode texture, color and part identity, so defects that are invisible in pixel error, such as fine texture damage or swapped and missing components, become obvious.

### 3. PatchCore (memory bank)
PatchCore ([Roth et al., 2022](https://arxiv.org/abs/2106.08265)), implemented from scratch on **exactly the same WideResNet-50 features**. Instead of learning to reconstruct normal patches, it stores them: the feature vectors of every patch in the training images, reduced by greedy coreset selection to a representative 1%. A test patch's score is its distance to the nearest stored normal patch, and the image score is the worst patch. Nothing is trained.

Because both feature-based models share the same backbone, layers, resolution and validation split, the comparison isolates one question: **reconstruct normal features, or look them up?**

### Data augmentation
With only 60–400 training images per category, augmentation matters for the autoencoders, but it must never create something that *looks like a defect*, or the model learns to treat real defects as normal.

| Category type | Augmentations | Avoided |
|---|---|---|
| Textures (carpet, grid, leather, tile, wood) | H/V flips, 90° rotations | noise, blur |
| Rotation-invariant objects (bottle, hazelnut, metal_nut, screw) | flips, 90° rotations | strong color jitter |
| Orientation-sensitive objects (cable, capsule, pill, toothbrush, transistor, zipper) | very mild brightness/contrast only | flips and rotations (a flipped transistor *is* a defect) |

### Scoring and threshold (fully unsupervised)
- **Image score:** mean of the top 1% of anomaly-map values for the autoencoders; the single highest patch score for PatchCore.
- **Threshold:** 15% of the training images are held out as a normal-only validation set (the same seeded split for every model). The threshold is the 95th percentile of their scores. No defective image is ever used to tune it.

## Results

Image-level AUROC measures separating good from defective parts; pixel-level AUROC measures defect localization. Feature-autoencoder values are the mean over three training runs (± std); the pixel autoencoder was trained once; PatchCore involves no training.

| Category | Pixel AE | Feature AE | **PatchCore** | Feature AE (pix) | **PatchCore (pix)** |
|---|---|---|---|---|---|
| bottle | 0.969 | 1.000 ± 0.000 | **1.000** | 0.985 | **0.986** |
| cable | 0.319 | 0.922 ± 0.013 | **0.998** | 0.969 | **0.986** |
| capsule | 0.674 | 0.906 ± 0.009 | **0.978** | **0.989** | 0.988 |
| carpet | 0.488 | 0.980 ± 0.006 | **0.991** | 0.988 | **0.990** |
| grid | 0.872 | 0.860 ± 0.046 | **0.971** | 0.956 | **0.981** |
| hazelnut | 0.919 | 0.998 ± 0.001 | **1.000** | 0.984 | **0.988** |
| leather | 0.764 | 0.998 ± 0.001 | **1.000** | 0.992 | **0.993** |
| metal_nut | 0.792 | 0.995 ± 0.001 | **1.000** | 0.975 | **0.986** |
| pill | 0.840 | 0.917 ± 0.007 | **0.954** | **0.982** | 0.978 |
| screw | 0.474 | 0.431 ± 0.076 | **0.940** | 0.967 | **0.974** |
| tile | 0.647 | 0.997 ± 0.002 | **1.000** | 0.955 | **0.958** |
| toothbrush | **0.986** | 0.949 ± 0.005 | 0.928 | 0.988 | 0.988 |
| transistor | 0.645 | 0.931 ± 0.020 | **0.998** | 0.906 | **0.971** |
| wood | 0.930 | **0.987 ± 0.000** | 0.986 | 0.942 | **0.944** |
| zipper | 0.924 | 0.975 ± 0.002 | **0.985** | 0.985 | **0.987** |
| **Mean** | 0.749 | 0.923 ± 0.005 | **0.982** | 0.971 | **0.980** |

PatchCore is the best or tied-best model on 13 of 15 categories. The demo automatically uses whichever model scored highest for each category, which is PatchCore everywhere except toothbrush (pixel autoencoder) and wood (feature autoencoder, a 0.001 difference).

### Ablation: PatchCore memory-bank size

| Coreset | Mean image AUROC | Mean pixel AUROC | Bank per category | Build time | Inference |
|---|---|---|---|---|---|
| 10% | 0.980 | 0.981 | 5,222–34,099 patches (~30–210 MB) | 9 s – 6 min | 13–41 ms |
| **1%** | **0.982** | **0.980** | **522–3,409 patches (~3–21 MB)** | **2–46 s** | **11–12 ms** |

Keeping 1% of patches instead of 10% loses nothing, consistent with the PatchCore paper, and makes the memory bank about the same size as the feature autoencoder's weights.

### Ablation: anomaly scoring (pixel AE, bottle)

| Scoring | Image AUROC | Pixel AUROC |
|---|---|---|
| L2 error only | 0.883 | 0.785 |
| SSIM error only | **0.951** | **0.909** |
| L2 + SSIM (combined) | 0.933 | 0.905 |

Bottle defects (cracks, broken rims, contamination) are structural changes that squared pixel error largely misses.

### Qualitative results

![Carpet — feature AE](docs/carpet_qualitative.png)

*Input, anomaly map and ground-truth mask for each carpet defect type (feature autoencoder).*

## What didn't work, and why

- **Pixel AE on textures (carpet 0.488).** The autoencoder can't reproduce fine, high-frequency detail such as carpet fibers, so *every* image has high error and good and bad parts become indistinguishable. Pixel AUROC stayed around 0.91: the heatmaps found the defects, but scattered errors on normal images swamped the image-level score.
- **Pixel AE on cable (0.319, worse than random).** Several cable defects are *logical*: a missing wire or swapped wire colors. A missing wire makes the image *simpler* and easier to reconstruct, so it scores as *more* normal. Pretrained features fix this because they encode color and part identity.
- **Screw, with reconstruction (0.431 ± 0.076).** Localization was good (pixel AUROC 0.967), but good and defective screws got almost identical image scores. Full-range rotation augmentation, finer top-k scoring and 512 px input (0.424) didn't fix it. **PatchCore on the same features reaches 0.940**, so the bottleneck was the reconstruction approach itself, not the features or the resolution: the autoencoder generalizes well enough to reconstruct a slightly damaged thread, while a nearest-neighbour lookup finds no close normal match for it.
- **Toothbrush.** The pixel autoencoder (0.986) beats both feature-based models (0.949, 0.928). Toothbrush has only 51 training images, the smallest category, which gives PatchCore a bank of just 522 patches.
- **AUROC vs. threshold.** Zipper's pixel AE reached 0.92 image AUROC but only 0.29 recall at a 99th-percentile threshold calibrated on ~36 normal images. Moving to the 95th percentile raised recall to 0.94. AUROC measures the model; precision and recall measure the threshold.

## Demo features

- Upload any image and get an anomaly heatmap, bounding boxes around the most anomalous regions, and a verdict with score and threshold.
- **Automatic category detection** by nearest neighbours on a global feature vector: 100% correct on all 1,725 test images. The first version picked the category where the image looked most normal, which misrouted 31% of *defective* images, because a defect makes an image look abnormal to its own model. Measuring the feature exposed the flaw.
- **Model choice:** best per category (default), PatchCore, feature autoencoder or pixel autoencoder.
- **Threshold slider** to explore the precision/recall trade-off live.
- Inference time is shown (~11 ms per image with PatchCore on an Apple-silicon GPU).

## Project structure

```
src/
  dataset.py               MVTec loader with category-aware augmentation
  model.py                 Convolutional (pixel) autoencoder
  scoring.py               Error maps, normalization, image score
  train.py                 Train pixel AE
  evaluate.py              Evaluate pixel AE (L2 / SSIM / combined), calibrate threshold
  feature_model.py         WideResNet-50 feature extractor + feature autoencoder
  train_feat.py            Train feature AE (--size for higher resolution)
  evaluate_feat.py         Evaluate feature AE, calibrate threshold (--size, --topk)
  patchcore.py             PatchCore: coreset memory bank, evaluation, saved bank for the app
  build_category_index.py  Build the index used for automatic category detection
  eval_autodetect.py       Measure category-detection accuracy on the test set
app.py                     Gradio demo
docs/                      Technical documentation, figures and demo GIF
```

## Running it yourself

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download MVTec AD into data/ (category folders directly inside data/)

# Pixel autoencoder
python src/train.py --category bottle --epochs 200
python src/evaluate.py --category bottle --pct 95

# Feature autoencoder
python src/train_feat.py --category bottle
python src/evaluate_feat.py --category bottle

# PatchCore (no training; builds and saves the memory bank)
python src/patchcore.py --category bottle

# Category-detection index (after training the feature models)
python src/build_category_index.py

# Demo
python app.py
```

Uses CUDA or Apple-silicon (MPS) GPUs automatically, falling back to CPU.

## Limitations and future work

- One model per category; a single multi-class model would be more practical to deploy.
- Thresholds are calibrated on small normal-only validation sets (~10–60 images), so they are noisy. Real deployments would calibrate on more data and pick the operating point from the cost of a missed defect versus a false alarm.
- The simplified PatchCore omits the paper's score reweighting and channel reduction; the published method reports ~0.99 mean image AUROC.
- Next steps: the PRO localization metric and a per-defect-type breakdown, and ONNX export for CPU latency.

## References

- Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
- Bergmann et al., *Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders*, 2018.
- Shi et al., *Unsupervised Anomaly Segmentation via Deep Feature Reconstruction*, Neurocomputing 2021.
- Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (PatchCore), CVPR 2022.

The MVTec AD dataset is licensed under CC BY-NC-SA 4.0 and is not included in this repository.
