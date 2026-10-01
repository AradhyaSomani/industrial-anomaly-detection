# Unsupervised Anomaly Detection for Industrial Inspection

Detecting and localizing manufacturing defects **without a single labeled defect**. The models are trained only on images of good parts and flag anything that doesn't look normal.

![Demo](docs/demo.gif)

*Auto-detects the product category, then highlights the defect. Runs locally with `python app.py`.*

📄 **[Technical documentation](docs/TECHNICAL.md)**: architecture, model details, scoring, full results and failure analysis.

---

## Why this problem is hard

Real production lines produce very few defects, and the defects they do produce are varied and unpredictable. Collecting and pixel-labeling enough examples to train a supervised classifier is expensive, and a supervised model only recognizes the defect types it was shown.

This project takes the unsupervised route used in industrial visual inspection: **learn what "normal" looks like, and treat deviation from it as an anomaly.** No anomaly labels are used for training *or* for choosing the decision threshold. Test labels and masks are used only to report metrics.

## Dataset

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad): about 5,350 high-resolution images across 15 categories (5 textures, 10 objects). Each category has 60–400 defect-free training images and a test set of good and defective images with pixel-level ground-truth masks. One model is trained per category.

## Approach

### 1. Pixel-reconstruction autoencoder (baseline)
A convolutional autoencoder (256×256 input, 16×16 bottleneck) trained to reconstruct normal images. Defective regions can't be reconstructed well, so reconstruction error highlights them.

- **Loss:** 0.5 · MSE + 0.5 · (1 − SSIM). Pure MSE produces blurry reconstructions whose edge errors cause false positives; SSIM focuses on structure, where most defects appear.
- **Decoder:** bilinear upsampling + convolution instead of transposed convolutions, which removed checkerboard artifacts that showed up as false anomalies.
- **Anomaly map:** per-pixel L2 error and SSIM error, each z-normalized using statistics from held-out normal images, combined and Gaussian-smoothed (σ = 4).

### 2. Feature-reconstruction autoencoder (main model)
Instead of reconstructing pixels, the autoencoder reconstructs **features from a frozen ImageNet-pretrained WideResNet-50** (layers 2 + 3, 1536 channels at 32×32, with 3×3 local averaging for neighborhood context). A small 1×1-convolution autoencoder (latent size 200) learns to reconstruct normal feature vectors. The anomaly map is the per-location feature reconstruction error, upsampled and smoothed. This follows the idea of deep feature reconstruction (DFR).

Pretrained features already encode texture, color and part identity, so defects that are invisible in pixel error, such as fine texture damage or swapped and missing components, become obvious.

### Data augmentation
With only 60–400 training images per category, augmentation matters, but it must never create something that *looks like a defect*, or the model learns to treat real defects as normal.

| Category type | Augmentations | Avoided |
|---|---|---|
| Textures (carpet, grid, leather, tile, wood) | H/V flips, 90° rotations | noise, blur |
| Rotation-invariant objects (bottle, hazelnut, metal_nut, screw) | flips, 90° rotations | strong color jitter |
| Orientation-sensitive objects (cable, capsule, pill, toothbrush, transistor, zipper) | very mild brightness/contrast only | flips and rotations (a flipped transistor *is* a defect) |

### Scoring and threshold (fully unsupervised)
- **Image score:** mean of the top 1% of anomaly-map values. It is robust to single noisy pixels and doesn't dilute small defects the way a global mean does.
- **Threshold:** 15% of the training images are held out as a normal-only validation set. The threshold is the 95th percentile of their scores. No defective image is ever used to tune it.

## Results

Image-level AUROC measures separating good from defective parts; pixel-level AUROC measures defect localization. Feature-model values are the mean ± standard deviation over three training runs; the pixel model was trained once.

| Category | Pixel AE (img) | **Feature AE (img)** | Pixel AE (pix) | **Feature AE (pix)** |
|---|---|---|---|---|
| bottle | 0.969 | **1.000 ± 0.000** | 0.766 | **0.985 ± 0.000** |
| cable | 0.319 | **0.922 ± 0.013** | 0.507 | **0.969 ± 0.002** |
| capsule | 0.674 | **0.906 ± 0.009** | 0.844 | **0.989 ± 0.000** |
| carpet | 0.488 | **0.980 ± 0.006** | 0.910 | **0.988 ± 0.001** |
| grid | **0.872** | 0.860 ± 0.046 | **0.977** | 0.956 ± 0.005 |
| hazelnut | 0.919 | **0.998 ± 0.001** | 0.932 | **0.984 ± 0.000** |
| leather | 0.764 | **0.998 ± 0.001** | 0.963 | **0.992 ± 0.000** |
| metal_nut | 0.792 | **0.995 ± 0.001** | 0.580 | **0.975 ± 0.001** |
| pill | 0.840 | **0.917 ± 0.007** | 0.894 | **0.982 ± 0.001** |
| screw | **0.474** | 0.431 ± 0.076 | 0.912 | **0.967 ± 0.003** |
| tile | 0.647 | **0.997 ± 0.002** | 0.875 | **0.955 ± 0.002** |
| toothbrush | **0.986** | 0.949 ± 0.005 | 0.978 | **0.988 ± 0.000** |
| transistor | 0.645 | **0.931 ± 0.020** | 0.670 | **0.906 ± 0.003** |
| wood | 0.930 | **0.987 ± 0.000** | 0.842 | **0.942 ± 0.003** |
| zipper | 0.924 | **0.975 ± 0.002** | 0.958 | **0.985 ± 0.000** |
| **Mean** | 0.749 | **0.923 ± 0.005** | 0.841 | **0.971 ± 0.000** |

Feature reconstruction raises mean image AUROC from **0.749 → 0.923** and mean pixel AUROC from **0.841 → 0.971**. The small run-to-run spread on the mean (± 0.005) shows the gain is not a lucky run. The demo automatically uses whichever model scored higher for each category.

### Ablation: anomaly scoring (pixel AE, bottle)

| Scoring | Image AUROC | Pixel AUROC |
|---|---|---|
| L2 error only | 0.883 | 0.785 |
| SSIM error only | **0.951** | **0.909** |
| L2 + SSIM (combined) | 0.933 | 0.905 |

Bottle defects (cracks, broken rims, contamination) are structural changes that squared pixel error largely misses. SSIM gives the best separation; the combined score gave the best F1 at the automatic threshold.

### Qualitative results

![Carpet — feature AE](docs/carpet_qualitative.png)

*Input, anomaly map and ground-truth mask for each carpet defect type (feature autoencoder).*

## What didn't work, and why

- **Pixel AE on textures (carpet 0.488).** The autoencoder can't reproduce fine, high-frequency detail such as carpet fibers, so *every* image has high error and good and bad parts become indistinguishable. Pixel AUROC stayed around 0.91: the heatmaps found the defects, but scattered errors on normal images swamped the image-level score.
- **Pixel AE on cable (0.319, worse than random).** Several cable defects are *logical*: a missing wire or swapped wire colors. A missing wire makes the image *simpler* and easier to reconstruct, so it scores as *more* normal. Reconstruction error can't detect "the parts are in the wrong arrangement." Pretrained features fix this (0.922) because they encode color and part identity.
- **Screw remains unsolved (0.431 ± 0.076 with features).** Localization is good (pixel AUROC 0.967), but good and defective screws get almost identical image scores. Full-range rotation augmentation and finer top-k scoring didn't help; 512 px input raised it to 0.424, which points to defects being smaller than what one cell of the 32×32 feature grid can register. Tile-based inference at full resolution is the next thing to try.
- **Grid and toothbrush.** Grid's feature-model result varies a lot between runs (± 0.046), and toothbrush is consistently lower with the feature model than the pixel model (0.949 vs 0.986), so the demo uses the pixel model for toothbrush and screw.
- **AUROC vs. threshold.** Zipper's pixel AE reached 0.92 image AUROC but only 0.29 recall at a 99th-percentile threshold calibrated on ~36 normal images. The model separated the classes fine; the threshold was too strict. Moving to the 95th percentile raised recall to 0.94. AUROC measures the model; precision and recall measure the threshold.

## Demo features

- Upload any image and get an anomaly heatmap, bounding boxes around the most anomalous regions, and a verdict with score and threshold.
- **Automatic category detection** by nearest neighbours on a global feature vector: 100% correct on all 1,725 test images. The first version picked the category where the image looked most normal, which misrouted 31% of *defective* images, because a defect makes an image look abnormal to its own model. Measuring the feature exposed the flaw.
- **Threshold slider** to explore the precision/recall trade-off live.
- Switch between pixel and feature models; inference time is shown (~30 ms per image on an Apple-silicon GPU).

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

# Category-detection index (after training the feature models)
python src/build_category_index.py

# Demo
python app.py
```

Uses CUDA or Apple-silicon (MPS) GPUs automatically, falling back to CPU. Training all 15 categories takes a few hours on a laptop GPU.

## Limitations and future work

- One model per category; a single multi-class model would be more practical to deploy.
- Thresholds are calibrated on small normal-only validation sets (~10–60 images), so they are noisy. Real deployments would calibrate on more data and pick the operating point from the cost of a missed defect versus a false alarm.
- The pixel baseline was trained once per category; the feature model three times.
- Next steps: tile-based full-resolution inference for screw, the PRO localization metric, a comparison against memory-bank methods such as PatchCore, and ONNX export for faster CPU inference.

## References

- Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
- Bergmann et al., *Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders*, 2018.
- Shi et al., *Unsupervised Anomaly Segmentation via Deep Feature Reconstruction*, Neurocomputing 2021.

The MVTec AD dataset is licensed under CC BY-NC-SA 4.0 and is not included in this repository.
