# UA-MT on ISIC 2018

This repository tries to reproduce the semi-supervised skin lesion segmentation results reported in the semi-GDA paper by re-implementing **Uncertainty-Aware Mean Teacher (UA-MT)** and training/evaluating it on the **ISIC 2018 (Task 1: Lesion Boundary Segmentation)** dataset. The goal is to establish a faithful UA-MT baseline under the same low-labeled-data regime (10% labeled fraction) used in the semi-GDA study.

## Background

- **UA-MT** trains a student/teacher pair of U-Nets with EMA teacher weights. Unlabeled images are only used for a consistency loss between student and teacher predictions, gated by an uncertainty estimate obtained via Monte Carlo Dropout on the teacher. Only teacher predictions with low predictive entropy contribute to the consistency loss, which is ramped up over training.
- **semi-GDA** proposes a generative/discriminative augmentation approach for semi-supervised medical image segmentation and reports comparisons against UA-MT (among other semi-supervised baselines) on ISIC 2018.

## References

- Yu, L., Wang, S., Li, X., Fu, C.-W., & Heng, P.-A. (2019). *Uncertainty-aware Self-ensembling Model for Semi-supervised 3D Left Atrium Segmentation.* MICCAI 2019. [arXiv:1907.07034](https://arxiv.org/abs/1907.07034)
- Huang, K., Zhou, Y., Zhang, Y., Li, J., & Zhou, T. (2026). *SemiGDA: Generative Dual-distribution Alignment for Semi-Supervised Medical Image Segmentation.* CVPR 2026. [arXiv:2604.23274](https://arxiv.org/abs/2604.23274)
- ISIC 2018 Challenge: Codella, N. et al. (2019). *Skin Lesion Analysis Toward Melanoma Detection 2018.* [arXiv:1902.03368](https://arxiv.org/abs/1902.03368); dataset available at [challenge.isic-archive.com](https://challenge.isic-archive.com/data/#2018)


## Reproduction Status


| Labeled Fraction | Dice  | IoU   | 95HD  |
|-------------------|-------|-------|-------|
| 10%               | 83.48 | 75.98 | 22.89 |
| 30%               | 84.08 | 76.94 | 22.33 |

This reproduction currently matches the reported **Dice** and **IoU** scores, but **95HD (Hausdorff Distance) is not yet reproduced** — current runs fall short of/diverge from the paper's 95HD numbers above. The boundary-distance metric needs further investigation before the reproduction can be considered complete.

## Repository Structure

- `model.py` — U-Net with encoder-only Monte Carlo dropout (`UnetMCDoptout`), used for both student and teacher.
- `dataset.py` — `ISIC2018DataSet` (label-blind labeled/unlabeled split + augmentations) and `TwoStreamBatchSampler` (mixes labeled/unlabeled indices per batch).
- `utils.py` — CLI argument parsing, consistency-loss ramp-up schedule, Dice loss, and seeding.
- `main.py` — Training loop (student/teacher UA-MT) and evaluation (Dice, IoU, HD95) with best-checkpoint saving.

## Setup

```bash
pip install torch torchvision albumentations opencv-python medpy numpy scipy
```

## Dataset Layout

Point `--data-dir` at a directory containing the standard ISIC 2018 Task 1 folders:

```
<data-dir>/
  ISIC2018_Task1-2_Training_Input/
  ISIC2018_Task1_Training_GroundTruth/
  ISIC2018_Task1-2_Validation_Input/
  ISIC2018_Task1_Validation_GroundTruth/
  ISIC2018_Task1-2_Test_Input/
  ISIC2018_Task1_Test_GroundTruth/
```

## Usage

Train (and evaluate on the held-out test set at the end):

```bash
python main.py --data-dir /path/to/ISIC2018
```

Evaluate only, using an existing `checkpoint_best.pth`:

```bash
python main.py --data-dir /path/to/ISIC2018 --eval-only
```

Outputs:
- `checkpoint_best.pth` — best student/teacher checkpoint by validation Dice.
- `validation_log.csv` — Dice/IoU/HD95 logged every 200 iterations.
- `test_per_image_log.csv` — per-image test metrics from the final evaluation.
