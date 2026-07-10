import argparse
import random
import numpy as np
import torch
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="UA-MT training on ISIC2018")
    parser.add_argument("--data-dir", type=Path, required=True, help="Root directory containing the ISIC2018 dataset folders")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; load checkpoint_best.pth and evaluate on the test set")
    return parser.parse_args()


def ramp_up(epoch, ramp_up_duration, max_weight=0.1):
    epoch = np.clip(epoch, 0, ramp_up_duration)
    rampup_fraction = np.exp(-5 * (1 - epoch / ramp_up_duration) ** 2)
    lambda_t = max_weight * rampup_fraction
    return lambda_t, rampup_fraction


def dice_loss(pred, target):
    pred = torch.softmax(pred, dim=1)[:, 1, :, :]
    target = target.float()
    intersection = torch.sum(pred * target, dim=(1, 2))
    denominator = torch.sum(pred, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
    per_image_dice = (2 * intersection + 1e-5) / (denominator + 1e-5)
    dice = 1 - per_image_dice.mean()
    return dice


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
