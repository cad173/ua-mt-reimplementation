import os
import random
import itertools
import numpy as np
from torch.utils.data import Dataset, Sampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Seed for the labeled/unlabeled partition. The split is LABEL-BLIND: it draws
# purely from (sorted) filenames, never from mask content. 
SPLIT_SEED = 42

class ISIC2018DataSet(Dataset):
    def __init__(self, image_dir, mask_dir, mode, labeled_fraction, aug_seed=SPLIT_SEED):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mode = mode

        rng = random.Random(SPLIT_SEED)
        self.all_names = sorted(name for name in os.listdir(image_dir) if name.endswith(".jpg"))

        if mode in ('labeled', 'unlabeled', 'train'):
            self.labeled, self.unlabeled = self._make_split(rng, labeled_fraction)
        else:
            rng.shuffle(self.all_names)

        if self.mode == 'labeled':
            self.samples = self.labeled
        elif self.mode == 'unlabeled':
            self.samples = self.unlabeled
        elif self.mode == 'train':
            self.samples = self.labeled + self.unlabeled
            self.labeled_indices = list(range(len(self.labeled)))
            self.unlabeled_indices = list(range(len(self.labeled), len(self.samples)))
        else:
            self.samples = self.all_names

        # augmentation: flips + 90-degree rotations only.
        # RandomResizedCrop (scale/zoom-invariance) + brightness/contrast jitter
        # target the worst-HD95 tail: under-segmentation of large, low-contrast lesions. 
        self.transform = A.Compose([
            A.RandomResizedCrop(size=(224, 224), scale=(0.75, 1.0), ratio=(0.75, 1.33), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.5),
            A.Normalize( mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) ),
            ToTensorV2(),
        ], seed=aug_seed)

        self.val_transform = A.Compose([
            A.Resize(224,224),
            A.Normalize( mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) ),
            ToTensorV2(),
        ])

    def _make_split(self, rng, labeled_fraction):
        """Partition self.all_names into (labeled, unlabeled) using ONLY the
        filenames — never mask content — so no ground-truth label of the
        unlabeled set can influence experimental design.

        The chosen labeled files are frozen to a split file: written on first
        run, reloaded verbatim afterwards, so the split is reproducible across
        machines / Python versions and inspectable. Delete the file to regenerate.
        """
        split_path = f"labeled_split_seed{SPLIT_SEED}_frac{labeled_fraction}.txt"

        if os.path.exists(split_path):
            with open(split_path) as f:
                labeled_set = {line.strip() for line in f if line.strip()}
            labeled = [n for n in self.all_names if n in labeled_set]
            unlabeled = [n for n in self.all_names if n not in labeled_set]
            return labeled, unlabeled


        names = list(self.all_names)
        rng.shuffle(names)
        n_labeled = round(len(names) * labeled_fraction)
        labeled, unlabeled = names[:n_labeled], names[n_labeled:]

        with open(split_path, "w") as f:
            f.write("\n".join(sorted(labeled)) + "\n")
        return labeled, unlabeled

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        # pick the right list based on self.mode to get the image filename at idx
        img_name = self.samples[idx]

        # build full paths
        img_path = os.path.join(self.image_dir, img_name)

        # load image and mask
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.mode != 'unlabeled':
             # derive the mask filename from the image name (stip _segmentation.png, add .jpg)
            mask_name = img_name.replace('.jpg', '_segmentation.png')
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # apply and transform based on mode
        if self.mode in ('labeled', 'unlabeled', 'train'):
             augmented = self.transform(image=img) if self.mode == 'unlabeled' else self.transform(image=img, mask=mask)
        else:
            augmented = self.val_transform(image=img, mask=mask)

        img = augmented['image']
        if self.mode != 'unlabeled':
            mask = (augmented["mask"] > 0).long()

        # return the right thing based on mode
        if self.mode == 'unlabeled':
            return img
        else:
            return img, mask


class TwoStreamBatchSampler(Sampler):
    """Yield batches that mix a fixed number of labeled and unlabeled indices.

    Each yielded batch is ``labeled_batch_size`` labeled indices followed by
    (batch_size - labeled_batch_size) unlabeled indices, so downstream code can
    split a collated batch with a constant ``labeled_bs`` slice. Labeled indices
    are traversed once per epoch (defining epoch length); unlabeled indices are
    cycled *eternally*, reshuffled on every cycle.

    Crucially we cycle raw indices, not materialized batches: each index is
    re-dispatched to the Dataset, so augmentations are freshly sampled and no
    tensors are cached."""

    def __init__(self, labeled_indices, unlabeled_indices, batch_size, labeled_batch_size):
        self.labeled_indices = list(labeled_indices)
        self.unlabeled_indices = list(unlabeled_indices)
        self.labeled_batch_size = labeled_batch_size
        self.unlabeled_batch_size = batch_size - labeled_batch_size
        assert len(self.labeled_indices) >= self.labeled_batch_size > 0
        assert len(self.unlabeled_indices) >= self.unlabeled_batch_size > 0

    def __iter__(self):
        labeled_iter = iter(np.random.permutation(self.labeled_indices))
        unlabeled_iter = _iterate_eternally(self.unlabeled_indices)
        return (
            [int(i) for i in labeled_batch] + [int(i) for i in unlabeled_batch]
            for labeled_batch, unlabeled_batch in zip(
                _grouper(labeled_iter, self.labeled_batch_size),
                _grouper(unlabeled_iter, self.unlabeled_batch_size),
            )
        )

    def __len__(self):
        return len(self.labeled_indices) // self.labeled_batch_size


def _iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def _grouper(iterable, n):
    "Collect data into fixed-length chunks, dropping any final partial chunk."
    args = [iter(iterable)] * n
    return zip(*args)
