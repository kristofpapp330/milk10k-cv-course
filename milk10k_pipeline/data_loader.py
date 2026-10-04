"""
Part 4 (Session 2) — Data loader, upgraded in Milestone 1 (B8).

- MILK10kDataLoader: the original Session 2 NumPy loader, kept for quick EDA.
  Milestone 1 change: a missing/broken image now raises an error unless
  allow_missing=True is passed explicitly.
- MILK10kDataset + make_dataloaders: the project Dataset/DataLoaders built on it:
  reads a split CSV and the label map, uses torchvision transforms
  (augmentation in train only), returns (image_tensor, label, isic_id),
  handles class imbalance (class weights + WeightedRandomSampler).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import preprocess_image


def filter_to_available_images(
    df: pd.DataFrame, img_dir, id_col: str = "isic_id"
) -> pd.DataFrame:
    """
    Task 13. Keep only rows whose image file actually exists on disk.
    Never assume every metadata row has a matching downloaded image.
    """
    img_dir = Path(img_dir)
    exists_mask = df[id_col].apply(lambda i: (img_dir / f"{i}.jpg").exists())
    return df.loc[exists_mask].reset_index(drop=True)


class MILK10kDataLoader:
    """
    Task 12/14. Minimal batching data loader for MILK10k.

    Construct with:
        loader = MILK10kDataLoader(
            metadata=df,              # metadata table (already filtered or not)
            img_dir="milk10k/images", # folder containing the .jpg files
            target_col="diagnosis_1", # label column to yield
            batch_size=32,
            size=(224, 224),          # passed through to preprocess_image
            color_space="rgb",
            normalization="minmax",
            mean=None, std=None,      # only needed if normalization="zscore"
            shuffle=True,
            seed=0,
        )

    Iterating over it (`for images, labels in loader: ...`) yields, per
    batch:
        images : np.ndarray, shape (batch_size, H, W, C) [or (H, W) if
                 color_space="grayscale"], preprocessed on the fly -- not
                 pre-loaded into memory all at once.
        labels : np.ndarray of the target_col values for that batch.

    Milestone 1 change (fail loudly): by default every metadata row must have
    an image on disk, otherwise a FileNotFoundError lists the missing files.
    Only with allow_missing=True are missing rows filtered out (Session 2
    behaviour) and broken files skipped and logged in `self.skipped`.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        img_dir,
        target_col: str = "diagnosis_1",
        id_col: str = "isic_id",
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 0,
        allow_missing: bool = False,
        **preprocess_kwargs,
    ):
        self.img_dir = Path(img_dir)
        self.target_col = target_col
        self.id_col = id_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.preprocess_kwargs = preprocess_kwargs

        self.allow_missing = allow_missing
        available = filter_to_available_images(metadata, self.img_dir, id_col)
        if len(available) < len(metadata) and not allow_missing:
            missing = sorted(set(metadata[id_col]) - set(available[id_col]))
            raise FileNotFoundError(
                f"{len(missing)} images listed in the metadata are missing from {self.img_dir}: "
                f"{missing[:10]}{' ...' if len(missing) > 10 else ''} (pass allow_missing=True to skip them)"
            )
        self.metadata = available
        self.skipped = []  # only used when allow_missing=True

    def __len__(self) -> int:
        return math.ceil(len(self.metadata) / self.batch_size)

    def __iter__(self):
        order = self.metadata.index.to_numpy().copy()  # .copy(): to_numpy()
        # can return a read-only view, which np.random.Generator.shuffle
        # refuses to shuffle in place.
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(order)

        for start in range(0, len(order), self.batch_size):
            batch_idx = order[start : start + self.batch_size]
            batch_rows = self.metadata.loc[batch_idx]

            images, labels = [], []
            for _, row in batch_rows.iterrows():
                path = self.img_dir / f"{row[self.id_col]}.jpg"
                try:
                    arr, _info = preprocess_image(path, **self.preprocess_kwargs)
                except (FileNotFoundError, OSError, ValueError) as e:
                    if not self.allow_missing:
                        raise
                    self.skipped.append((row[self.id_col], f"{type(e).__name__}: {e}"))
                    continue
                images.append(arr)
                labels.append(row[self.target_col])

            if not images:
                continue  # whole batch failed (rare) -- skip it, don't crash

            yield np.stack(images, axis=0), np.array(labels)


# =============================================================================
# Milestone 1 (B8): PyTorch Dataset and DataLoaders
# =============================================================================
import json
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from . import config
from .data_io import find_images, load_rgb


def seed_everything(seed=config.SEED):
    """Seed Python, NumPy and PyTorch so shuffling, sampling and augmentation are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MILK10kDataset(Dataset):
    """One item = one IMAGE: (image_tensor, integer_label, isic_id).

    split_csv : one of the split files (train.csv / val.csv / test.csv), one row per image
    label_map : the dict loaded from label_map.json
    views     : which image types to use (config.VIEWS; B6 option (b) = dermoscopic only)
    A missing file raises FileNotFoundError (fail loudly) unless allow_missing=True.
    """

    def __init__(self, split_csv, img_dir=config.IMG_DIR, label_map=None, transform=None,
                 views=config.VIEWS, target="primary", allow_missing=False, image_paths=None):
        table = pd.read_csv(split_csv)
        self.table = table[table["view"].isin(views)].reset_index(drop=True)
        target_info = label_map[target]
        self.class_names = target_info["class_names"]
        self.labels = self.table[target_info["source_column"]].map(target_info["mapping"]).astype(int).tolist()
        self.transform = transform
        self.paths = image_paths if image_paths is not None else find_images(img_dir)

        missing = [i for i in self.table["isic_id"] if i not in self.paths]
        if missing and not allow_missing:
            raise FileNotFoundError(f"{len(missing)} images of {split_csv} are missing from {img_dir}: {missing[:10]}")
        if missing:   # explicit opt-in only
            keep = ~self.table["isic_id"].isin(missing)
            self.labels = [l for l, k in zip(self.labels, keep) if k]
            self.table = self.table[keep].reset_index(drop=True)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        isic_id = self.table.at[i, "isic_id"]
        img = load_rgb(self.paths[isic_id], min_size=config.RESIZE)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], isic_id


def compute_class_weights(labels, n_classes):
    """Inverse-frequency class weights from the TRAIN labels: n_samples / (n_classes * count_c).

    A class with half the average frequency gets weight 2. Used in the loss: CrossEntropyLoss(weight=...).
    """
    counts = np.bincount(labels, minlength=n_classes)
    return (len(labels) / (n_classes * counts)).tolist()


def make_weighted_sampler(labels, seed=config.SEED):
    """WeightedRandomSampler: each image is drawn with probability 1 / (count of its class),
    so every class appears about equally often in the training batches."""
    counts = np.bincount(labels)
    sample_weights = [1.0 / counts[l] for l in labels]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True, generator=generator)


def make_dataloaders(label_map, train_transform, eval_transform, splits_dir=config.SPLITS_DIR,
                     batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, use_sampler=True, seed=config.SEED):
    """Build the three DataLoaders.

    train : augmentation + WeightedRandomSampler (or plain shuffle if use_sampler=False)
    val   : eval_transform, no shuffle
    test  : eval_transform, no shuffle
    """
    seed_everything(seed)
    paths = find_images()
    datasets = {
        "train": MILK10kDataset(splits_dir / "train.csv", label_map=label_map, transform=train_transform, image_paths=paths),
        "val": MILK10kDataset(splits_dir / "val.csv", label_map=label_map, transform=eval_transform, image_paths=paths),
        "test": MILK10kDataset(splits_dir / "test.csv", label_map=label_map, transform=eval_transform, image_paths=paths),
    }
    train_labels = datasets["train"].labels
    if use_sampler:
        train_loader = DataLoader(datasets["train"], batch_size=batch_size, sampler=make_weighted_sampler(train_labels, seed),
                                  num_workers=num_workers)
    else:
        train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  generator=torch.Generator().manual_seed(seed))
    loaders = {
        "train": train_loader,
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return datasets, loaders


def compute_normalization_stats(dataset):
    """Per-channel mean and std of the TRAIN images (dataset must use a transform WITHOUT Normalize).

    Statistics come from the training split only, so nothing about val/test leaks into preprocessing.
    """
    total = torch.zeros(3)
    total_sq = torch.zeros(3)
    n_pixels = 0
    for i in range(len(dataset)):
        x, _, _ = dataset[i]                  # (3, H, W) in [0, 1]
        total += x.sum(dim=(1, 2))
        total_sq += (x ** 2).sum(dim=(1, 2))
        n_pixels += x.shape[1] * x.shape[2]
    mean = total / n_pixels
    std = (total_sq / n_pixels - mean ** 2).sqrt()
    return [round(v, 4) for v in mean.tolist()], [round(v, 4) for v in std.tolist()]


def time_one_epoch(loader):
    """Iterate once over a DataLoader and return the time in seconds (measures loading cost only)."""
    start = time.time()
    for _ in loader:
        pass
    return time.time() - start
