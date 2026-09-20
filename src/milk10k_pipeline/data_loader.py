"""
Part 4 — Data loader.

The object that will feed (processed image, label) batches into model
training code in later course milestones.
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

    Images that fail to load/process are silently skipped (see
    `preprocessing.preprocess_batch`) and logged in `self.skipped`, so a
    single corrupt file never crashes an epoch. `len(loader)` gives the
    number of batches per epoch based on the *currently available* image
    count (post-filtering), not the raw metadata row count.
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
        **preprocess_kwargs,
    ):
        self.img_dir = Path(img_dir)
        self.target_col = target_col
        self.id_col = id_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.preprocess_kwargs = preprocess_kwargs

        # Task 13: only keep rows that actually have a downloaded image.
        self.metadata = filter_to_available_images(metadata, self.img_dir, id_col)
        self.skipped = []  # populated as batches are consumed

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
                    self.skipped.append((row[self.id_col], f"{type(e).__name__}: {e}"))
                    continue
                images.append(arr)
                labels.append(row[self.target_col])

            if not images:
                continue  # whole batch failed (rare) -- skip it, don't crash

            yield np.stack(images, axis=0), np.array(labels)
