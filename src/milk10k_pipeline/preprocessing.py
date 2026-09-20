"""
Part 3 — Reusable image processing functions.

Built once here, imported everywhere else (color_analysis, data_loader,
future project milestones) instead of copy-pasted per script.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


@dataclass
class ProcessInfo:
    """What was done to an image, returned alongside the processed array."""
    source: str
    original_size_wh: tuple
    final_shape: tuple
    dtype: str
    value_range: tuple
    color_space: str
    normalization: str


def preprocess_image(
    image: Union[str, Path, np.ndarray],
    size: tuple = (224, 224),
    color_space: str = "rgb",         # "rgb" or "grayscale"
    normalization: str = "minmax",     # "minmax", "zscore", or "none"
    mean=None,
    std=None,
) -> tuple[np.ndarray, ProcessInfo]:
    """
    Task 9. Configurable single-image preprocessing pipeline.

    Parameters
    ----------
    image : path to a .jpg/.png OR an already-loaded HxWx3 uint8 array
    size : target (width, height) to resize to
    color_space : "rgb" (H,W,3) or "grayscale" (H,W)
    normalization :
        "minmax" -> x / 255.0, output range [0, 1]   (documented default:
                    simple, robust first step, matches Session 2 slides)
        "zscore" -> (x - mean) / std, output roughly zero-centered
                    (requires `mean`/`std`, e.g. dataset stats or ImageNet
                    stats -- pass whichever your downstream model expects)
        "none"   -> leave pixel values as-is (0-255 float)

    Returns
    -------
    (array, info) where `info` is a ProcessInfo describing exactly what
    was done (final shape, dtype, value range) so downstream code (and
    future-you) doesn't have to guess.
    """
    if isinstance(image, (str, Path)):
        source = str(image)
        img = Image.open(image).convert("RGB")
    else:
        source = "<in-memory array>"
        img = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")

    original_size_wh = img.size  # PIL: (W, H)

    img = img.resize(size)
    arr = np.array(img).astype(np.float32)  # (H, W, 3)

    if color_space == "grayscale":
        # same luminosity weights covered in Session 2 theory
        arr = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    elif color_space != "rgb":
        raise ValueError("color_space must be 'rgb' or 'grayscale'")

    if normalization == "minmax":
        arr = arr / 255.0
    elif normalization == "zscore":
        if mean is None or std is None:
            raise ValueError("normalization='zscore' requires `mean` and `std`")
        arr = (arr / 255.0 - np.asarray(mean)) / np.asarray(std)
    elif normalization != "none":
        raise ValueError("normalization must be 'minmax', 'zscore', or 'none'")

    info = ProcessInfo(
        source=source,
        original_size_wh=original_size_wh,
        final_shape=arr.shape,
        dtype=str(arr.dtype),
        value_range=(float(arr.min()), float(arr.max())),
        color_space=color_space,
        normalization=normalization,
    )
    return arr, info


@dataclass
class BatchResult:
    images: object              # np.ndarray if all same shape, else list[np.ndarray]
    labels: list
    ids: list
    skipped: list = field(default_factory=list)  # list of (id_or_path, reason)


def preprocess_batch(
    items,
    img_dir=None,
    label_col: str = None,
    id_col: str = "isic_id",
    **preprocess_kwargs,
) -> BatchResult:
    """
    Task 10. Apply `preprocess_image` to a batch of images.

    Parameters
    ----------
    items : list[str|Path] of image paths, OR a pandas DataFrame of
        metadata rows (in which case `img_dir` and `id_col` are used to
        build each path, and `label_col` -- e.g. "diagnosis_1" -- is
        pulled out as the label).
    img_dir : required if `items` is a DataFrame.
    label_col : column name to use as the label, if `items` is a DataFrame.
    id_col : column holding the image id, if `items` is a DataFrame.
    **preprocess_kwargs : forwarded to `preprocess_image` (size, color_space,
        normalization, mean, std).

    Returns
    -------
    BatchResult with a stacked array (if every processed image ended up
    the same shape -- true whenever a fixed `size` is used) or a plain
    list otherwise, plus which items were skipped and why. One bad file
    never crashes the whole batch.
    """
    is_dataframe = hasattr(items, "iterrows")

    images, labels, ids, skipped = [], [], [], []

    iterator = items.iterrows() if is_dataframe else enumerate(items)
    for key, item in iterator:
        if is_dataframe:
            row = item
            img_id = row[id_col]
            path = Path(img_dir) / f"{img_id}.jpg"
            label = row[label_col] if label_col else None
        else:
            path = item
            img_id = Path(path).stem
            label = None

        try:
            arr, _info = preprocess_image(path, **preprocess_kwargs)
        except (FileNotFoundError, OSError, ValueError) as e:
            skipped.append((img_id, f"{type(e).__name__}: {e}"))
            continue

        images.append(arr)
        labels.append(label)
        ids.append(img_id)

    # stack into one array if every image ended up the same shape
    if images and all(im.shape == images[0].shape for im in images):
        stacked = np.stack(images, axis=0)
    else:
        stacked = images  # heterogeneous shapes -> keep as a list

    return BatchResult(images=stacked, labels=labels, ids=ids, skipped=skipped)
