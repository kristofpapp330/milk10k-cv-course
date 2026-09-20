"""
Part 5 — Data visualizer.

Small, reusable plotting utilities for sanity-checking the dataset and the
Part 4 data loader's output at a glance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _to_displayable(img: np.ndarray) -> np.ndarray:
    """Best-effort conversion of a processed array back into something
    matplotlib's imshow can render sensibly (handles normalized floats)."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = arr - arr.min()
        max_val = arr.max()
        if max_val > 0:
            arr = arr / max_val
        arr = (arr * 255).astype(np.uint8)
    return arr


def show_image_grid(images, labels=None, n: int = 6, title: str = None):
    """
    Task 15. Display a grid of sample images with their labels as titles.

    `images` can be:
      - a list/array of image arrays, or
      - a (images, labels) batch tuple straight from MILK10kDataLoader
        (in which case `labels` is taken from the tuple if not given
        separately).
    """
    import matplotlib.pyplot as plt

    if isinstance(images, tuple) and len(images) == 2:
        images, labels = images

    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.imshow(_to_displayable(images[i]))
        if labels is not None:
            ax.set_title(str(labels[i]), fontsize=9)
        ax.axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_class_balance(df: pd.DataFrame, target_col: str = "diagnosis_1", ax=None):
    """
    Task 16. Bar chart of how many images/lesions fall into each class.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    counts = df[target_col].value_counts().sort_values(ascending=False)
    counts.plot(kind="bar", ax=ax, color="#F08200")
    ax.set_ylabel("Number of images")
    ax.set_title(f"Class balance — {target_col}")

    imbalance_ratio = counts.max() / counts.min()
    ax.text(
        0.98, 0.95,
        f"max/min ratio: {imbalance_ratio:.1f}x",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
    )
    return ax, counts


def plot_batch_summary(processed_images, raw_images=None, n_examples: int = 4):
    """
    Task 17. Quick visual/statistical debug summary for a processed batch.

    Panel 1: histogram of all pixel values in the batch (confirms whether
             normalization behaved as intended -- e.g. should sit in
             [0, 1] for min-max, or be roughly zero-centered for z-score).
    Panel 2 (optional, if `raw_images` given): a small grid comparing a
             few raw images against their processed counterparts.
    """
    import matplotlib.pyplot as plt

    arr = np.asarray(processed_images)
    flat = arr.ravel()

    if raw_images is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(flat, bins=50, color="#555555")
        ax.set_title(
            f"Batch pixel value distribution "
            f"(min={flat.min():.3f}, max={flat.max():.3f}, mean={flat.mean():.3f})"
        )
        ax.set_xlabel("pixel value")
        ax.set_ylabel("count")
        return fig, ax

    n_examples = min(n_examples, len(processed_images), len(raw_images))
    fig, axes = plt.subplots(3, n_examples, figsize=(2.6 * n_examples, 7))

    for i in range(n_examples):
        axes[0, i].imshow(_to_displayable(raw_images[i]))
        axes[0, i].set_title("raw", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(_to_displayable(processed_images[i]))
        axes[1, i].set_title("processed", fontsize=8)
        axes[1, i].axis("off")

    # bottom row: full-batch pixel value histogram (shown once, spanning row)
    gs = axes[2, 0].get_gridspec()
    for a in axes[2, :]:
        a.remove()
    hist_ax = fig.add_subplot(gs[2, :])
    hist_ax.hist(flat, bins=50, color="#555555")
    hist_ax.set_title(
        f"Batch pixel value distribution "
        f"(min={flat.min():.3f}, max={flat.max():.3f}, mean={flat.mean():.3f})"
    )

    fig.tight_layout()
    return fig, axes
