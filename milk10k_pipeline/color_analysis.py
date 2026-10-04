"""
Part 2 — Color and histogram analysis across the dataset.

Extends the single-image histogram work from Session 2 to a dataset-level,
per-class comparison.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _load_rgb_array(img_dir: Path, isic_id: str) -> np.ndarray:
    path = Path(img_dir) / f"{isic_id}.jpg"
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def sample_per_class(
    df: pd.DataFrame,
    target_col: str = "diagnosis_1",
    n_per_class: int = 25,
    seed: int = 0,
) -> pd.DataFrame:
    """Take up to n_per_class rows per class (fewer if a class is smaller).

    NOTE: implemented as an explicit loop + concat rather than
    `groupby().apply()` because newer pandas versions (2.2+) silently drop
    the grouping column when the applied function returns a same-shaped
    subset of the group -- a real, easy-to-miss version-compatibility trap.
    """
    parts = [
        g.sample(min(n_per_class, len(g)), random_state=seed)
        for _, g in df.groupby(target_col)
    ]
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# Task 5: average histograms per class
# --------------------------------------------------------------------------

def average_grayscale_histogram(
    rows: pd.DataFrame, img_dir, bins: int = 256
) -> np.ndarray:
    """Average grayscale intensity histogram (density) over a set of rows."""
    import cv2

    acc = np.zeros(bins, dtype=np.float64)
    n = 0
    for _, r in rows.iterrows():
        arr = _load_rgb_array(img_dir, r["isic_id"])
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        hist, _ = np.histogram(gray.ravel(), bins=bins, range=(0, 255), density=True)
        acc += hist
        n += 1
    return acc / max(n, 1)


def average_rgb_histograms(
    rows: pd.DataFrame, img_dir, bins: int = 256
) -> dict[str, np.ndarray]:
    """Average per-channel RGB histogram (density) over a set of rows."""
    acc = {"R": np.zeros(bins), "G": np.zeros(bins), "B": np.zeros(bins)}
    n = 0
    for _, r in rows.iterrows():
        arr = _load_rgb_array(img_dir, r["isic_id"])
        for i, ch in enumerate("RGB"):
            hist, _ = np.histogram(
                arr[:, :, i].ravel(), bins=bins, range=(0, 255), density=True
            )
            acc[ch] += hist
        n += 1
    return {ch: v / max(n, 1) for ch, v in acc.items()}


def histograms_by_class(
    df: pd.DataFrame,
    img_dir,
    target_col: str = "diagnosis_1",
    n_per_class: int = 25,
    bins: int = 256,
    seed: int = 0,
) -> dict:
    """
    Task 5. For every class, compute the average grayscale histogram and
    the average per-channel RGB histograms over a consistent sample size.

    Returns
    -------
    {
        "<class_name>": {"gray": np.ndarray, "rgb": {"R": ..., "G": ..., "B": ...}},
        ...
    }
    """
    sample = sample_per_class(df, target_col, n_per_class, seed)
    out = {}
    for cls, rows in sample.groupby(target_col):
        out[cls] = {
            "gray": average_grayscale_histogram(rows, img_dir, bins),
            "rgb": average_rgb_histograms(rows, img_dir, bins),
        }
    return out


def plot_avg_grayscale_by_class(hist_by_class: dict, ax=None):
    """Task 5: overlaid average grayscale histogram, one line per class."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for cls, data in hist_by_class.items():
        ax.plot(data["gray"], label=str(cls))
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Average density")
    ax.set_title("Average grayscale histogram per class")
    ax.legend()
    return ax


def plot_avg_rgb_by_class(hist_by_class: dict):
    """Task 5: one subplot per RGB channel, one line per class in each."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, ch in zip(axes, "RGB"):
        for cls, data in hist_by_class.items():
            ax.plot(data["rgb"][ch], label=str(cls))
        ax.set_title(f"{ch} channel")
        ax.set_xlabel("Pixel intensity")
    axes[0].set_ylabel("Average density")
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig, axes


# --------------------------------------------------------------------------
# Task 6: per-image color summary statistics
# --------------------------------------------------------------------------

def color_summary_stats(
    df: pd.DataFrame,
    img_dir,
    target_col: str = "diagnosis_1",
    n_per_class: int = 25,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Task 6. Compute mean and std of each RGB channel per image, for a
    consistent sample per class. One row per image.
    """
    sample = sample_per_class(df, target_col, n_per_class, seed)
    rows = []
    for _, r in sample.iterrows():
        arr = _load_rgb_array(img_dir, r["isic_id"]).astype(np.float32)
        flat = arr.reshape(-1, 3)
        rows.append(
            {
                "isic_id": r["isic_id"],
                target_col: r[target_col],
                "R_mean": flat[:, 0].mean(),
                "G_mean": flat[:, 1].mean(),
                "B_mean": flat[:, 2].mean(),
                "R_std": flat[:, 0].std(),
                "G_std": flat[:, 1].std(),
                "B_std": flat[:, 2].std(),
            }
        )
    return pd.DataFrame(rows)


def plot_color_stats_boxplots(stats_df: pd.DataFrame, target_col: str = "diagnosis_1"):
    """
    Task 6/7. One boxplot subplot per RGB-mean column, grouped by class,
    so you can visually compare brightness/color profile across classes.
    """
    import matplotlib.pyplot as plt

    cols = ["R_mean", "G_mean", "B_mean"]
    fig, axes = plt.subplots(1, len(cols), figsize=(15, 4))
    for ax, col in zip(axes, cols):
        stats_df.boxplot(column=col, by=target_col, ax=ax)
        ax.set_title(col)
        ax.figure.suptitle("")
    fig.tight_layout()
    return fig, axes
