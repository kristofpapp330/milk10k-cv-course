"""
Loading the raw MILK10k tables and locating image files.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config


def load_tables(data_dir=None):
    """Load metadata.csv (one row per image) and training_gt.csv (one row per lesion).

    Adds to the metadata:
      - dx    : the 11-class label (from the one-hot columns of training_gt.csv)
      - view  : "dermoscopic" or "clinical" (short version of image_type)
    Returns (meta, gt, class_names).
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    meta = pd.read_csv(data_dir / "metadata.csv")
    gt = pd.read_csv(data_dir / "supplements" / "training_gt.csv")
    class_names = [c for c in gt.columns if c != "lesion_id"]
    gt["dx"] = gt[class_names].idxmax(axis=1)
    meta = meta.merge(gt[["lesion_id", "dx"]], on="lesion_id", how="left")
    meta["view"] = np.where(meta["image_type"].str.contains("dermoscop", case=False), "dermoscopic", "clinical")
    return meta, gt, class_names


def find_images(img_dir=None):
    """Return {isic_id: path} for every .jpg under img_dir (searched once, also in sub-folders)."""
    img_dir = Path(img_dir or config.IMG_DIR)
    return {p.stem: p for p in img_dir.rglob("*.jpg")}


def load_rgb(path, min_size=None):
    """Open an image as RGB. Raises FileNotFoundError with a clear message if it is missing.

    If min_size is given, the JPEG is decoded directly at a reduced scale
    (Pillow 'draft' mode) that is never smaller than min_size -- faster, same result after resizing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    img = Image.open(path)
    if min_size:
        img.draft("RGB", (min_size, min_size))
    return img.convert("RGB")
