"""
MILK10k — Exploratory Data Analysis (Session 1 homework)

Run this top to bottom in VS Code (with the Jupyter extension, using the
'# %%' cell markers), in Jupyter, or in Google Colab.

BEFORE RUNNING: download and unzip the dataset so you end up with a
'milk10k/' folder containing 'metadata.csv', 'images/', and
'supplements/training_gt.csv'.

  - In Google Colab, run this in a cell:
      !wget https://isic-archive.s3.amazonaws.com/dois/10.34970-648456/milk10k.zip
      !unzip milk10k.zip -d milk10k

  - Locally (Mac/Linux terminal), you can run the same two commands
    (drop the leading '!'), or download the .zip in your browser from
    https://api.isic-archive.com/doi/milk10k/ and unzip it into this
    project's data/ folder.
"""

# %% [1] Imports -------------------------------------------------------
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# %% [2] Paths -----------------------------------------------------------
# Point this at wherever you unzipped the dataset.
DATA_DIR = Path("../data/milk10k")   # e.g. project/data/milk10k
IMG_DIR = DATA_DIR / "images"

# %% [3] Load the two CSVs ------------------------------------------------
# metadata.csv  -> one row PER IMAGE
# training_gt.csv -> one row PER LESION (one-hot columns for the 11 classes)
df = pd.read_csv(DATA_DIR / "metadata.csv")
gt = pd.read_csv(DATA_DIR / "supplements" / "training_gt.csv")

print("metadata.csv shape:", df.shape)
print("training_gt.csv shape:", gt.shape)
print(df["lesion_id"].nunique(), "unique lesions")

# %% [4] Explore the metadata --------------------------------------------
print("\nCoarse diagnosis counts:")
print(df["diagnosis_1"].value_counts())

print("\nMissing values (fraction) in key columns:")
print(df[["age_approx", "sex", "anatom_site_general"]].isna().mean())

# Confirm the 2-images-per-lesion structure mentioned in class
print("\nEvery lesion has exactly 2 images:",
      (df.groupby("lesion_id").size() == 2).all())

# %% [5] Class distribution (11 fine-grained classes) ---------------------
counts = gt.drop(columns="lesion_id").sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
counts.plot(kind="bar", ax=ax, color="#F08200")
ax.set_ylabel("Number of lesions")
ax.set_title("MILK10k — class distribution (11 classes)")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.show()

# %% [6] Load and display one real image ----------------------------------
row = df.iloc[0]
img_path = IMG_DIR / f'{row["isic_id"]}.jpg'
img = Image.open(img_path).convert("RGB")

plt.imshow(img)
sub_label = row["diagnosis_2"] if pd.notna(row["diagnosis_2"]) else ""
plt.title(f'{row["diagnosis_1"]} — {sub_label}')
plt.axis("off")
plt.show()

# %% [7] Inspect the image as a NumPy array -------------------------------
arr = np.array(img)
print("shape:", arr.shape)     # (height, width, channels)
print("dtype:", arr.dtype)     # usually uint8
print("min / max pixel value:", arr.min(), arr.max())

# %% [8] Reusable image-grid viewer ---------------------------------------
def show_samples(df, img_dir, n=6, diagnosis=None, seed=0):
    """Show n random sample images, optionally filtered by diagnosis_1."""
    subset = df if diagnosis is None else df[df["diagnosis_1"] == diagnosis]
    sample = subset.sample(n=n, random_state=seed)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3))
    for ax, (_, r) in zip(axes, sample.iterrows()):
        im = Image.open(img_dir / f'{r["isic_id"]}.jpg')
        ax.imshow(im)
        ax.set_title(r["diagnosis_1"], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

show_samples(df, IMG_DIR, n=6, diagnosis="Malignant")
show_samples(df, IMG_DIR, n=6, diagnosis="Benign")

# %% [9] Two images, one lesion --------------------------------------------
# Each lesion has ONE dermoscopic image and ONE clinical close-up.
# This is why any train/val/test split must group by lesion_id.
lesion = df["lesion_id"].value_counts().index[0]
pair = df[df["lesion_id"] == lesion]

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
for ax, (_, r) in zip(axes, pair.iterrows()):
    im = Image.open(IMG_DIR / f'{r["isic_id"]}.jpg')
    ax.imshow(im)
    ax.set_title(r["image_type"], fontsize=9)
    ax.axis("off")
plt.show()

# %% [10] Quick image-size statistics over a sample ------------------------
widths, heights = [], []
for _, r in df.sample(200, random_state=0).iterrows():
    with Image.open(IMG_DIR / f'{r["isic_id"]}.jpg') as im:
        w, h = im.size
    widths.append(w)
    heights.append(h)

print("avg width:", np.mean(widths))
print("avg height:", np.mean(heights))

# %% [11] Summary printed to console ---------------------------------------
print("\n--- EDA SUMMARY ---")
print(f"Total images: {len(df)}")
print(f"Total unique lesions: {df['lesion_id'].nunique()}")
print(f"Coarse diagnosis breakdown:\n{df['diagnosis_1'].value_counts()}")
print(f"Most common fine-grained class: {counts.idxmax()} ({counts.max()} lesions)")
print(f"Rarest fine-grained class: {counts.idxmin()} ({counts.min()} lesions)")
