# Computer Vision & Speech Recognition — Course Project (MILK10k)

## Clinical task

Skin cancer is one of the most common cancers worldwide, and early detection strongly improves
outcomes. Dermatologists often use **dermoscopic images** (close-up, magnified photos of skin
lesions taken with a special lens) to decide whether a lesion looks suspicious.

The goal of this project is to build a **computer vision system that looks at an image of a skin
lesion and predicts whether it is malignant or benign** (a screening/triage task), and later in
the course to go further into specific diagnosis categories (e.g. melanoma, basal cell carcinoma,
nevus, etc.).

This matters because:
- A **missed malignant lesion** (false negative) can delay a real cancer diagnosis.
- A **false positive** causes unnecessary anxiety and biopsies.
- Such a model could act as a **decision-support / pre-screening tool** for doctors, not a
  replacement for clinical judgment.

## Dataset — MILK10k

MILK10k is a real dataset built from the ISIC Archive (the same archive family behind HAM10000
and the ISIC challenges).

| Quantity | Value |
|---|---|
| Images | 10,480 |
| Unique lesions | 5,240 (2 images per lesion: 1 dermoscopic + 1 clinical close-up) |
| Diagnosis (coarse) | Malignant, Benign, Indeterminate |
| Diagnosis (fine, 11 classes) | BCC, NV, BKL, SCCKA, MEL, AKIEC, DF, INF, VASC, BEN_OTH, MAL_OTH |

Source: https://api.isic-archive.com/doi/milk10k/

**Important:** each lesion has two different images (a dermoscopic one and a clinical
close-up), linked by `lesion_id`. This matters for later modeling — you must split
train/val/test **by `lesion_id`**, never by image row, or the same lesion could leak across splits.

## Project structure

```
milk10k-cv-course/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # dataset lives here locally (NOT pushed to GitHub, see .gitignore)
├── notebooks/
│   └── 01_eda.ipynb       # exploratory data analysis (open in Jupyter or Colab)
└── src/
    └── eda.py             # same EDA code as a plain importable script
```

## How to run

1. Install dependencies: `pip install -r requirements.txt`
2. Download the MILK10k dataset (see instructions at the top of `notebooks/01_eda.ipynb`).
3. Open `notebooks/01_eda.ipynb` in VS Code (or Jupyter, or Google Colab) and run the cells top
   to bottom.

## Session 1 EDA — what this notebook covers

- Loading `metadata.csv` (per-image) and `training_gt.csv` (per-lesion) with pandas
- Basic sanity checks: shapes, missing values, confirming the 2-images-per-lesion structure
- Class distribution plot across the 11 diagnosis categories (severe class imbalance)
- Loading and displaying real images with PIL / matplotlib
- Inspecting an image as a NumPy array (shape, dtype, value range)
- A reusable `show_samples()` grid-viewer function
- Visualizing the two images (dermoscopic + clinical) that belong to the same lesion
- Basic image-size statistics (average width/height over a sample)
