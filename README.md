# MILK10k skin-lesion classification — course project

Computer Vision and Speech Recognition (EADA). Author: Kristof Papp.

## 1. What this project is

**Task.** Classify a skin lesion from its **dermoscopic image** as **Benign** or **Malignant**
(`diagnosis_1`, with *Indeterminate* merged into Malignant, see [label strategy](#5-key-decisions-so-far)).
Stretch goal: the 11 diagnostic classes (AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC).
The model is meant as decision support for a clinician deciding whether a lesion needs work-up / biopsy.

**Dataset.** [MILK10k](https://api.isic-archive.com/doi/milk10k/) from the ISIC Archive (DOI 10.34970/648456):
5,240 lesions, each with 2 images (1 dermoscopic + 1 clinical close-up) = 10,480 JPEG images, plus
`metadata.csv` (one row per image) and `supplements/training_gt.csv` (one-hot 11-class label per lesion).
License: **CC BY-NC** (non-commercial use with attribution). Attribution: MILK10k / ISIC Archive;
described in *"MILK10k: A Hierarchical Multimodal Imaging-Learning Toolkit for Diagnosing Pigmented and
Nonpigmented Skin Cancer and its Simulators"*, Journal of Investigative Dermatology, 2025, doi:10.1016/j.jid.2025.06.1594.

**Goal of Milestone 1.** A leak-free, reproducible data pipeline: integrity check, label strategy,
data-quality report, lesion-level splits, transforms, and the Dataset/DataLoaders that later milestones
will train on. It is an evolution of the Session 2 code (`milk10k_pipeline/`), not a rewrite.

## 2. Setup and how to run

- **Python 3.10** or newer.
- Install the packages (from the repository root):
  ```
  python -m pip install -r requirements.txt
  ```
- **Data.** Download MILK10k and unzip it so that these exist:
  `data/milk10k/metadata.csv`, `data/milk10k/supplements/training_gt.csv`, `data/milk10k/images/*.jpg`.
  The path is set in ONE place, [`milk10k_pipeline/config.py`](milk10k_pipeline/config.py). To use another
  folder without editing code, set the environment variable `MILK10K_DIR` (PowerShell:
  `$env:MILK10K_DIR = "D:/datasets/milk10k"`; macOS/Linux: `export MILK10K_DIR=/data/milk10k`).

Commands (run from the repository root):

| what | command |
|---|---|
| tests (no dataset needed) | `python tests/test_pipeline.py` |
| full Milestone 1 pipeline: integrity check, EDA figures, label map, quality report, splits, normalization stats, transforms check, class weights, DataLoader sanity checks | `python run_milestone1.py` |
| Part A homework | open `notebooks/homework_part_a.ipynb` → Restart & Run All |
| Session 2 analysis (metadata + colour figures) | `python run_analysis.py` |

`run_milestone1.py` takes a few minutes (it verifies all 10,480 images and loads one training epoch) and
writes everything to `outputs/`, including `outputs/run_log.txt` with all printed checks.

## 3. Repository structure

```
milk10k-cv-course/
├── README.md                     this file
├── SUBMISSION.md                 direct links to every deliverable
├── requirements.txt              Python packages
├── .gitignore                    keeps the raw data (data/) out of git
├── run_milestone1.py             runs the whole Milestone 1 pipeline (B1–B8) and writes outputs/
├── run_analysis.py               Session 2 analysis script (metadata + colour figures)
├── milk10k_pipeline/             reusable, importable project code
│   ├── __init__.py               package index (lists every module)
│   ├── config.py                 the ONLY place for paths, seed, split sizes, image size, batch size
│   ├── data_io.py                loads the CSVs, finds image files, opens images (clear error if missing)
│   ├── metadata_analysis.py      Session 2: metadata vs target statistics
│   ├── color_analysis.py         Session 2: colour histograms/statistics per class (reused in B2)
│   ├── preprocessing.py          Session 2: NumPy preprocessing (now fails loudly on missing files)
│   ├── data_loader.py            Session 2 loader (now fails loudly) + B8 PyTorch Dataset/DataLoaders,
│   │                             class weights, WeightedRandomSampler, train-only normalization stats
│   ├── visualizer.py             Session 2 plots + 3x4 class gallery + transformed-batch viewer
│   ├── integrity.py              B1: existence + Image.verify() of every image, size summary
│   ├── labels.py                 B3: label strategy, label_map.json, lesion-level table
│   ├── quality.py                B4: data-quality report (missing values, pair checks, shortcuts, banned columns)
│   ├── splits.py                 B5: lesion-level train/val/test split and its verification
│   └── transforms.py             B7: train_transform / eval_transform + augmentation justification table
├── tests/
│   └── test_pipeline.py          split properties (10 seeds), deterministic eval transform, fail-loudly, class weights
├── notebooks/
│   ├── 01_eda.ipynb              Session 1 exploration
│   ├── homework_part_a.ipynb     Part A exercises (Sessions 1–3)
│   └── homework_part_a.pdf       Part A exported to PDF
├── src/
│   └── eda.py                    Session 1 EDA script
├── reports/
│   └── milestone1_report.md      B9 report (max 2 pages)
├── outputs/                      GENERATED files only (re-created by the scripts, never edited by hand)
│   ├── splits/                   train.csv, val.csv, test.csv (one row per image) + split_info.json (seed, date)
│   ├── figures/                  class distribution, 11→diagnosis_1 mapping, class gallery,
│   │                             augmentation examples, transformed training batch
│   ├── label_map.json            B3 label mapping, loaded by every later milestone
│   ├── class_weights.json        B8 loss weights computed on the train split
│   ├── normalization_stats.json  per-channel mean/std computed on the train split
│   ├── image_size_summary.csv    B1 min/median/max width and height (+ image_sizes.csv per image)
│   ├── data_quality_report.md    B4 report
│   ├── session2_findings.csv     B2 "Session 2 finding → still true? → consequence" table
│   ├── split_class_proportions.csv, dx_vs_diagnosis1.csv   tables behind the B5/B2 checks
│   ├── lesions.csv               lesion-level table from Part A (A1.3)
│   └── run_log.txt, milestone1_summary.json   full printed output and key numbers of the last run
└── data/                         NOT in git — the MILK10k download goes here
```

**Why this organisation.** Logic that later milestones will reuse (splitting, transforms, Dataset, label map)
lives in the importable package `milk10k_pipeline/`, one module per responsibility, so it is written once
and imported everywhere instead of copy-pasted between notebooks. Notebooks are only for exploration and
the Part A exercises. Scripts at the root (`run_milestone1.py`) only *call* the package in order. Everything
the code generates goes to `outputs/`, separate from source code, so it can be deleted and regenerated at any
time. Configuration (paths, seed, sizes) is in a single file, so nothing is hard-coded twice.

## 4. Data handling rules

- **Not committed:** the raw data (`data/`, ~345 MB of images and CSVs) — see `.gitignore`.
- **Committed:** code, the split CSVs, JSON files, figures and reports in `outputs/` and `reports/` (all small).
- Generated files are written only to `outputs/`.
- **Seed:** `42` (in `config.py`; used for the split, sampler, augmentation and NumPy/PyTorch).
- **Splits created on:** 2026-10-04, seed 42, 60/20/20 % of lesions (also stored in `outputs/splits/split_info.json`).
- Splitting is done on **lesions**, never on images: both images of a lesion are always in the same split.
- No statistic is computed on val/test: normalization stats and class weights come from the train split only.

## 5. Key decisions so far

Details and numbers: [reports/milestone1_report.md](reports/milestone1_report.md).

- **Label strategy:** primary target `diagnosis_1`; the 123 *Indeterminate* lesions (2.3 %, all AKIEC) are merged
  into *Malignant* (= "needs clinical action") → 2 classes. Stretch goal keeps all 11 classes, rare ones
  handled with class weights. Saved in `outputs/label_map.json`.
- **Split design:** `StratifiedGroupKFold`, grouped by `lesion_id`, stratified on the 11-class label,
  60/20/20, seed 42 (`milk10k_pipeline/splits.py`). Zero lesion overlap is checked in code.
- **Views:** one dermoscopic image per lesion (option b) → one image = one lesion, no double counting.
- **Preprocessing:** resize shorter side to 256, crop 224×224, normalize with train-split mean/std;
  mild, medically justified augmentation in train only (flips, ±20° rotation, small crop shift,
  brightness 0.1 / hue 0.02) — see `milk10k_pipeline/transforms.py`.
- **Imbalance:** class-weighted loss weights (`outputs/class_weights.json`) and a `WeightedRandomSampler`
  for the training loader.
- **Banned inputs:** label columns, `diagnosis_confirm_type`, `image_manipulation`, contributor columns
  (see `outputs/data_quality_report.md`).
