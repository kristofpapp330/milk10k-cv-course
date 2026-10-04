"""
Project configuration: the ONE place for paths, seed and image settings.

Every other module and script imports from here, so nothing else hard-codes
a path. The data folder defaults to <repo>/data/milk10k and can be changed
without touching code by setting the environment variable MILK10K_DIR, e.g.

    Windows PowerShell:  $env:MILK10K_DIR = "D:/datasets/milk10k"
    macOS / Linux:       export MILK10K_DIR=/data/milk10k
"""
import os
from pathlib import Path

# --- paths (all relative to the repository root, never absolute) ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("MILK10K_DIR", REPO_ROOT / "data" / "milk10k"))
IMG_DIR = DATA_DIR / "images"
METADATA_CSV = DATA_DIR / "metadata.csv"
GT_CSV = DATA_DIR / "supplements" / "training_gt.csv"

OUTPUT_DIR = REPO_ROOT / "outputs"          # everything generated goes here
FIGURES_DIR = OUTPUT_DIR / "figures"
SPLITS_DIR = OUTPUT_DIR / "splits"

# --- reproducibility --------------------------------------------------------
SEED = 42

# --- splits -------------------------------------------------------------------
VAL_SIZE = 0.20      # fraction of LESIONS
TEST_SIZE = 0.20     # fraction of LESIONS  (train = the remaining 0.60)
SPLIT_LABEL = "dx"   # stratify on the 11-class label

# --- images -----------------------------------------------------------------
VIEWS = ("dermoscopic",)   # B6 option (b): use the dermoscopic image of each lesion
RESIZE = 256               # shorter side is resized to this ...
IMAGE_SIZE = 224           # ... then a 224x224 crop is taken

# --- data loading -----------------------------------------------------------
BATCH_SIZE = 32
NUM_WORKERS = 0            # 0 = load in the main process: works everywhere (Windows, notebooks)
