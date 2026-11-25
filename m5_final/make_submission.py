"""
Milestone 5 -- the official MILK10k leaderboard file (Session 10).

    python m5_final/make_submission.py --test-dir MILK10k_Test_Input --metadata MILK10k_Test_Metadata.csv

Output: m5_final/submission.csv, one row per lesion (479), columns: lesion, AKIEC, BCC, ..., VASC.

Steps
  1. keep the dermoscopic image of every test lesion (the model was trained on dermoscopic images only)
  2. prepare it exactly like the validation images (prepare_image + VAL_TRANSFORM), softmax -> 11 probabilities
  3. if the checkpoint says so, rescale the probabilities so that the per-class thresholds tuned on
     VALIDATION sit at 0.5 (the official metric binarises at 0.5); otherwise keep the raw softmax
  4. the 4 checks of the brief; if they pass, the file is ready to upload
The test images are only used to predict: no labels exist, nothing is trained or tuned on them.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from milk10k_pipeline.deep import CLASSES11, VAL_TRANSFORM, load_checkpoint, prepare_image   # noqa: E402
from milk10k_pipeline.evaluation import rescale_to_half                                         # noqa: E402

EXPECTED_LESIONS = 479


def check_submission(sub, n_expected=EXPECTED_LESIONS):
    """The 4 lines of the brief. Raises AssertionError if the file would be rejected."""
    assert list(sub.columns) == ["lesion"] + CLASSES11, "wrong columns or wrong column order"
    assert len(sub) == n_expected and sub.lesion.is_unique, f"need {n_expected} unique lesions, got {len(sub)}"
    assert sub[CLASSES11].isna().sum().sum() == 0, "empty cells"
    assert ((sub[CLASSES11] >= 0) & (sub[CLASSES11] <= 1)).all().all(), "values outside [0, 1]"


@torch.no_grad()
def make_submission(test_dir, metadata_csv, checkpoint, out_csv, use_thresholds=None, batch_size=32,
                    n_expected=EXPECTED_LESIONS):
    model, ckpt = load_checkpoint(checkpoint, device="cpu")
    assert ckpt["classes"] == CLASSES11, "this is not the 11-class model"
    meta = pd.read_csv(metadata_csv)
    derm = meta[meta["image_type"].str.contains("dermoscop", case=False)].drop_duplicates("lesion_id")
    paths = {p.stem: p for p in Path(test_dir).rglob("*.jpg")}            # works with or without sub-folders
    missing = [i for i in derm["isic_id"] if i not in paths]
    assert not missing, f"{len(missing)} dermoscopic test images not found, e.g. {missing[:3]}"

    probs = []
    ids = derm["isic_id"].tolist()
    for start in range(0, len(ids), batch_size):
        x = torch.stack([VAL_TRANSFORM(prepare_image(paths[i])) for i in ids[start:start + batch_size]])
        probs.append(torch.softmax(model(x), dim=1).numpy())
        print(f"  {min(start + batch_size, len(ids))}/{len(ids)} lesions", end="\r")
    probs = np.concatenate(probs)

    if use_thresholds is None:
        use_thresholds = bool(ckpt.get("use_thresholds", False))
    if use_thresholds:
        probs = rescale_to_half(probs, ckpt["thresholds"])
        print("\nprobabilities rescaled with the per-class thresholds tuned on validation")
    else:
        print("\nraw softmax probabilities")

    sub = pd.DataFrame(probs, columns=CLASSES11)
    sub.insert(0, "lesion", derm["lesion_id"].to_numpy())
    check_submission(sub, n_expected)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print(f"all 4 checks passed -> {out_csv} ({len(sub)} lesions) is ready to upload")
    return sub


def main():
    parser = argparse.ArgumentParser(description="Create the MILK10k leaderboard submission.csv")
    parser.add_argument("--test-dir", default="MILK10k_Test_Input")
    parser.add_argument("--metadata", default="MILK10k_Test_Metadata.csv")
    parser.add_argument("--checkpoint", default=str(REPO / "m5_final" / "model_11class.pt"))
    parser.add_argument("--out", default=str(REPO / "m5_final" / "submission.csv"))
    parser.add_argument("--raw", action="store_true", help="ignore the tuned thresholds, submit raw softmax")
    args = parser.parse_args()
    make_submission(args.test_dir, args.metadata, args.checkpoint, args.out,
                    use_thresholds=False if args.raw else None)


if __name__ == "__main__":
    main()
