"""
Milestone 4 -- fairness check by skin tone (runs on a laptop in seconds, no GPU, no images).

The metadata.csv of the MILK10k download has no skin-tone column; the skin tone class is in the
challenge's training metadata (challenge.isic-archive.com/data, "Training Metadata"). This script joins
it with the saved test predictions of the Milestone 4 model and reports macro-F1 per skin tone group.

    python run_fairness_skin_tone.py --metadata path/to/MILK10k_Training_Metadata.csv

Output: outputs/deep/m4_fairness_skin_tone_class.csv (+ the table printed).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from milk10k_pipeline import evaluation as ev   # noqa: E402

CLASSES3 = ["Benign", "Malignant", "Indeterminate"]


def find_skin_column(meta):
    """The skin tone column, whatever its exact name (e.g. 'skin_tone_class')."""
    candidates = [c for c in meta.columns if any(k in c.lower() for k in ("skin_tone", "skin tone", "fitzpatrick", "monk"))]
    if not candidates:
        raise SystemExit(f"no skin tone column found. Columns are: {list(meta.columns)}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="challenge training metadata CSV with the skin tone column")
    parser.add_argument("--predictions", default=str(REPO / "outputs" / "deep" / "m4_test_predictions.csv"))
    args = parser.parse_args()

    pred = pd.read_csv(args.predictions)
    meta = pd.read_csv(args.metadata)
    col = find_skin_column(meta)
    print(f"skin tone column: '{col}' | values: {sorted(meta[col].dropna().unique().tolist())}")

    # join on the image id if the file has one, otherwise on the lesion id (same value for both images)
    key = "isic_id" if "isic_id" in meta and pred["isic_id"].isin(meta["isic_id"]).mean() > 0.9 else "lesion_id"
    tone = meta.drop_duplicates(key).set_index(key)[col]
    groups = pred[key].map(tone)
    print(f"joined on {key}: skin tone known for {groups.notna().sum()} of {len(pred)} test lesions")

    y = pred["diagnosis_1"].map({c: i for i, c in enumerate(CLASSES3)}).to_numpy()
    p = pred["pred"].map({c: i for i, c in enumerate(CLASSES3)}).to_numpy()
    rep = ev.group_report(y, p, groups.to_numpy(), min_n=50)
    rep["share of test lesions"] = (rep["n lesions"] / len(pred)).round(3)
    out = REPO / "outputs" / "deep" / "m4_fairness_skin_tone_class.csv"
    rep.to_csv(out, index=False)
    print(rep.round(3).to_string(index=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
