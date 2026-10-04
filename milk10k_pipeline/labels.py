"""
B3 -- Label strategy, and the lesion-level table (A1.3).
"""
import json

import pandas as pd

# ---------------------------------------------------------------------------
# Primary target: diagnosis_1, with "Indeterminate" MERGED into "Malignant".
#   - only 123 of 5,240 lesions (2.3%) are Indeterminate -> too few for a reliable 3rd class
#   - all of them are AKIEC, a class that is otherwise Malignant -> not separable from the image
#   - clinically, an indeterminate lesion needs the same action as a malignant one (work-up / biopsy),
#     so merging it into the "needs action" class is the safe side (see the A3.4 cost matrix)
# ---------------------------------------------------------------------------
PRIMARY = {
    "source_column": "diagnosis_1",
    "class_names": ["Benign", "Malignant"],
    "mapping": {"Benign": 0, "Malignant": 1, "Indeterminate": 1},
    "decision": "Indeterminate merged into Malignant (= 'needs clinical action'); 2-class task.",
}

# ---------------------------------------------------------------------------
# Stretch goal: the 11 classes, ALL kept (no merging); rare classes are handled with class weights.
# Merging the rare classes into one "other" group would mix benign (DF, INF, VASC, BEN_OTH) and
# malignant (MAL_OTH) lesions, which is clinically meaningless.
# ---------------------------------------------------------------------------
ELEVEN_CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
STRETCH = {
    "source_column": "dx",
    "class_names": ELEVEN_CLASSES,
    "mapping": {name: i for i, name in enumerate(ELEVEN_CLASSES)},
    "decision": "All 11 classes kept; rare classes (<~50 lesions) handled with class weights; "
                "MAL_OTH (9 lesions) is reported with a warning because its metrics are very noisy.",
}


def label_map():
    """The full label map that is saved to label_map.json and loaded by every later milestone."""
    return {"primary": PRIMARY, "stretch_11_class": STRETCH}


def save_label_map(path):
    with open(path, "w") as f:
        json.dump(label_map(), f, indent=2)


def load_label_map(path):
    with open(path) as f:
        return json.load(f)


def build_lesion_table(meta):
    """One row per lesion with derm_id / clinical_id side by side (pivot, no loops) -- same as A1.3."""
    wide = meta.pivot(index="lesion_id", columns="view", values="isic_id")
    wide = wide.rename(columns={"dermoscopic": "derm_id", "clinical": "clinical_id"})
    wide.columns.name = None
    info = meta.groupby("lesion_id")[["diagnosis_1", "dx", "age_approx", "sex", "anatom_site_general"]].first()
    lesions = wide.join(info).reset_index()
    lesions["label"] = lesions["diagnosis_1"].map(PRIMARY["mapping"])
    return lesions
