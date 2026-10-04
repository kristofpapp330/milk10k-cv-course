"""
B5 -- Leak-free train / val / test split at LESION level (cleaned-up version of A3.2).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def stratified_group_folds(table, label_col, n_splits, seed):
    """StratifiedGroupKFold folds as (train_positions, val_positions), grouped by lesion_id.

    The rows are shuffled with `seed` first and the splitter runs with shuffle=False,
    because scikit-learn < 1.8 balanced the classes badly with its own shuffle=True (see Part A, A3.3).
    """
    order = np.random.default_rng(seed).permutation(len(table))
    shuffled = table.iloc[order]
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    return [(order[tr], order[va]) for tr, va in sgkf.split(shuffled, shuffled[label_col], groups=shuffled["lesion_id"])]


def split_lesions(lesions, val_size=0.2, test_size=0.2, seed=42, label_col="dx"):
    """Split a lesion table into three lists of lesion_id: (train, val, test).

    1. one StratifiedGroupKFold fold is taken as test  (test_size=0.2 -> 5 folds)
    2. one fold of the remaining lesions is taken as val (0.2 of all = 1/4 of the rest)
    Stratified on `label_col`, grouped by lesion_id, reproducible for a given seed.
    """
    rest_pos, test_pos = stratified_group_folds(lesions, label_col, round(1 / test_size), seed)[0]
    rest = lesions.iloc[rest_pos]
    train_pos, val_pos = stratified_group_folds(rest, label_col, round((1 - test_size) / val_size), seed)[0]
    return (rest.iloc[train_pos]["lesion_id"].tolist(),
            rest.iloc[val_pos]["lesion_id"].tolist(),
            lesions.iloc[test_pos]["lesion_id"].tolist())


def assign_images(meta, train_ids, val_ids, test_ids):
    """Images are assigned AFTER the lesion split: every image gets the split of its lesion."""
    split_of = {**{i: "train" for i in train_ids}, **{i: "val" for i in val_ids}, **{i: "test" for i in test_ids}}
    images = meta.copy()
    images["split"] = images["lesion_id"].map(split_of)
    assert images["split"].notna().all(), "some lesions were not assigned to any split"
    return images


def verify_splits(images, class_col="dx"):
    """Run and print the B5 checks. Returns (class_proportions_table, max_deviation_pp)."""
    ids = {s: set(images.loc[images["split"] == s, "lesion_id"]) for s in ["train", "val", "test"]}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = len(ids[a] & ids[b])
        print(f"  lesion overlap {a}-{b}: {overlap}")
        assert overlap == 0

    per_lesion = images.groupby(["split", "lesion_id"]).agg(n=("isic_id", "size"), n_views=("view", "nunique"))
    print(f"  lesions with exactly 2 images (1 dermoscopic + 1 clinical) in their split: "
          f"{((per_lesion['n'] == 2) & (per_lesion['n_views'] == 2)).sum()} of {len(per_lesion)}")
    assert (per_lesion["n"] == 2).all() and (per_lesion["n_views"] == 2).all()

    lesions = images.drop_duplicates("lesion_id")
    props = pd.crosstab(lesions[class_col], lesions["split"], normalize="columns") * 100
    props["overall"] = lesions[class_col].value_counts(normalize=True) * 100
    props = props[["overall", "train", "val", "test"]]
    max_dev = (props[["train", "val", "test"]].sub(props["overall"], axis=0)).abs().max().max()
    return props.round(2), round(float(max_dev), 3)
