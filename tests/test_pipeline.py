"""
Tests for the Milestone 1 pipeline. They use small synthetic data, so they run without the dataset.

Run from the repository root with either:
    python tests/test_pipeline.py
    python -m pytest tests
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make the package importable

from milk10k_pipeline import labels, splits, transforms
from milk10k_pipeline.data_loader import MILK10kDataLoader, MILK10kDataset, compute_class_weights


def synthetic_lesions(n=2000, seed=0):
    """A fake lesion table with an imbalanced 11-class label (same shape as the real one)."""
    rng = np.random.default_rng(seed)
    probs = np.array([6, 48, 1, 10, 1, 1, 0.5, 9, 14, 9, 1], dtype=float)
    dx = rng.choice(labels.ELEVEN_CLASSES, size=n, p=probs / probs.sum())
    return pd.DataFrame({"lesion_id": [f"L{i:05d}" for i in range(n)], "dx": dx})


def test_split_properties():
    """No overlap, sizes within +-1 percentage point, same seed -> same output (10 seeds)."""
    lesions = synthetic_lesions()
    for seed in range(10):
        tr, va, te = splits.split_lesions(lesions, 0.2, 0.2, seed)
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
        assert len(tr) + len(va) + len(te) == len(lesions)
        assert abs(len(va) / len(lesions) - 0.2) <= 0.01
        assert abs(len(te) / len(lesions) - 0.2) <= 0.01
        assert splits.split_lesions(lesions, 0.2, 0.2, seed) == (tr, va, te)


def test_eval_transform_is_deterministic():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (450, 600, 3), dtype=np.uint8))
    tf = transforms.build_eval_transform()
    assert torch.equal(tf(img), tf(img))


def test_train_transform_shape():
    img = Image.fromarray(np.zeros((450, 600, 3), dtype=np.uint8))
    assert tuple(transforms.build_train_transform()(img).shape) == (3, 224, 224)


def test_class_weights():
    w = compute_class_weights([0, 0, 0, 1], 2)      # 3 vs 1 -> rare class gets 3x the weight
    assert np.isclose(w[1] / w[0], 3.0)


def test_label_map_merges_indeterminate():
    m = labels.label_map()["primary"]["mapping"]
    assert m["Indeterminate"] == m["Malignant"] == 1 and m["Benign"] == 0


def test_missing_file_fails_loudly():
    """Both the new Dataset and the old Session 2 loader must raise on a missing image."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "images").mkdir()
        Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8)).save(tmp / "images" / "ISIC_1.jpg")
        table = pd.DataFrame({"isic_id": ["ISIC_1", "ISIC_MISSING"], "view": ["dermoscopic"] * 2,
                              "diagnosis_1": ["Benign", "Malignant"]})
        table.to_csv(tmp / "split.csv", index=False)

        for build in [lambda: MILK10kDataset(tmp / "split.csv", img_dir=tmp / "images", label_map=labels.label_map()),
                      lambda: MILK10kDataLoader(table, tmp / "images")]:
            try:
                build()
                raise AssertionError("a missing file did not raise an error")
            except FileNotFoundError:
                pass
        # explicit opt-in keeps working
        ds = MILK10kDataset(tmp / "split.csv", img_dir=tmp / "images", label_map=labels.label_map(), allow_missing=True)
        assert len(ds) == 1


if __name__ == "__main__":
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASSED  {test.__name__}")
    print(f"\nall {len(tests)} tests passed")
