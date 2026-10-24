"""
Tests for the Milestone 2 classical baseline (features + models). No dataset needed.

The first tests check that the code gives EXACTLY the numbers computed by hand on the
Session 5 slides for the 5x5 toy lesion.

Run from the repository root with either:
    python tests/test_features.py
    python -m pytest tests
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make the package importable

from milk10k_pipeline import classical, features

# The toy lesion of Sessions 4 and 5: skin = 100, lesion = 10 in the bottom-right corner.
TOY = np.array([[100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100],
                [100, 100, 10, 10, 10],
                [100, 100, 10, 10, 10],
                [100, 100, 10, 10, 10]], dtype=float)


def test_toy_histogram_matches_slide():
    """Slide 'Color histogram by hand': 4 bins -> [0.36, 0.64, 0, 0]."""
    rgb = np.stack([TOY] * 3, axis=-1)
    h = features.color_histogram(rgb, bins=4)
    assert np.allclose([h["hist_R0"], h["hist_R1"], h["hist_R2"], h["hist_R3"]], [0.36, 0.64, 0, 0])


def test_toy_mean_std_match_slide():
    """Slide 'Mean and standard deviation by hand': mu = 67.6, sigma = 43.2."""
    s = features.color_stats(np.stack([TOY] * 3, axis=-1))
    assert round(s["mean_R"], 1) == 67.6 and round(s["std_R"], 1) == 43.2


def test_toy_edge_density_matches_slide():
    """Slide 'Edge density by hand': mean Sobel magnitude of the 3x3 interior = 279.8."""
    mag = features.sobel_magnitude(TOY)
    assert np.allclose(mag[1:-1, 1:-1].round(1), [[127.3, 284.6, 360], [284.6, 381.8, 360], [360, 360, 0]])
    assert round(mag[1:-1, 1:-1].mean(), 1) == 279.8


def test_texture_sees_where_pixels_are():
    """Slide 'Same histogram, different image': shuffling pixels keeps colour features, changes texture."""
    rng = np.random.default_rng(0)
    img = np.kron(TOY, np.ones((20, 20)))                       # 100 x 100 version of the toy
    shuffled = rng.permutation(img.ravel()).reshape(img.shape)
    assert np.isclose(img.mean(), shuffled.mean()) and np.isclose(img.std(), shuffled.std())
    t1, t2 = features.texture_features(img), features.texture_features(shuffled)
    assert t2["edge_density"] > 5 * t1["edge_density"]
    g1, g2 = features.glcm_features(img), features.glcm_features(shuffled)
    assert g2["glcm_contrast"] > g1["glcm_contrast"] and g2["glcm_homogeneity"] < g1["glcm_homogeneity"]


def synthetic_lesion_image(irregular, seed, size=256):
    """Light skin with a dark blob in the middle; 'irregular' = notched border and patchy colour."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    angle = np.arctan2(yy - size / 2, xx - size / 2)
    radius = size * 0.25 * (1 + (0.25 * np.sin(7 * angle + rng.uniform(0, 6)) if irregular else 0))
    mask = np.hypot(yy - size / 2, xx - size / 2) < radius
    img = np.zeros((size, size, 3)) + [205, 165, 150]
    img[mask] = [110, 75, 60]
    if irregular:
        img[mask & (rng.random((size, size)) < 0.3)] = [60, 40, 35]
    img += rng.normal(0, 5, img.shape)
    return np.clip(img, 0, 255)


def test_feature_vector_shape_and_mask():
    """45 features, no NaN on a clear lesion, and the mask finds a compact round blob."""
    f = features.extract_features(synthetic_lesion_image(False, 0))
    assert len(f) == 45
    assert not np.isnan(list(f.values())).any()
    assert 0.12 < f["mask_area"] < 0.28                     # disc of radius 0.25*size covers ~0.196
    assert f["mask_compactness"] < 1.6                      # close to a circle
    irregular = features.extract_features(synthetic_lesion_image(True, 0))
    assert irregular["mask_compactness"] > f["mask_compactness"]
    groups = features.feature_groups(f.keys())
    assert [len(groups[g]) for g in ["color_stats", "color_hist", "texture", "glcm", "abcd"]] == [6, 24, 3, 4, 8]
    for name in features.FEATURE_SETS:                       # the number in each set's name is its size
        assert len(features.feature_set_columns(name, f.keys())) == int(name.split("(")[1].rstrip(")"))


def test_models_learn_and_never_see_test():
    """On separable synthetic features every real model beats the majority class on validation."""
    rng = np.random.default_rng(0)
    rows, labels = [], []
    for k in range(240):
        irregular = k % 3 == 0
        rows.append(features.extract_features(synthetic_lesion_image(irregular, k, size=96)))
        labels.append(int(irregular))
    feats = pd.DataFrame(rows, index=[f"ISIC_{k}" for k in range(240)])
    table = pd.DataFrame({"isic_id": feats.index, "label": labels})
    train, val = table.iloc[:160], table.iloc[160:]
    results = classical.run_grid(feats, train, val, "label", ["Benign", "Malignant"],
                                 feature_sets=["color+texture (33)"], verbose=False)
    best = classical.best_per_cell(results).set_index("model")["macro_f1"]
    assert best["majority"] < 0.5
    for m in ["logistic_reg", "knn", "random_forest", "svm_rbf"]:
        assert best[m] > 0.9, (m, best[m])


def test_threshold_for_sensitivity():
    y = np.array([1] * 10 + [0] * 10)
    score = np.r_[np.linspace(0.3, 1, 10), np.linspace(0, 0.5, 10)]
    thr = classical.threshold_for_sensitivity(y, score, target=0.9)
    assert ((score >= thr) & (y == 1)).sum() >= 9


if __name__ == "__main__":
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASSED  {test.__name__}")
    print(f"\nall {len(tests)} tests passed")
