"""
Tests for Milestones 3-5 (deep models, leaderboard metric, predict, submission). No dataset, no internet:
the ImageNet weights are replaced by random ones (MILK10K_OFFLINE_TEST=1), which is enough to check
shapes, parameter counts, the preprocessing and the file formats.

    python tests/test_deep.py        or        python -m pytest tests
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ["MILK10K_OFFLINE_TEST"] = "1"
import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "m5_final"))

from milk10k_pipeline import deep, evaluation as ev   # noqa: E402


def test_parameter_counts_match_session7():
    """Session 7 slides: ResNet-18 head 512->3 = 1,539; layer4 + fc = 8,395,267; total 11,178,051."""
    m = deep.build_model("resnet18", 3)
    assert deep.n_total(m) == 11_178_051
    assert deep.n_trainable(deep.freeze(m, ["fc"])) == 1_539
    assert deep.n_trainable(deep.freeze(m, ["layer4", "fc"])) == 8_395_267
    vit = deep.build_model("vit_b_16", 3)
    assert deep.n_trainable(deep.freeze(vit, ["heads"])) == 768 * 3 + 3


def test_output_shapes():
    x = torch.zeros(2, 3, 224, 224)
    for arch in ["cnn_scratch", "resnet18", "vit_b_16"]:
        assert deep.build_model(arch, 11).eval()(x).shape == (2, 11)


def test_inference_preprocessing_is_deterministic_and_normalised():
    """prepare_image + VAL_TRANSFORM: any image size -> (3, 224, 224), ImageNet-normalised, always identical."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "x.jpg"
        Image.fromarray(np.full((300, 451, 3), 200, dtype=np.uint8)).save(path)
        a, b = deep.VAL_TRANSFORM(deep.prepare_image(path)), deep.VAL_TRANSFORM(deep.prepare_image(path))
        assert a.shape == (3, 224, 224) and torch.equal(a, b)
        # Session 7 by hand: red = 200 -> (200/255 - 0.485) / 0.229 = 1.31
        assert abs(a[0].mean().item() - 1.31) < 0.02
        # the cache stores exactly what predict() sees
        cache = deep.build_cache(["x"], {"x": path})
        assert torch.equal(deep.VAL_TRANSFORM(Image.fromarray(cache[0])), a)


def test_frozen_batchnorm_stays_fixed():
    """With layer4 + fc trainable, a training step must not change anything in conv1 ... layer3 (incl. BN stats)."""
    m = deep.build_model("resnet18", 3)
    deep.freeze(m, ["layer4", "fc"])
    before = {k: v.clone() for k, v in m.state_dict().items()}
    m.train(); deep.keep_frozen_batchnorm_fixed(m, ["layer4", "fc"])
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    loss = m(torch.randn(4, 3, 224, 224)).sum(); loss.backward(); opt.step()
    after = m.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before if not k.startswith(("layer4", "fc")))
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith("layer4"))


def test_checkpoint_roundtrip_keeps_trained_weights():
    m = deep.build_model("resnet18", 11)
    with tempfile.TemporaryDirectory() as d:
        p = deep.save_checkpoint(m, Path(d) / "m.pt", "resnet18", deep.CLASSES11, ["layer4", "fc"])
        m2, ck = deep.load_checkpoint(p)
        assert ck["classes"] == deep.CLASSES11
        assert torch.equal(m.fc.weight, m2.fc.weight) and torch.equal(m.layer4[1].conv2.weight, m2.layer4[1].conv2.weight)


def test_leaderboard_metric_by_hand():
    """Official rule on 4 lesions, 2 classes: class 0 -> TP 1, FP 1, FN 1 -> F1 0.5; class 1 never >= 0.5 -> F1 0."""
    y = np.array([0, 0, 1, 1])
    p = np.array([[0.9, 0.1], [0.4, 0.2], [0.6, 0.3], [0.2, 0.45]])
    score, per_class = ev.leaderboard_f1(y, p, 0.5)
    assert np.allclose(per_class, [0.5, 0.0]) and np.isclose(score, 0.25)
    # thresholds tuned at 0.3 for class 1: lesions 2 and 3 become 'yes' -> F1 1.0
    assert np.isclose(ev.leaderboard_f1(y, p, [0.5, 0.3])[1][1], 1.0)


def test_rescaling_moves_thresholds_to_half_and_keeps_order():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(11), 300)
    t = rng.uniform(0.05, 0.6, 11)
    q = ev.rescale_to_half(p, t)
    assert q.min() >= 0 and q.max() <= 1
    assert np.array_equal(q >= 0.5, p >= t)
    assert all(np.array_equal(np.argsort(p[:, c], kind="stable"), np.argsort(q[:, c], kind="stable")) for c in range(11))
    y = rng.integers(0, 11, 300)
    assert np.isclose(ev.leaderboard_f1(y, p, t)[0], ev.leaderboard_f1(y, q, 0.5)[0])


def test_predict_and_submission_files():
    """predict() returns 11 probabilities summing to 1; make_submission writes the exact leaderboard format."""
    from make_submission import check_submission, make_submission
    from predict import predict
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ck = deep.save_checkpoint(deep.build_model("resnet18", 11), d / "model.pt", "resnet18", deep.CLASSES11,
                                  ["layer4", "fc"], extra={"thresholds": [0.3] * 11, "use_thresholds": True})
        rows = []
        for k in range(5):
            lesion = f"IL_{k:07d}"
            (d / "MILK10k_Test_Input" / lesion).mkdir(parents=True)
            for j, kind in enumerate(["dermoscopic", "clinical: close-up"]):
                iid = f"ISIC_{k}{j:06d}"
                Image.fromarray(np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)).save(
                    d / "MILK10k_Test_Input" / lesion / f"{iid}.jpg")
                rows.append({"lesion_id": lesion, "image_type": kind, "isic_id": iid})
        pd.DataFrame(rows).to_csv(d / "meta.csv", index=False)
        probs = predict(d / "MILK10k_Test_Input" / "IL_0000000" / "ISIC_0000000.jpg", checkpoint=ck)
        assert list(probs) == deep.CLASSES11 and abs(sum(probs.values()) - 1) < 1e-5
        sub = make_submission(d / "MILK10k_Test_Input", d / "meta.csv", ck, d / "submission.csv", n_expected=5)
        check_submission(pd.read_csv(d / "submission.csv"), n_expected=5)
        assert list(sub.lesion) == [f"IL_{k:07d}" for k in range(5)]          # one row per lesion, not per image


if __name__ == "__main__":
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASSED  {test.__name__}")
    print(f"\nall {len(tests)} tests passed")
