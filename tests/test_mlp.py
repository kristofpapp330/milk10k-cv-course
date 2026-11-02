"""
Tests for the Session 6 homework (MLP + nn.Conv2d). No dataset needed.

Most tests check that PyTorch gives EXACTLY the numbers computed by hand on the Session 4 and
Session 6 slides.

Run from the repository root with either:
    python tests/test_mlp.py
    python -m pytest tests
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make the package importable

from milk10k_pipeline import mlp

# Session 4 toy lesion: skin = 100, lesion = 10, as a (batch, channel, height, width) tensor
TOY = torch.tensor([[100, 100, 100, 100, 100],
                    [100, 100, 100, 100, 100],
                    [100, 100, 10, 10, 10],
                    [100, 100, 10, 10, 10],
                    [100, 100, 10, 10, 10]], dtype=torch.float32).view(1, 1, 5, 5)


def conv_with(kernel, **kwargs):
    """An nn.Conv2d whose single 3x3 kernel is set by hand (no bias)."""
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False, **kwargs)
    with torch.no_grad():
        conv.weight[:] = torch.tensor(kernel, dtype=torch.float32).view(1, 1, 3, 3)
    return conv


BOX = np.ones((3, 3)) / 9
SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]


@torch.no_grad()
def test_conv2d_box_blur_no_padding():
    """Session 4: box blur, kernel 3, stride 1, no padding -> 3x3 output [[90,80,70],[80,60,40],[70,40,10]]."""
    out = conv_with(BOX)(TOY)[0, 0]
    assert out.shape == (3, 3)
    assert torch.allclose(out, torch.tensor([[90., 80, 70], [80, 60, 40], [70, 40, 10]]))


@torch.no_grad()
def test_conv2d_stride_2():
    """Session 4 'Stride by hand': stride 2 -> 2x2 output [[90, 70], [70, 10]]."""
    out = conv_with(BOX, stride=2)(TOY)[0, 0]
    assert torch.allclose(out, torch.tensor([[90., 70], [70, 10]]))


@torch.no_grad()
def test_conv2d_padding_zero_vs_mirror():
    """Session 4 'Padding by hand': padding 1 keeps 5x5; corner = 44.4 with zeros, 100 with mirror padding."""
    zero = conv_with(BOX, padding=1)(TOY)[0, 0]
    mirror = conv_with(BOX, padding=1, padding_mode="reflect")(TOY)[0, 0]
    assert zero.shape == (5, 5) and round(zero[0, 0].item(), 1) == 44.4
    assert round(mirror[0, 0].item(), 1) == 100.0
    same = conv_with(BOX, padding="same")(TOY)[0, 0]
    assert same.shape == (5, 5)


@torch.no_grad()
def test_conv2d_sobel_matches_session4():
    """Sobel Gx with nn.Conv2d = the Session 4 slide (PyTorch does cross-correlation, no kernel flip)."""
    gx = conv_with(SOBEL_X)(TOY)[0, 0]
    assert torch.allclose(gx, torch.tensor([[-90., -90, 0], [-270, -270, 0], [-360, -360, 0]]))


def test_output_size_formula():
    """floor((n + 2p - k) / s) + 1 from Session 4 = the H_out formula of the PyTorch docs (dilation 1)."""
    for n, k, s, p in [(5, 3, 1, 0), (5, 3, 2, 0), (5, 3, 1, 1), (32, 3, 2, 1), (224, 7, 2, 3)]:
        conv = nn.Conv2d(1, 1, kernel_size=k, stride=s, padding=p)
        assert conv(torch.zeros(1, 1, n, n)).shape[-1] == (n + 2 * p - k) // s + 1


def test_parameter_count_matches_slide():
    """Session 6 'A layer is a matrix multiplication': 33 -> 64 -> 3 has 2,176 + 195 = 2,371 parameters."""
    assert mlp.n_parameters(mlp.make_mlp(33, 3, hidden=(64,))) == 2371


def test_forward_pass_by_hand():
    """Session 6 'The forward pass, step by step': output 0.570 for the 2-2-1 network of the slide."""
    net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1), nn.Sigmoid())
    with torch.no_grad():
        net[0].weight[:] = torch.tensor([[0.5, 0.3], [-0.3, 0.4]])    # one row per hidden neuron
        net[0].bias[:] = torch.tensor([0.1, -0.3])
        net[2].weight[:] = torch.tensor([[0.4, -0.2]])
        net[2].bias[:] = torch.tensor([0.1])
        assert abs(net(torch.tensor([[0.6, 0.2]])).item() - 0.570) < 1e-3     # sigma(0.284) = 0.5705


def test_backprop_by_hand():
    """Session 6 'Backpropagation by hand': autograd gives the gradients -0.241, -0.080, -0.401."""
    w = torch.tensor([1.5, -0.5], requires_grad=True)
    b = torch.tensor(-0.4, requires_grad=True)
    y_hat = torch.sigmoid(w @ torch.tensor([0.6, 0.2]) + b)
    loss = -torch.log(y_hat)                                     # truth y = 1
    loss.backward()
    assert round(y_hat.item(), 3) == 0.599 and round(loss.item(), 3) == 0.513
    assert np.allclose(w.grad.numpy().round(3), [-0.241, -0.080]) and round(b.grad.item(), 3) == -0.401


def test_class_weights_match_slide():
    """Session 6 'Class imbalance': n / (3 n_c) = 1.18, 0.48, 14.43 for 1016 / 2494 / 83 lesions."""
    y = np.repeat([0, 1, 2], [1016, 2494, 83])
    assert np.allclose(mlp.class_weights(y, 3).round(2), [1.18, 0.48, 14.43])


def test_mlp_learns_xor_like_data():
    """On a non-linear (XOR-like) problem the MLP gets far above chance on validation."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (1200, 2)).astype(np.float32)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
    model, hist, best = mlp.train_mlp(X[:800], y[:800], X[800:], y[800:], 2, hidden=(32,), lr=1e-2, epochs=40)
    assert hist["val_macro_f1"].max() > 0.9
    assert 1 <= best <= 40


if __name__ == "__main__":
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASSED  {test.__name__}")
    print(f"\nall {len(tests)} tests passed")
