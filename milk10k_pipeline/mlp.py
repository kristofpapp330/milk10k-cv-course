"""
Session 6 -- Multilayer perceptron (MLP) in PyTorch, on handcrafted features or on raw pixels.

The model and the training loop are the ones of the Session 6 slides:

    model     = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, n_classes))
    loss_fn   = nn.CrossEntropyLoss(weight=class_weights)      # softmax is inside the loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    50 epochs, mini-batches of 64, shuffled every epoch

plus the regularisation tools of the slides (dropout, weight decay, early stopping), which are
switched on only in the tuned configurations (CONFIGS below).

Leakage rules (Session 3): imputation medians, standardisation mean/std and class weights are all
computed on the TRAIN split only. The validation split is used for early stopping and for choosing
the configuration; the test split is never seen here.
"""
import copy
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .data_io import load_rgb

# --------------------------------------------------------------------------------------------------
# configurations that are compared (identical for every kind of input)
# --------------------------------------------------------------------------------------------------
CONFIGS = {
    "class MLP (64)":          {"hidden": (64,), "dropout": 0.0, "weight_decay": 0.0},   # the Session 6 model
    "wide (256) + dropout/wd": {"hidden": (256,), "dropout": 0.3, "weight_decay": 1e-4},
    "deep (64,64) + dropout/wd": {"hidden": (64, 64), "dropout": 0.3, "weight_decay": 1e-4},
}
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
SEEDS = (0, 1, 2)            # every configuration is trained 3 times: neural networks are noisy


def make_mlp(n_in, n_classes, hidden=(64,), dropout=0.0):
    """Linear -> ReLU (-> Dropout) for every hidden layer, then a Linear output layer with raw logits."""
    layers, width = [], n_in
    for h in hidden:
        layers += [nn.Linear(width, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        width = h
    layers.append(nn.Linear(width, n_classes))
    return nn.Sequential(*layers)


def n_parameters(model):
    """Number of learnable numbers (weights + biases): n_in*n_out + n_out per Linear layer (Session 6)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def class_weights(y, n_classes):
    """Session 6 'Class imbalance' slide: weight_c = n / (K * n_c), computed on TRAIN labels."""
    counts = np.bincount(y, minlength=n_classes)
    return len(y) / (n_classes * counts)


# --------------------------------------------------------------------------------------------------
# preparing the inputs
# --------------------------------------------------------------------------------------------------
class Standardizer:
    """Median imputation + z-score, both learned on TRAIN only (Session 5 'Feature scale matters')."""

    def fit(self, X):
        self.median = np.nanmedian(X, axis=0)
        Xf = np.where(np.isnan(X), self.median, X)
        self.mean, self.std = Xf.mean(axis=0), Xf.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        Xf = np.where(np.isnan(X), self.median, X)
        return ((Xf - self.mean) / self.std).astype(np.float32)


def load_pixels(images, image_paths, size=32, cache_path=None, every=1000):
    """Every image shrunk to size x size x 3 (uint8). Returns an array (N, size, size, 3) in the row order of `images`.

    The whole image is resized (not cropped), so the full lesion stays in view.
    Cached to an .npz file because decoding 5,240 JPEGs takes a few minutes.
    """
    ids = list(images["isic_id"])
    if cache_path is not None and cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        cached = dict(zip(data["ids"], data["pixels"]))
        if all(i in cached for i in ids) and data["pixels"].shape[1] == size:
            print(f"loaded cached {size}x{size} pixels for {len(ids)} images")
            return np.stack([cached[i] for i in ids])
    out, start = np.zeros((len(ids), size, size, 3), dtype=np.uint8), time.time()
    for k, isic_id in enumerate(ids):
        img = load_rgb(image_paths[isic_id], min_size=size)
        out[k] = np.asarray(img.resize((size, size), Image.BILINEAR))
        if (k + 1) % every == 0 or k + 1 == len(ids):
            print(f"  {k + 1}/{len(ids)} images ({time.time() - start:.0f}s)")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, ids=np.array(ids), pixels=out)
    return out


# --------------------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_proba(model, X, batch_size=1024):
    """Softmax probabilities, in evaluation mode (dropout off)."""
    model.eval()
    out = [torch.softmax(model(torch.from_numpy(X[i:i + batch_size])), dim=1) for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()


def train_mlp(X_tr, y_tr, X_va, y_va, n_classes, hidden=(64,), dropout=0.0, weight_decay=0.0,
              lr=LR, epochs=EPOCHS, batch_size=BATCH_SIZE, seed=0):
    """The Session 6 training loop. X_* already standardised float32 arrays, y_* integer labels.

    Early stopping: after every epoch the validation macro-F1 is measured, and the weights of the
    BEST epoch are kept (not the last ones).
    Returns (model with the best weights, history table, best epoch).
    """
    set_seed(seed)
    model = make_mlp(X_tr.shape[1], n_classes, hidden, dropout)
    weights = torch.tensor(class_weights(y_tr, n_classes), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long()),
                        batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    Xv, yv = torch.from_numpy(X_va), torch.from_numpy(y_va).long()

    history, best_f1, best_state, best_epoch = [], -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()                 # 0. forget the old gradients
            loss = loss_fn(model(xb), yb)         # 1-2. forward pass + loss
            loss.backward()                       # 3. backpropagation
            optimizer.step()                      # 4. update rule
            total, n = total + loss.item() * len(xb), n + len(xb)
        model.eval()
        with torch.no_grad():
            logits = model(Xv)
            val_loss = loss_fn(logits, yv).item()
            val_f1 = f1_score(y_va, logits.argmax(1).numpy(), average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss, "val_macro_f1": val_f1})
        if val_f1 > best_f1:
            best_f1, best_state, best_epoch = val_f1, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch


def run_configs(X_tr, y_tr, X_va, y_va, n_classes, configs=None, seeds=SEEDS, label=""):
    """Every configuration x every seed, scored on VALIDATION. Returns (results table, trained models, histories)."""
    configs = configs or CONFIGS
    rows, models, histories = [], {}, {}
    for name, cfg in configs.items():
        for seed in seeds:
            start = time.time()
            model, hist, best_epoch = train_mlp(X_tr, y_tr, X_va, y_va, n_classes, seed=seed, **cfg)
            rows.append({"config": name, "seed": seed, "val_macro_f1": hist["val_macro_f1"].max(),
                         "best_epoch": best_epoch, "n_params": n_parameters(model),
                         "seconds": time.time() - start})
            models[(name, seed)], histories[(name, seed)] = model, hist
        mean = np.mean([r["val_macro_f1"] for r in rows if r["config"] == name])
        print(f"  {label:<28} {name:<26} mean val macro-F1 = {mean:.3f}")
    return pd.DataFrame(rows), models, histories


def select_config(results):
    """The configuration with the best MEAN validation macro-F1 over the seeds."""
    return results.groupby("config", sort=False)["val_macro_f1"].mean().idxmax()
