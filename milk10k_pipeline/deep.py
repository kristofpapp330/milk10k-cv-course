"""
Milestones 3-5 -- Deep models (Sessions 7 and 9): CNN from scratch, ResNet-18 and ViT-B/16.

ONE preprocessing for everything that is not training (Session 10, "the most important rule"):

    prepare_image(path)  ->  RGB, PIL resize to 224 x 224        (validation, test, predict, submission)
    VAL_TRANSFORM        ->  ToTensor + ImageNet mean/std         (Session 7, "speak ImageNet's language")

Training adds only label-preserving augmentations on top (flips, small rotation, mild colour jitter,
the Milestone 1 table), never anything at inference time.

To make training fast on Colab, every image is passed through prepare_image ONCE and kept as a
224 x 224 x 3 uint8 array (the "cache"). Because the cache is the output of prepare_image itself,
a validation image from the cache and a new image in predict() go through exactly the same steps.
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from . import config
from .data_io import load_rgb
from .labels import ELEVEN_CLASSES
from .mlp import class_weights
from .transforms import IMAGENET_MEAN, IMAGENET_STD

CLASSES3 = ["Benign", "Malignant", "Indeterminate"]     # diagnosis_1 (analysis target, Path A)
CLASSES11 = list(ELEVEN_CLASSES)                         # training_gt.csv order = submission column order
IMAGE_SIZE = 224


# ==================================================================================================
# 1. preprocessing
# ==================================================================================================
def prepare_image(path, size=IMAGE_SIZE):
    """THE inference preprocessing: open as RGB and resize the whole image to 224 x 224 (no crop,
    so the full lesion stays in view). Used for the cache, validation, test, predict() and the submission."""
    img = load_rgb(path, min_size=size)            # JPEG decoded at reduced scale, never below 224
    return img.resize((size, size), Image.BILINEAR)


VAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),                          # 0-255 -> 0-1, HWC -> CHW
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

TRAIN_TRANSFORM = transforms.Compose([              # Milestone 1 augmentation table, minus the crop
    transforms.RandomHorizontalFlip(p=0.5),         # skin has no left/right
    transforms.RandomVerticalFlip(p=0.5),           # ... and no up/down
    transforms.RandomRotation(degrees=20),          # dermoscope orientation is arbitrary
    transforms.ColorJitter(brightness=0.1, hue=0.02),  # mild: stays below the real colour gap (Part A)
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_cache(isic_ids, image_paths, cache_path=None, every=500):
    """uint8 array (N, 224, 224, 3) = prepare_image() of every id, in the given order. Cached as .npz."""
    ids = [str(i) for i in isic_ids]
    if cache_path is not None and Path(cache_path).exists():
        data = np.load(cache_path)
        cached_ids = list(data["ids"])
        if cached_ids == ids:
            print(f"loaded image cache ({len(ids)} images) from {cache_path}")
            return data["images"]
        print("cache does not match the requested images -> rebuilding")
    out, start = np.zeros((len(ids), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8), time.time()
    for k, isic_id in enumerate(ids):
        out[k] = np.asarray(prepare_image(image_paths[isic_id]))
        if (k + 1) % every == 0 or k + 1 == len(ids):
            print(f"  {k + 1}/{len(ids)} images prepared ({time.time() - start:.0f}s)")
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, ids=np.array(ids), images=out)
    return out


class ArrayDataset(Dataset):
    """(image tensor, label) pairs from the uint8 cache; `transform` = TRAIN_TRANSFORM or VAL_TRANSFORM."""

    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, np.asarray(labels), transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i])), int(self.labels[i])


def make_loader(images, labels, train, batch_size=64, num_workers=2, seed=config.SEED):
    ds = ArrayDataset(images, labels, TRAIN_TRANSFORM if train else VAL_TRANSFORM)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      generator=torch.Generator().manual_seed(seed) if train else None,
                      pin_memory=torch.cuda.is_available())


# ==================================================================================================
# 2. models
# ==================================================================================================
class SmallCNN(nn.Module):
    """CNN from scratch (Session 7): 4 x [conv 3x3 -> batch-norm -> ReLU -> max-pool 2x2], then global
    average pooling (one number per feature map) and a dropout + linear head.

    224 -> 112 -> 56 -> 28 -> 14 pixels; 3 -> 32 -> 64 -> 128 -> 256 channels.
    Batch-norm re-centres every channel during training, which makes training from zero much more stable.
    """

    def __init__(self, n_classes, channels=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        layers, c_in = [], 3
        for c_out in channels:
            layers += [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.BatchNorm2d(c_out),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def build_model(arch, n_classes, pretrained=True):
    """arch: 'cnn_scratch', 'resnet18' or 'vit_b_16'. Pretrained = ImageNet-1k weights from torchvision.
    The last layer is replaced by a new random one with n_classes outputs (Session 7 'transfer learning')."""
    if os.environ.get("MILK10K_OFFLINE_TEST") == "1":
        pretrained = False          # only for the offline tests (no internet to download ImageNet weights)
    if arch == "cnn_scratch":
        return SmallCNN(n_classes)
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)             # 512 -> n_classes
        return model
    if arch == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, n_classes)   # 768 -> n_classes
        return model
    raise ValueError(f"unknown architecture {arch}")


HEAD_PREFIX = {"cnn_scratch": "head", "resnet18": "fc", "vit_b_16": "heads"}

# What is trained in every deep model of the scoreboard (None = everything).
RECIPES = {
    "CNN from scratch": {
        "arch": "cnn_scratch", "pretrained": False, "trainable": None,
        "lr_head": 1e-3, "lr_backbone": 1e-3, "epochs": 30, "patience": 8},
    "ResNet-18 fine-tune layer4": {          # Session 7 strategy 2, "the usual best default"
        "arch": "resnet18", "pretrained": True, "trainable": ["layer4", "fc"],
        "lr_head": 1e-3, "lr_backbone": 1e-4, "epochs": 15, "patience": 5},
    "ResNet-18 fine-tune layer3+4": {        # Session 7 mini challenge
        "arch": "resnet18", "pretrained": True, "trainable": ["layer3", "layer4", "fc"],
        "lr_head": 1e-3, "lr_backbone": 1e-4, "epochs": 15, "patience": 5},
    "ViT-B/16 fine-tune last 2 blocks": {    # Session 9 + GPU: the same idea as ResNet strategy 2
        "arch": "vit_b_16", "pretrained": True,
        "trainable": ["encoder.layers.encoder_layer_10", "encoder.layers.encoder_layer_11", "encoder.ln", "heads"],
        "lr_head": 1e-3, "lr_backbone": 1e-4, "epochs": 10, "patience": 4},
}


def freeze(model, trainable):
    """requires_grad = True only for parameters whose name starts with one of the `trainable` prefixes
    (Session 7: freezing = param.requires_grad = False). trainable=None -> train everything."""
    for name, p in model.named_parameters():
        p.requires_grad = trainable is None or name.startswith(tuple(trainable))
    return model


def keep_frozen_batchnorm_fixed(model, trainable):
    """A frozen BatchNorm layer would still update its running mean/std in train mode. Put those layers
    in eval mode so the frozen part of the network really stays exactly as pretrained."""
    if trainable is None:
        return
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and not name.startswith(tuple(trainable)):
            m.eval()


def n_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def n_total(model):
    return sum(p.numel() for p in model.parameters())


# ==================================================================================================
# 3. training (Session 6's loop, unchanged: forward, loss, backward, step)
# ==================================================================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_proba(model, loader, device):
    """Softmax probabilities for every image of a (non-shuffled) loader, in eval mode."""
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        out.append(torch.softmax(logits.float(), dim=1).cpu())
    return torch.cat(out).numpy()


def train_model(model, recipe, train_loader, val_loader, y_train, y_val, n_classes, device, seed=config.SEED,
                epochs=None, verbose=True):
    """Train with class-weighted cross-entropy and Adam (two learning rates: new head vs pretrained layers).

    After every epoch the validation macro-F1 is computed; the weights of the BEST epoch are kept
    (early stopping), and training stops after `patience` epochs without improvement.
    Returns (model with best weights, history table, best epoch).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    trainable = recipe["trainable"]
    freeze(model, trainable)
    model.to(device)
    head = HEAD_PREFIX[recipe["arch"]]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(head)]
    body_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(head)]
    groups = [{"params": head_params, "lr": recipe["lr_head"]}]
    if body_params:
        groups.append({"params": body_params, "lr": recipe["lr_backbone"]})
    optimizer = torch.optim.Adam(groups)
    weights = torch.tensor(class_weights(np.asarray(y_train), n_classes), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")     # mixed precision on GPU = faster

    epochs = epochs or recipe["epochs"]
    history, best_f1, best_state, best_epoch, waited = [], -1.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        keep_frozen_batchnorm_fixed(model, trainable)
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)                            # 0. forget old gradients
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)                                  # 1-2. forward + loss
            scaler.scale(loss).backward()                                    # 3. backpropagation
            scaler.step(optimizer)                                           # 4. update
            scaler.update()
            total, n = total + loss.item() * len(y), n + len(y)
        proba = predict_proba(model, val_loader, device)
        # same class-weighted cross-entropy as in training, computed from the validation probabilities
        val_loss = nn.functional.nll_loss(torch.log(torch.tensor(proba).clamp_min(1e-8)),
                                          torch.tensor(np.asarray(y_val)).long(), weight=weights.cpu()).item()
        val_f1 = f1_score(y_val, proba.argmax(1), average="macro", labels=list(range(n_classes)), zero_division=0)
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss, "val_macro_f1": val_f1,
                        "seconds": time.time() - start})
        if verbose:
            print(f"  epoch {epoch:>2}: train loss {total / n:.3f} | val loss {val_loss:.3f} | "
                  f"val macro-F1 {val_f1:.3f} | {time.time() - start:.0f}s")
        if val_f1 > best_f1:
            best_f1, best_epoch, waited = val_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= recipe.get("patience", epochs):
                if verbose:
                    print(f"  early stop: no improvement for {waited} epochs (best epoch {best_epoch})")
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch


# ==================================================================================================
# 4. frozen backbone = feature extractor (Session 7 strategy 1, Session 9 fair comparison)
# ==================================================================================================
@torch.no_grad()
def backbone_embeddings(arch, images, device, batch_size=128, num_workers=2):
    """The frozen pretrained backbone turns every image into one vector (ResNet-18: 512, ViT-B/16: 768,
    the class token). Computed once; a linear head is then trained on these numbers (Session 6 code)."""
    model = build_model(arch, n_classes=1, pretrained=True)
    if arch == "resnet18":
        model.fc = nn.Identity()
    else:
        model.heads = nn.Identity()
    model.to(device).eval()
    loader = make_loader(images, np.zeros(len(images)), train=False, batch_size=batch_size, num_workers=num_workers)
    out = []
    for x, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(x.to(device)).float().cpu())
    return torch.cat(out).numpy()


def model_from_frozen_head(arch, head_linear, n_classes):
    """Put a head trained on frozen embeddings back on top of the pretrained backbone -> one normal model."""
    model = build_model(arch, n_classes, pretrained=True)
    target = model.fc if arch == "resnet18" else model.heads.head
    target.load_state_dict(head_linear.state_dict())
    return model


# ==================================================================================================
# 5. checkpoints: only the trained part is saved (the frozen part = torchvision's ImageNet weights)
# ==================================================================================================
def save_checkpoint(model, path, arch, classes, trainable, pretrained=True, extra=None):
    """Saves arch + class names + the weights that were trained. For a fine-tuned or frozen pretrained model
    the frozen weights are NOT saved (they are re-downloaded from torchvision), which keeps the file small."""
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if trainable is None or k.startswith(tuple(trainable))}
    ckpt = {"arch": arch, "classes": list(classes), "trainable": None if trainable is None else list(trainable),
            "pretrained": bool(pretrained), "image_size": IMAGE_SIZE,
            "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD), "state_dict": state}
    if extra:
        ckpt.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    return Path(path)


def load_checkpoint(path, device="cpu"):
    """Rebuild the model: architecture -> ImageNet weights (if it was pretrained) -> trained weights on top."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=ckpt["pretrained"])
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:5]}"
    if ckpt["trainable"] is not None:
        bad = [k for k in missing if k.startswith(tuple(ckpt["trainable"]))]
        assert not bad, f"trained weights missing from checkpoint: {bad[:5]}"
    else:
        assert not missing, f"weights missing from checkpoint: {missing[:5]}"
    return model.to(device).eval(), ckpt
