"""
B7 -- Augmentation pipeline (importable, reused by every later milestone).

Augmentation table (each one keeps the diagnosis unchanged):

| augmentation          | parameters                        | why it does not change the label |
|-----------------------|-----------------------------------|----------------------------------|
| Resize + RandomCrop   | shorter side 256, crop 224        | small shifts of framing; the lesion stays in view (it is centred in dermoscopy) |
| RandomHorizontalFlip  | p = 0.5                           | skin has no left/right; a mirrored lesion is the same lesion |
| RandomVerticalFlip    | p = 0.5                           | skin has no up/down; dermoscopes are held at any orientation |
| RandomRotation        | +-20 degrees                      | orientation of the dermoscope is arbitrary; small angle limits black corners |
| ColorJitter           | brightness 0.1, hue 0.02          | mild lighting/camera variation; Part A A3.5 showed these stay BELOW the real Benign-Malignant colour gap, while larger values exceed it |

Not used: strong colour jitter, grayscale, blur, heavy crops (they can erase diagnostic colour or
border information). Augmentations must be approved by a dermatologist before real use.
"""
from torchvision import transforms

from . import config

# ImageNet statistics are only a fallback; the pipeline uses statistics computed on the TRAIN split.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(mean=IMAGENET_MEAN, std=IMAGENET_STD, normalize=True,
                          resize=config.RESIZE, size=config.IMAGE_SIZE):
    """Random augmentations for TRAINING only. normalize=False gives a displayable [0, 1] tensor."""
    steps = [
        transforms.Resize(resize),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.1, hue=0.02),
        transforms.ToTensor(),
    ]
    if normalize:
        steps.append(transforms.Normalize(mean, std))
    return transforms.Compose(steps)


def build_eval_transform(mean=IMAGENET_MEAN, std=IMAGENET_STD, normalize=True,
                         resize=config.RESIZE, size=config.IMAGE_SIZE):
    """Deterministic transform for validation / test: no randomness at all."""
    steps = [
        transforms.Resize(resize),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ]
    if normalize:
        steps.append(transforms.Normalize(mean, std))
    return transforms.Compose(steps)
