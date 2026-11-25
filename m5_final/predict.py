"""
Milestone 5 -- predict(): the final model on ONE new dermoscopic image.

    python m5_final/predict.py path/to/image.jpg
    python m5_final/predict.py path/to/image.jpg --checkpoint m5_final/model_11class.pt

or from Python:

    from m5_final.predict import predict
    probs = predict("ISIC_0000000.jpg")        # {'AKIEC': 0.01, 'BCC': 0.83, ...}

What it does (Session 10, "the most important rule"): the image is prepared EXACTLY like the validation
images: RGB -> PIL resize to 224 x 224 -> ToTensor -> ImageNet mean/std. No augmentation. Then softmax.

The first call downloads the ImageNet weights of the backbone from torchvision (once, ~45-330 MB,
cached in ~/.cache/torch); the checkpoint contains only the weights that were trained on MILK10k.

Decision support for research and teaching only, NOT a medical device (see model_card.md).
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from milk10k_pipeline.deep import VAL_TRANSFORM, load_checkpoint, prepare_image   # noqa: E402

DEFAULT_CHECKPOINT = REPO / "m5_final" / "model_11class.pt"
DIAGNOSIS_1 = {"AKIEC": "Malignant or Indeterminate", "BCC": "Malignant", "MAL_OTH": "Malignant",
               "MEL": "Malignant", "SCCKA": "Malignant", "BEN_OTH": "Benign", "BKL": "Benign", "DF": "Benign",
               "INF": "Benign", "NV": "Benign", "VASC": "Benign"}
_LOADED = {}


def load_model(checkpoint=DEFAULT_CHECKPOINT, device="cpu"):
    """Load (once) and return (model in eval mode, checkpoint dict)."""
    key = (str(checkpoint), str(device))
    if key not in _LOADED:
        _LOADED[key] = load_checkpoint(checkpoint, device=device)
    return _LOADED[key]


@torch.no_grad()
def predict(image_path, checkpoint=DEFAULT_CHECKPOINT, device="cpu"):
    """Class probabilities (softmax, they sum to 1) of one image, as {class name: probability}."""
    model, ckpt = load_model(checkpoint, device)
    x = VAL_TRANSFORM(prepare_image(image_path)).unsqueeze(0).to(device)     # (1, 3, 224, 224)
    probs = torch.softmax(model(x), dim=1)[0].cpu().tolist()
    return dict(zip(ckpt["classes"], probs))


def main():
    parser = argparse.ArgumentParser(description="Predict the diagnosis of one dermoscopic image.")
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()
    probs = predict(args.image, args.checkpoint)
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    print(f"\n{Path(args.image).name}")
    for name, p in ranked:
        print(f"  {name:<8} {p:6.3f}  {'#' * int(round(p * 40))}")
    top = ranked[0][0]
    print(f"\nmost likely: {top} ({DIAGNOSIS_1.get(top, '?')}), probability {ranked[0][1]:.2f}")
    print("Research / teaching use only. Not a diagnosis: a dermatologist must assess the lesion.")


if __name__ == "__main__":
    main()
