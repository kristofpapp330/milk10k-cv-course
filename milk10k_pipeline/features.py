"""
Milestone 2 -- Handcrafted features (Session 5): one fixed-length feature vector per image.

Every image is resized to FEATURE_SIZE x FEATURE_SIZE so that all features are measured on the
same scale, then turned into numbers grouped in five FEATURE GROUPS:

    group        n   what it measures                                   session
    -----------  --  -------------------------------------------------  ---------------------------
    color_stats   6  mean and std of R, G, B                            Session 5 (and Session 2)
    color_hist   24  8-bin histogram per channel, as fractions          Session 5
    texture       3  edge density, roughness, Laplacian variance         Session 5, filters from Session 4
    glcm          4  GLCM contrast, homogeneity, energy, correlation     Session 5 "optional extension"
    abcd          8  asymmetry (2) + rough lesion mask features (6)      Session 5 ABCDE slide

The FEATURE SETS compared by the models are combinations of these groups (FEATURE_SETS below).
"""
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu

from .data_io import load_rgb

FEATURE_SIZE = 256          # every image is resized to 256 x 256 before measuring anything
HIST_BINS = 8               # bins per colour channel (Session 5: 8 bins x 3 channels = 24 numbers)
ROUGHNESS_WINDOW = 9        # local-variance window (Session 5: 9 x 9)
GLCM_LEVELS = 32            # grey levels for the co-occurrence matrix (Session 5: "8-32 grey levels")
MASK_SIGMA = 2.0            # Gaussian blur before thresholding the lesion (Session 4 homework: sigma 1-2)
MIN_CONTRAST = 10           # a mask must be at least 10 grey levels darker than the rest, else "no mask"

CHANNELS = ("R", "G", "B")


# --------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------
def to_gray(rgb):
    """RGB (0-255) -> grayscale (0-255) with the standard luminance weights."""
    return rgb[..., :3] @ np.array([0.299, 0.587, 0.114])


def sobel_magnitude(gray):
    """Session 4: sqrt(Gx^2 + Gy^2) with the 3x3 Sobel kernels."""
    return np.hypot(ndi.sobel(gray, axis=1), ndi.sobel(gray, axis=0))


# --------------------------------------------------------------------------------------------------
# 1. colour features
# --------------------------------------------------------------------------------------------------
def color_stats(rgb):
    """6 numbers: mean and std of each channel (overall colour, and colour variability)."""
    feats = {}
    for i, ch in enumerate(CHANNELS):
        feats[f"mean_{ch}"] = rgb[..., i].mean()
        feats[f"std_{ch}"] = rgb[..., i].std()
    return feats


def color_histogram(rgb, bins=HIST_BINS):
    """24 numbers: per channel, the FRACTION of pixels in each of 8 equal intensity bins (0-31, 32-63, ...).

    Dividing by the number of pixels makes big and small images comparable (Session 5).
    """
    feats = {}
    n_pixels = rgb.shape[0] * rgb.shape[1]
    for i, ch in enumerate(CHANNELS):
        counts, _ = np.histogram(rgb[..., i], bins=bins, range=(0, 256))
        for b, c in enumerate(counts):
            feats[f"hist_{ch}{b}"] = c / n_pixels
    return feats


# --------------------------------------------------------------------------------------------------
# 2. texture features (Session 4 filters, each averaged into one number)
# --------------------------------------------------------------------------------------------------
def texture_features(gray, window=ROUGHNESS_WINDOW):
    """3 numbers.

    edge_density   mean Sobel magnitude               -> many borders / fine detail
    roughness      mean local variance in 9x9 windows -> rough vs uniform surface
                   local variance = mean(x^2) - mean(x)^2 in every window (box filter, no loop)
    laplacian_var  variance of the Laplacian           -> sharp detail vs blurry
    """
    g = gray.astype(np.float64)
    local_var = ndi.uniform_filter(g ** 2, window) - ndi.uniform_filter(g, window) ** 2
    return {
        "edge_density": sobel_magnitude(g).mean(),
        "roughness": np.clip(local_var, 0, None).mean(),     # clip tiny negative rounding errors
        "laplacian_var": ndi.laplace(g).var(),
    }


def glcm_features(gray, levels=GLCM_LEVELS):
    """4 numbers: Haralick GLCM properties, averaged over 4 directions (0, 45, 90, 135 degrees), distance 1.

    The GLCM counts how often grey level i sits next to grey level j:
      contrast     large when neighbours differ a lot          (rough texture)
      homogeneity  large when neighbours are similar           (smooth texture)
      energy       large when a few patterns repeat            (uniform texture)
      correlation  how linearly predictable a pixel is from its neighbour
    """
    q = np.clip(gray, 0, 255).astype(np.uint16) * levels // 256        # quantise 0-255 -> 0..levels-1
    glcm = graycomatrix(q.astype(np.uint8), distances=[1],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=levels, symmetric=True, normed=True)
    return {f"glcm_{p}": graycoprops(glcm, p).mean()
            for p in ("contrast", "homogeneity", "energy", "correlation")}


# --------------------------------------------------------------------------------------------------
# 3. ABCD-style features (Session 5 "Shape features: borrowing from dermatology")
# --------------------------------------------------------------------------------------------------
def asymmetry_features(rgb):
    """2 numbers (A = Asymmetry), whole-image approximation without a mask:
    mean absolute colour difference between the image and its left-right / top-bottom mirror image.
    0 = perfectly symmetric. Assumes the lesion is roughly centred (true for most dermoscopic images).
    """
    return {
        "asym_lr": np.abs(rgb - rgb[:, ::-1]).mean() / 255,
        "asym_tb": np.abs(rgb - rgb[::-1, :]).mean() / 255,
    }


def lesion_mask(gray, sigma=MASK_SIGMA):
    """A ROUGH lesion mask (proper segmentation comes in Session 8):

    1. Gaussian blur (removes noise and thin hairs a bit),
    2. Otsu threshold -> 'darker than the skin'. The threshold is computed on the CENTRAL half of the
       image only, so the black corners of the dermoscope (vignetting) do not pull it down,
    3. keep the dark connected region that overlaps the image centre the most (ignores dark corners).
    Returns a boolean mask, or None if there is no clearly darker region in the centre.
    """
    smooth = ndi.gaussian_filter(gray, sigma)
    h, w = gray.shape
    centre_box = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
    if np.ptp(smooth[centre_box]) < 1:
        return None                                      # completely flat centre
    dark = smooth < threshold_otsu(smooth[centre_box])
    labels, n = ndi.label(dark)
    if n == 0:
        return None
    overlap = np.bincount(labels[centre_box].ravel(), minlength=n + 1)
    overlap[0] = 0                                       # 0 = background, not a region
    if overlap.max() == 0:
        return None
    mask = ndi.binary_fill_holes(labels == overlap.argmax())
    if smooth[~mask].mean() - smooth[mask].mean() < MIN_CONTRAST:
        return None                                      # 'lesion' barely darker than skin = noise, not a lesion
    return mask


def mask_features(rgb, gray):
    """6 numbers from the rough mask (B = Border, C = Colour contrast, D = Diameter-like size):

    mask_area        fraction of the image covered by the lesion (D; pixels only, no mm scale)
    mask_compactness perimeter^2 / (4*pi*area): 1 for a circle, larger for irregular / notched borders (B)
    mask_border      fraction of the image border covered by the mask (lesion cut off by the frame)
    contrast_R/G/B   mean colour outside the lesion minus inside (C: how much darker / redder than the skin)
    If no mask was found, all six are NaN and are filled later with the TRAIN median.
    """
    mask = lesion_mask(gray)
    names = ["mask_area", "mask_compactness", "mask_border", "contrast_R", "contrast_G", "contrast_B"]
    if mask is None or mask.all() or mask.sum() < 50:
        return dict.fromkeys(names, np.nan)
    area = mask.sum()
    perimeter = (mask & ~ndi.binary_erosion(mask)).sum()
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    inside, outside = rgb[mask].mean(axis=0), rgb[~mask].mean(axis=0)
    feats = {
        "mask_area": area / mask.size,
        "mask_compactness": perimeter ** 2 / (4 * np.pi * area),
        "mask_border": border.mean(),
    }
    for i, ch in enumerate(CHANNELS):
        feats[f"contrast_{ch}"] = outside[i] - inside[i]
    return feats


# --------------------------------------------------------------------------------------------------
# feature groups and feature sets
# --------------------------------------------------------------------------------------------------
def extract_features(rgb):
    """All features of ONE image (RGB uint8/float array, already resized). Returns a dict name -> value."""
    rgb = rgb.astype(np.float64)
    gray = to_gray(rgb)
    feats = {}
    feats.update(color_stats(rgb))
    feats.update(color_histogram(rgb))
    feats.update(texture_features(gray))
    feats.update(glcm_features(gray))
    feats.update(asymmetry_features(rgb))
    feats.update(mask_features(rgb, gray))
    return feats


def feature_groups(columns):
    """Map each feature group name to its column names (in a fixed order)."""
    cols = list(columns)
    return {
        "color_stats": [c for c in cols if c.startswith(("mean_", "std_"))],
        "color_hist": [c for c in cols if c.startswith("hist_")],
        "texture": ["edge_density", "roughness", "laplacian_var"],
        "glcm": [c for c in cols if c.startswith("glcm_")],
        "abcd": [c for c in cols if c.startswith(("asym_", "mask_", "contrast_"))],
    }


# The feature sets that are compared (each one = a list of groups).
FEATURE_SETS = {
    "color (30)": ["color_stats", "color_hist"],
    "texture (3)": ["texture"],
    "color+texture (33)": ["color_stats", "color_hist", "texture"],          # = the Session 5 vector
    "color+texture+GLCM (37)": ["color_stats", "color_hist", "texture", "glcm"],
    "all incl. ABCD (45)": ["color_stats", "color_hist", "texture", "glcm", "abcd"],
}


def feature_set_columns(set_name, columns):
    groups = feature_groups(columns)
    return [c for g in FEATURE_SETS[set_name] for c in groups[g]]


# --------------------------------------------------------------------------------------------------
# extraction over the whole dataset (with a cache, because it is the slow part)
# --------------------------------------------------------------------------------------------------
def load_for_features(path, size=FEATURE_SIZE):
    """Open an image and resize it to size x size (same scale for every image)."""
    img = load_rgb(path, min_size=size)
    return np.asarray(img.resize((size, size), Image.BILINEAR))


def extract_dataset(images, image_paths, cache_path=None, recompute=False, every=500):
    """Feature table for every row of `images` (needs an 'isic_id' column).

    image_paths : {isic_id: path} (from data_io.find_images)
    cache_path  : CSV where the table is saved; if it exists (and covers all ids) it is loaded instead.
    Returns a DataFrame indexed by isic_id, one column per feature.
    """
    ids = list(images["isic_id"])
    if cache_path is not None and cache_path.exists() and not recompute:
        cached = pd.read_csv(cache_path, index_col="isic_id")
        if set(ids) <= set(cached.index):
            print(f"loaded cached features for {len(ids)} images from {cache_path.name}")
            return cached.loc[ids]
        print("cache does not cover all images -> recomputing")

    rows, start = [], time.time()
    for k, isic_id in enumerate(ids, 1):
        if isic_id not in image_paths:
            raise FileNotFoundError(f"Image file not found for {isic_id}")
        rows.append({"isic_id": isic_id, **extract_features(load_for_features(image_paths[isic_id]))})
        if k % every == 0 or k == len(ids):
            el = time.time() - start
            print(f"  {k}/{len(ids)} images  ({el:.0f}s elapsed, ~{el / k * (len(ids) - k):.0f}s left)")
    table = pd.DataFrame(rows).set_index("isic_id")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache_path)
        print(f"saved features to {cache_path}")
    return table
