"""
Milestone 1 -- the whole data pipeline (B1 to B8) in one command:

    python run_milestone1.py

Reads the data folder from milk10k_pipeline/config.py (or the MILK10K_DIR environment variable)
and writes everything to outputs/:
    image_size_summary.csv, image_sizes.csv            (B1)
    figures/class_distribution.png, figures/dx_vs_diagnosis1.png,
    figures/class_gallery.png, session2_findings.csv   (B2)
    label_map.json                                     (B3)
    data_quality_report.md                             (B4)
    splits/train.csv, val.csv, test.csv, split_info.json, split_class_proportions.csv  (B5)
    normalization_stats.json, figures/augmentation_examples.png  (B6, B7)
    class_weights.json, figures/train_batch.png        (B8)
    run_log.txt                                        (everything printed below)
"""
import json
import sys
from datetime import date

import matplotlib
matplotlib.use("Agg")            # save figures to files, never open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from milk10k_pipeline import color_analysis, config, data_loader, integrity, labels, quality, splits, transforms, visualizer
from milk10k_pipeline.data_io import find_images, load_rgb, load_tables


class Tee:
    """Print to the screen AND to outputs/run_log.txt."""
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
    def flush(self):
        self.stdout.flush()
        self.file.flush()


def header(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def save_fig(name):
    plt.savefig(config.FIGURES_DIR / name, dpi=130, bbox_inches="tight")
    plt.close("all")
    print(f"  saved figure: outputs/figures/{name}")


def main():
    for d in [config.OUTPUT_DIR, config.FIGURES_DIR, config.SPLITS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(config.OUTPUT_DIR / "run_log.txt")
    summary = {"date": str(date.today()), "seed": config.SEED}

    print(f"Data folder: {config.DATA_DIR}")
    meta, gt, class_names = load_tables()
    paths = find_images()
    print(f"metadata: {len(meta)} images | training_gt: {len(gt)} lesions | image files found: {len(paths)}")

    # ------------------------------------------------------------------ B1
    header("B1. Full-dataset integrity check")
    checks = integrity.check_images(meta, paths)
    checks.to_csv(config.OUTPUT_DIR / "image_sizes.csv", index=False)
    sizes = integrity.size_summary(checks)
    sizes.to_csv(config.OUTPUT_DIR / "image_size_summary.csv", index=False)
    print(sizes.to_string(index=False))
    missing = checks.loc[~checks["exists"], "isic_id"].tolist()
    unreadable = checks.loc[checks["exists"] & ~checks["readable"], ["isic_id", "error"]]
    print(f"  rows without an image file: {len(missing)} {missing[:20]}")
    print(f"  unreadable files (Image.verify failed): {len(unreadable)}")
    if len(unreadable):
        print(unreadable.to_string(index=False))
    if missing or len(unreadable):
        raise RuntimeError("Integrity check failed: fix the missing/unreadable images listed above before continuing.")
    summary["B1"] = {"n_images": len(meta), "missing": len(missing), "unreadable": len(unreadable)}

    # ------------------------------------------------------------------ B2
    header("B2. Label-centred EDA")
    lesions = labels.build_lesion_table(meta)
    n_d1 = lesions["diagnosis_1"].value_counts()
    n_dx = lesions["dx"].value_counts()
    print("Lesions per diagnosis_1:\n" + n_d1.to_string())
    print("Lesions per 11-class label:\n" + n_dx.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    n_d1.plot(kind="bar", ax=axes[0], color="#F08200")
    axes[0].set_title("diagnosis_1 (lesions)"); axes[0].set_ylabel("number of lesions")
    for i, v in enumerate(n_d1.values):
        axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    n_dx.plot(kind="bar", ax=axes[1], color="#F08200", logy=True)
    axes[1].set_title("11-class label (lesions, log scale)"); axes[1].set_ylabel("number of lesions (log)")
    for i, v in enumerate(n_dx.values):
        axes[1].text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    save_fig("class_distribution.png")

    mapping = pd.crosstab(lesions["dx"], lesions["diagnosis_1"])
    mapping.to_csv(config.OUTPUT_DIR / "dx_vs_diagnosis1.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(np.log1p(mapping.values), cmap="Oranges")
    ax.set_xticks(range(mapping.shape[1]), mapping.columns); ax.set_yticks(range(mapping.shape[0]), mapping.index)
    for (r, c), v in np.ndenumerate(mapping.values):
        ax.text(c, r, str(v), ha="center", va="center", fontsize=8)
    ax.set_title("11-class label -> diagnosis_1 (lesion counts)")
    save_fig("dx_vs_diagnosis1.png")
    print("Mapping 11-class -> diagnosis_1:\n" + mapping.to_string())

    # Session 2 findings re-checked on the full data
    derm = meta[meta["view"] == "dermoscopic"]
    colour = color_analysis.color_summary_stats(derm, config.IMG_DIR, target_col="diagnosis_1", n_per_class=300, seed=config.SEED)
    colour["brightness"] = colour[["R_mean", "G_mean", "B_mean"]].mean(axis=1)
    bm = colour[colour["diagnosis_1"].isin(["Benign", "Malignant"])]
    auc_brightness = roc_auc_score(bm["diagnosis_1"] == "Malignant", bm["brightness"])
    mean_brightness = colour.groupby("diagnosis_1")["brightness"].mean()

    age = lesions.groupby("diagnosis_1")["age_approx"].mean()
    site = lesions.assign(site=lesions["anatom_site_general"].fillna("missing"))
    pct_mal_site = (site.groupby("site")["diagnosis_1"].apply(lambda s: (s == "Malignant").mean() * 100)).round(1)
    nv_with = (lesions.loc[lesions["anatom_site_general"].notna(), "dx"] == "NV").mean() * 100
    nv_without = (lesions.loc[lesions["anatom_site_general"].isna(), "dx"] == "NV").mean() * 100
    darker = mean_brightness.get("Indeterminate", np.nan) < mean_brightness.drop("Indeterminate", errors="ignore").min()

    findings = pd.DataFrame([
        {"Session 2 finding": "Severe class imbalance (BCC dominant, MAL_OTH tiny)",
         "full dataset": f"BCC {n_dx.max()} lesions vs MAL_OTH {n_dx.min()} (ratio {n_dx.max() / n_dx.min():.0f}:1); Malignant {n_d1.get('Malignant', 0)} vs Benign {n_d1.get('Benign', 0)}",
         "still true?": "yes",
         "consequence for the pipeline": "stratified grouped split, class-weighted loss + WeightedRandomSampler, balanced metrics instead of accuracy"},
        {"Session 2 finding": "Malignant lesions come from older patients (mild risk factor)",
         "full dataset": f"mean age Malignant {age.get('Malignant', np.nan):.1f} vs Benign {age.get('Benign', np.nan):.1f} years",
         "still true?": "yes" if age.get("Malignant", 0) > age.get("Benign", 0) else "no",
         "consequence for the pipeline": "genuine risk factor, not a shortcut; may be added as a metadata input later (train-median imputation); image-only in Milestone 1"},
        {"Session 2 finding": "Anatomical site is associated with the diagnosis",
         "full dataset": "% Malignant by site: " + ", ".join(f"{k} {v}" for k, v in pct_mal_site.items())
                         + f"; NV is {nv_without:.1f}% of lesions WITHOUT site vs {nv_with:.1f}% with site",
         "still true?": "yes, and missing site is itself informative",
         "consequence for the pipeline": "missingness is a dataset artefact -> potential shortcut; site NOT used as an input in Milestone 1"},
        {"Session 2 finding": "Mean colour/brightness alone does not separate Benign from Malignant",
         "full dataset": f"ROC-AUC of mean brightness (dermoscopic, {len(bm)} images) = {auc_brightness:.3f}",
         "still true?": "yes" if 0.35 < auc_brightness < 0.65 else "no -- colour is more informative than expected",
         "consequence for the pipeline": "need a CNN learning texture/shape; colour augmentation kept mild (A3.5) so the small colour signal is preserved"},
        {"Session 2 finding": "Indeterminate images looked darker (small sample)",
         "full dataset": "mean brightness: " + ", ".join(f"{k} {v:.1f}" for k, v in mean_brightness.items()),
         "still true?": "yes" if darker else "no -- it was a small-sample effect",
         "consequence for the pipeline": "no special treatment; Indeterminate is merged into Malignant anyway (B3)"},
    ])
    findings.to_csv(config.OUTPUT_DIR / "session2_findings.csv", index=False)
    print("Session 2 findings on the full dataset:")
    for _, row in findings.iterrows():
        print(f"  - {row['Session 2 finding']}: {row['full dataset']} -> still true? {row['still true?']}")

    gallery = [derm[derm["dx"] == c].sample(1, random_state=config.SEED).iloc[0] for c in labels.ELEVEN_CLASSES if (derm["dx"] == c).any()]
    gallery_imgs = [np.array(load_rgb(paths[r["isic_id"]]).resize((256, 192))) for r in gallery]
    gallery_titles = [f"{r['dx']} ({r['diagnosis_1']})" for r in gallery]
    visualizer.show_image_grid(gallery_imgs, gallery_titles, n=len(gallery_imgs), ncols=4, title="One dermoscopic example per class")
    save_fig("class_gallery.png")
    summary["B2"] = {"diagnosis_1_counts": n_d1.to_dict(), "dx_counts": n_dx.to_dict(), "auc_brightness": round(auc_brightness, 3)}

    # ------------------------------------------------------------------ B3
    header("B3. Label strategy")
    labels.save_label_map(config.OUTPUT_DIR / "label_map.json")
    label_map = labels.load_label_map(config.OUTPUT_DIR / "label_map.json")
    print(json.dumps(label_map, indent=2))
    print("Primary target after merging:\n" + lesions["label"].map(dict(enumerate(labels.PRIMARY["class_names"]))).value_counts().to_string())

    # ------------------------------------------------------------------ B4
    header("B4. Data-quality report")
    quality.write_report(meta, config.OUTPUT_DIR / "data_quality_report.md")
    print(quality.missing_value_table(meta).to_string(index=False))
    print(quality.pair_checks(meta))
    for col, (counts, pct) in quality.shortcut_tables(meta).items():
        print(f"{col} vs diagnosis_1 (row %):\n{pct.to_string()}")
    print("  saved outputs/data_quality_report.md")

    # ------------------------------------------------------------------ B5
    header("B5. Splits (lesion level, grouped + stratified on the 11-class label)")
    train_ids, val_ids, test_ids = splits.split_lesions(lesions, config.VAL_SIZE, config.TEST_SIZE, config.SEED, config.SPLIT_LABEL)
    images = splits.assign_images(meta, train_ids, val_ids, test_ids)
    images["label"] = images["diagnosis_1"].map(labels.PRIMARY["mapping"])
    props, max_dev = splits.verify_splits(images)
    print("11-class proportions per split (% of lesions):\n" + props.to_string())
    print(f"  maximum deviation from the global proportions: {max_dev} percentage points")
    d1_props = pd.crosstab(images.drop_duplicates("lesion_id")["diagnosis_1"], images.drop_duplicates("lesion_id")["split"], normalize="columns") * 100
    print("diagnosis_1 proportions per split (%):\n" + d1_props.round(2).to_string())
    props.to_csv(config.OUTPUT_DIR / "split_class_proportions.csv")

    keep_cols = ["isic_id", "lesion_id", "view", "image_type", "diagnosis_1", "dx", "label", "age_approx", "sex", "anatom_site_general", "split"]
    for name in ["train", "val", "test"]:
        part = images.loc[images["split"] == name, keep_cols].sort_values(["lesion_id", "view"])
        part.to_csv(config.SPLITS_DIR / f"{name}.csv", index=False)
        print(f"  saved outputs/splits/{name}.csv: {part['lesion_id'].nunique()} lesions, {len(part)} images")
    split_info = {"seed": config.SEED, "date": str(date.today()), "val_size": config.VAL_SIZE, "test_size": config.TEST_SIZE,
                  "stratified_on": config.SPLIT_LABEL, "grouped_by": "lesion_id",
                  "n_lesions": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
                  "max_class_deviation_pp": max_dev}
    with open(config.SPLITS_DIR / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    summary["B5"] = split_info

    # ------------------------------------------------------------------ B6
    header("B6. Input resolution and views")
    dsizes = checks[checks["view"] == "dermoscopic"]
    short_side = dsizes[["width", "height"]].min(axis=1)
    pct_256 = (short_side >= config.RESIZE).mean() * 100
    pct_224 = (short_side >= config.IMAGE_SIZE).mean() * 100
    print(f"  dermoscopic images: shorter side min {short_side.min()}, median {short_side.median():.0f}, max {short_side.max()}")
    print(f"  {pct_256:.1f}% have shorter side >= {config.RESIZE} and {pct_224:.1f}% >= {config.IMAGE_SIZE} -> resizing only DOWN-samples")
    print(f"  views used: {config.VIEWS} (option b: one dermoscopic image per lesion -> 1 image = 1 lesion)")
    summary["B6"] = {"pct_short_side_ge_256": round(pct_256, 1), "pct_short_side_ge_224": round(pct_224, 1),
                     "short_side_median": float(short_side.median())}

    # ------------------------------------------------------------------ B7
    header("B7. Transforms: normalization stats (train only), determinism, augmentation figure")
    stats_ds = data_loader.MILK10kDataset(config.SPLITS_DIR / "train.csv", label_map=label_map,
                                          transform=transforms.build_eval_transform(normalize=False), image_paths=paths)
    mean, std = data_loader.compute_normalization_stats(stats_ds)
    with open(config.OUTPUT_DIR / "normalization_stats.json", "w") as f:
        json.dump({"mean": mean, "std": std, "computed_on": f"train split, {len(stats_ds)} dermoscopic images, after Resize({config.RESIZE}) + CenterCrop({config.IMAGE_SIZE})"}, f, indent=2)
    print(f"  train mean {mean}, std {std} -> saved outputs/normalization_stats.json")

    train_tf = transforms.build_train_transform(mean, std)
    eval_tf = transforms.build_eval_transform(mean, std)
    sample_img = load_rgb(paths[images.loc[images["view"] == "dermoscopic", "isic_id"].iloc[0]])
    a, b = eval_tf(sample_img), eval_tf(sample_img)
    assert torch.equal(a, b), "eval_transform is not deterministic!"
    print(f"  eval_transform applied twice -> identical tensors: {torch.equal(a, b)}")

    torch.manual_seed(config.SEED)
    show_tf = transforms.build_train_transform(normalize=False)
    rare = "VASC" if (derm["dx"] == "VASC").any() else labels.ELEVEN_CLASSES[-1]
    fig, axes = plt.subplots(3, 8, figsize=(18, 7))
    for r, cls in enumerate(["BCC", "NV", rare]):
        row = derm[derm["dx"] == cls].sample(1, random_state=config.SEED).iloc[0]
        img = load_rgb(paths[row["isic_id"]])
        views = [transforms.build_eval_transform(normalize=False)(img)] + [show_tf(img) for _ in range(7)]
        for c, t in enumerate(views):
            axes[r, c].imshow(t.permute(1, 2, 0).numpy()); axes[r, c].axis("off")
            axes[r, c].set_title(f"{cls} original" if c == 0 else f"augmented {c}", fontsize=9)
    save_fig("augmentation_examples.png")

    # ------------------------------------------------------------------ B8
    header("B8. Dataset / DataLoaders, imbalance handling, sanity checks")
    datasets, loaders = data_loader.make_dataloaders(label_map, train_tf, eval_tf)
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} images, {len(loaders[name])} batches of {config.BATCH_SIZE}")

    n_cls = len(label_map["primary"]["class_names"])
    train_labels = datasets["train"].labels
    counts = np.bincount(train_labels, minlength=n_cls)
    weights_primary = data_loader.compute_class_weights(train_labels, n_cls)
    train_table = datasets["train"].table
    dx_idx = train_table["dx"].map(label_map["stretch_11_class"]["mapping"]).tolist()
    weights_11 = data_loader.compute_class_weights(dx_idx, len(labels.ELEVEN_CLASSES))
    class_weights = {
        "primary": dict(zip(label_map["primary"]["class_names"], [round(w, 4) for w in weights_primary])),
        "stretch_11_class": dict(zip(labels.ELEVEN_CLASSES, [round(w, 4) for w in weights_11])),
        "formula": "n_train_images / (n_classes * count_of_class), computed on the TRAIN split only",
    }
    with open(config.OUTPUT_DIR / "class_weights.json", "w") as f:
        json.dump(class_weights, f, indent=2)
    imbalance = counts.max() / counts.min()
    dx_counts = np.bincount(dx_idx, minlength=len(labels.ELEVEN_CLASSES))
    print(f"  train class counts {dict(zip(label_map['primary']['class_names'], counts.tolist()))} -> imbalance ratio {imbalance:.2f}:1")
    print(f"  11-class train imbalance ratio: {dx_counts.max()}:{dx_counts.min()} = {dx_counts.max() / dx_counts.min():.0f}:1")
    print(f"  class weights -> saved outputs/class_weights.json: {class_weights['primary']}")

    images_b, labels_b, ids_b = next(iter(loaders["train"]))
    print(f"  train batch: images {tuple(images_b.shape)} {images_b.dtype}, min {images_b.min():.3f}, max {images_b.max():.3f}, "
          f"labels {tuple(labels_b.shape)} {labels_b.dtype}, first ids {list(ids_b[:3])}")
    vb = next(iter(loaders["val"]))
    print(f"  val batch:   images {tuple(vb[0].shape)}, min {vb[0].min():.3f}, max {vb[0].max():.3f}")

    hist = np.zeros(n_cls, dtype=int)
    for i, (_, lab, _) in enumerate(loaders["train"]):
        hist += np.bincount(lab.numpy(), minlength=n_cls)
        if i == 19:
            break
    print(f"  label histogram of 20 train batches WITH WeightedRandomSampler: {dict(zip(label_map['primary']['class_names'], hist.tolist()))}")
    print(f"  (without the sampler the expected share would be {dict(zip(label_map['primary']['class_names'], (counts / counts.sum()).round(3).tolist()))})")

    epoch_s = data_loader.time_one_epoch(loaders["train"])
    print(f"  time to load one train epoch ({len(datasets['train'])} images, num_workers={config.NUM_WORKERS}): {epoch_s:.1f} s")

    visualizer.show_tensor_batch(images_b, labels_b, label_map["primary"]["class_names"], mean, std, n=16, ncols=4,
                                 title="16 training images after augmentation (WeightedRandomSampler)")
    save_fig("train_batch.png")

    summary["B8"] = {"train_counts": counts.tolist(), "imbalance_ratio_train": round(float(imbalance), 2),
                     "imbalance_ratio_train_11class": round(float(dx_counts.max() / dx_counts.min()), 1),
                     "class_weights": class_weights["primary"], "hist_20_batches": hist.tolist(),
                     "epoch_seconds": round(epoch_s, 1), "mean": mean, "std": std}
    with open(config.OUTPUT_DIR / "milestone1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    header("DONE -- all outputs are in the outputs/ folder")


if __name__ == "__main__":
    main()
