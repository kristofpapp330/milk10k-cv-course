"""
Run the full Session 2/3 homework pipeline against the REAL MILK10k dataset
and save every figure + table you need for the written report into
./report_assets/.

Usage (from the project root, with milk10k/ already downloaded):

    python run_analysis.py

Edit DATA_DIR below if your dataset lives somewhere else.
"""
import matplotlib
matplotlib.use("Agg")  # save figures to disk; no interactive display needed
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from milk10k_pipeline import metadata_analysis as ma
from milk10k_pipeline import color_analysis as ca
from milk10k_pipeline import preprocessing as pp
from milk10k_pipeline import data_loader as dl
from milk10k_pipeline import visualizer as viz

# ---------------------------------------------------------------------
DATA_DIR = Path("data/milk10k")          # <-- adjust if needed
IMG_DIR = DATA_DIR / "images"
TARGET_COL = "diagnosis_1"
OUT_DIR = Path("report_assets")
OUT_DIR.mkdir(exist_ok=True)
N_PER_CLASS = 25                          # Part 2 sample size per class
# ---------------------------------------------------------------------

print("Loading metadata...")
df = ma.load_metadata(DATA_DIR)
df = dl.filter_to_available_images(df, IMG_DIR)  # only rows with a real file
print(f"{len(df)} images available locally.")

# ======================================================================
# PART 1 — Metadata analysis
# ======================================================================
print("\n--- Part 1: metadata analysis ---")

summary = ma.summarize_columns(df, target_col=TARGET_COL)
summary.to_csv(OUT_DIR / "part1_column_summary.csv", index=False)
print(summary)

categorical_cols = summary.loc[summary.inferred_type == "categorical", "column"].tolist()
numeric_cols = summary.loc[summary.inferred_type == "numeric", "column"].tolist()
print("Categorical columns detected:", categorical_cols)
print("Numeric columns detected:", numeric_cols)

cat_results, num_results = [], []

for col in categorical_cols:
    try:
        res = ma.categorical_vs_target(df, col, TARGET_COL)
    except ValueError:
        continue  # e.g. column with only 1 unique value
    cat_results.append(res)
    ma.plot_categorical_vs_target(res)
    plt.savefig(OUT_DIR / f"part1_categorical_{col}.png", bbox_inches="tight")
    plt.close()
    res.crosstab.to_csv(OUT_DIR / f"part1_crosstab_{col}.csv")

for col in numeric_cols:
    if df[col].notna().sum() < 2 or df.dropna(subset=[col])[TARGET_COL].nunique() < 2:
        print(f"  skipping '{col}': not enough non-missing data / classes for ANOVA")
        continue
    try:
        res = ma.numeric_vs_target(df, col, TARGET_COL)
    except (ValueError, TypeError) as e:
        print(f"  skipping '{col}': {e}")
        continue
    num_results.append(res)
    ma.plot_numeric_by_class(df, col, TARGET_COL, kind="box")
    plt.savefig(OUT_DIR / f"part1_numeric_{col}.png", bbox_inches="tight")
    plt.close()
    res.per_class_stats.to_csv(OUT_DIR / f"part1_stats_{col}.csv")

ranking = ma.rank_associations(cat_results, num_results)
ranking.to_csv(OUT_DIR / "part1_ranking.csv", index=False)
print("\nAssociation ranking (lower p = stronger apparent association):")
print(ranking)

# ======================================================================
# PART 2 — Color and histogram analysis
# ======================================================================
print("\n--- Part 2: color analysis ---")

hist_by_class = ca.histograms_by_class(df, IMG_DIR, TARGET_COL, n_per_class=N_PER_CLASS)
ca.plot_avg_grayscale_by_class(hist_by_class)
plt.savefig(OUT_DIR / "part2_avg_grayscale_by_class.png", bbox_inches="tight")
plt.close()

ca.plot_avg_rgb_by_class(hist_by_class)
plt.savefig(OUT_DIR / "part2_avg_rgb_by_class.png", bbox_inches="tight")
plt.close()

stats_df = ca.color_summary_stats(df, IMG_DIR, TARGET_COL, n_per_class=N_PER_CLASS)
stats_df.to_csv(OUT_DIR / "part2_color_stats_per_image.csv", index=False)

agg = stats_df.groupby(TARGET_COL)[["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"]].mean()
agg.to_csv(OUT_DIR / "part2_color_stats_aggregated.csv")
print(agg)

ca.plot_color_stats_boxplots(stats_df, TARGET_COL)
plt.savefig(OUT_DIR / "part2_color_boxplots.png", bbox_inches="tight")
plt.close()

# ======================================================================
# PART 3 — Preprocessing: before/after demo
# ======================================================================
print("\n--- Part 3: preprocessing demo ---")

demo_rows = df.sample(4, random_state=0)
raw_imgs, proc_imgs = [], []
for _, row in demo_rows.iterrows():
    path = IMG_DIR / f"{row['isic_id']}.jpg"
    raw, _ = pp.preprocess_image(path, size=(224, 224), normalization="none")
    proc, info = pp.preprocess_image(path, size=(224, 224), normalization="minmax")
    raw_imgs.append(raw)
    proc_imgs.append(proc)

viz.plot_batch_summary(proc_imgs, raw_images=raw_imgs)
plt.savefig(OUT_DIR / "part3_before_after.png", bbox_inches="tight")
plt.close()

batch_result = pp.preprocess_batch(
    df.sample(50, random_state=0), img_dir=IMG_DIR, label_col=TARGET_COL, size=(224, 224)
)
print(f"Batch preprocessing: {len(batch_result.ids)} succeeded, "
      f"{len(batch_result.skipped)} skipped.")
if batch_result.skipped:
    print("Skipped:", batch_result.skipped)

# ======================================================================
# PART 4 — Data loader
# ======================================================================
print("\n--- Part 4: data loader ---")

loader = dl.MILK10kDataLoader(
    metadata=df, img_dir=IMG_DIR, target_col=TARGET_COL,
    batch_size=32, size=(224, 224), normalization="minmax", shuffle=True, seed=0,
)
print(f"Loader: {len(df)} images available, {len(loader)} batches per epoch.")

first_batch = next(iter(loader))
images, labels = first_batch
print("First batch shape:", images.shape, "labels sample:", labels[:5])

# ======================================================================
# PART 5 — Visualizer
# ======================================================================
print("\n--- Part 5: visualizer ---")

viz.show_image_grid(first_batch, n=6, title="Sample batch from MILK10kDataLoader")
plt.savefig(OUT_DIR / "part5_loader_sample_grid.png", bbox_inches="tight")
plt.close()

ax, counts = viz.plot_class_balance(df, TARGET_COL)
plt.savefig(OUT_DIR / "part5_class_balance.png", bbox_inches="tight")
plt.close()
counts.to_csv(OUT_DIR / "part5_class_counts.csv")
print(counts)

viz.plot_batch_summary(images)
plt.savefig(OUT_DIR / "part5_batch_pixel_distribution.png", bbox_inches="tight")
plt.close()

# ======================================================================
# Save a small JSON summary for quick reference while writing the report
# ======================================================================
summary_json = {
    "n_images_available": len(df),
    "class_counts": counts.to_dict(),
    "most_associated_fields": ranking.head(3)["column"].tolist(),
    "least_associated_fields": ranking.tail(3)["column"].tolist(),
    "skipped_in_batch_demo": len(batch_result.skipped),
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary_json, f, indent=2, default=str)

print(f"\nAll figures and tables saved to {OUT_DIR.resolve()}")
print("Open that folder and pull the plots/tables you need into your report.")
