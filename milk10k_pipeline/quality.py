"""
B4 -- Data-quality report (extends the Session 2 metadata audit; does not repeat its tables).

Produces a Markdown report: missing-value decisions, the 2-images-per-lesion checks,
a shortcut check and the list of columns banned as model inputs.
"""
import pandas as pd

# How each column with missing values is handled. Milestone 1 uses IMAGES only as model input,
# so for metadata columns this is the rule that will apply if they are added as inputs later.
MISSING_DECISIONS = {
    "age_approx": "impute with the TRAIN median (only if age is used as an input later)",
    "sex": "replace with an explicit 'unknown' category",
    "anatom_site_general": "explicit 'unknown' category; missingness is informative (NV twice as frequent, Part A Q4), so it is NOT used as input in Milestone 1",
    "skin_tone_class": "leave NaN; used only to analyse bias, not as an input",
    "diagnosis_2": "leave NaN; label column, never an input",
    "diagnosis_3": "leave NaN; label column, never an input",
    "diagnosis_4": "leave NaN; label column, never an input",
    "diagnosis_5": "leave NaN; label column, never an input",
}
DEFAULT_DECISION = "leave NaN; not used as a model input"

# Columns that must NEVER be model inputs, and why.
BANNED_COLUMNS = {
    "diagnosis_1": "the target itself",
    "diagnosis_2": "finer version of the label",
    "diagnosis_3": "finer version of the label",
    "diagnosis_4": "finer version of the label",
    "diagnosis_5": "finer version of the label",
    "dx": "the 11-class label",
    "diagnosis_confirm_type": "result of the diagnostic process (histopathology = suspicious lesion, 72% malignant)",
    "image_manipulation": "acquisition/editing artefact that differs between classes -> shortcut",
    "attribution": "identifies the contributing clinic -> site/acquisition shortcut",
    "copyright_license": "tied to the contributor -> acquisition shortcut",
    "lesion_id": "identifier (needed only for grouping the split)",
    "isic_id": "identifier",
}


def md_table(df, index=True):
    """Render a DataFrame as a Markdown table (no extra package needed)."""
    if index:
        df = df.reset_index()
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def missing_value_table(meta):
    """One line per column that has missing values, with the handling decision."""
    missing = meta.isna().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    return pd.DataFrame({"column": missing.index,
                         "% missing": missing.round(1).values,
                         "decision": [MISSING_DECISIONS.get(c, DEFAULT_DECISION) for c in missing.index]})


def pair_checks(meta):
    """Exactly 2 images per lesion, one of each image type."""
    per_lesion = meta.groupby("lesion_id").agg(n_images=("isic_id", "size"), n_views=("view", "nunique"))
    return {"lesions": len(per_lesion),
            "lesions with exactly 2 images": int((per_lesion["n_images"] == 2).sum()),
            "lesions with 1 dermoscopic + 1 clinical": int(((per_lesion["n_images"] == 2) & (per_lesion["n_views"] == 2)).sum())}


def shortcut_tables(meta):
    """Cross-tabs of image_manipulation and image_type against diagnosis_1 (row % and counts)."""
    tables = {}
    for col in ["image_manipulation", "image_type"]:
        counts = pd.crosstab(meta[col], meta["diagnosis_1"], margins=True)
        pct = (pd.crosstab(meta[col], meta["diagnosis_1"], normalize="index") * 100).round(1)
        tables[col] = (counts, pct)
    return tables


def write_report(meta, path):
    """Write the full data-quality report to a Markdown file and return its text."""
    lines = ["# Data-quality report (B4)", "",
             f"Generated from `metadata.csv`: {len(meta)} images, {meta['lesion_id'].nunique()} lesions.", "",
             "## 1. Missing values and how each column is handled", "",
             md_table(missing_value_table(meta), index=False), "",
             "## 2. Label-consistency checks", ""]
    for name, value in pair_checks(meta).items():
        lines.append(f"- {name}: **{value}**")
    lines += ["- the two images of each lesion have identical age, sex, site and diagnosis fields (checked in Part A, A1.1e)", "",
              "## 3. Shortcut check", ""]

    tables = shortcut_tables(meta)
    for col, (counts, pct) in tables.items():
        lines += [f"### {col} vs diagnosis_1", "", "Counts:", "", md_table(counts), "",
                  "Row percentages:", "", md_table(pct), ""]

    pct_manip = tables["image_manipulation"][1]
    pct_type = tables["image_type"][1]
    lines += ["**Conclusion.**",
              f"- `image_manipulation`: {pct_manip.loc['altered', 'Malignant'] if 'altered' in pct_manip.index else float('nan'):.1f}% of *altered* images are Malignant "
              f"vs {pct_manip.drop(index='altered', errors='ignore')['Malignant'].mean():.1f}% of the others. "
              "The editing status of an image is not biology, so any difference here is a potential shortcut: the column is banned and all images get the same preprocessing.",
              f"- `image_type`: the class percentages are identical for dermoscopic and clinical images "
              f"({pct_type['Malignant'].min():.1f}% vs {pct_type['Malignant'].max():.1f}% Malignant), because every lesion has exactly one image of each type. "
              "Image type carries no label information by construction, so it is not a shortcut.", "",
              "## 4. Columns NOT used as model inputs", "",
              md_table(pd.DataFrame({"column": list(BANNED_COLUMNS), "reason": list(BANNED_COLUMNS.values())}), index=False), ""]
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text
