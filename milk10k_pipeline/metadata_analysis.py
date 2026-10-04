"""
Part 1 — Metadata analysis: correlation with target.

Functions here answer: which metadata fields (sex, anatomical site, age,
image type, ...) are actually associated with the diagnosis label, and
which look like noise (or worse, a leakage / bias risk)?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------
# Task 1: column inventory
# --------------------------------------------------------------------------

def load_metadata(data_dir) -> pd.DataFrame:
    """Load MILK10k's metadata.csv (one row per image)."""
    from pathlib import Path
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / "metadata.csv")


def summarize_columns(
    df: pd.DataFrame,
    id_cols=("isic_id", "lesion_id"),
    target_col: str = "diagnosis_1",
) -> pd.DataFrame:
    """
    Task 1. For every column that isn't an id or the target, report its
    inferred type (categorical / numeric / boolean / free_text) and the
    fraction of missing values.
    """
    rows = []
    for col in df.columns:
        if col in id_cols or col == target_col:
            continue

        series = df[col]
        missing_frac = series.isna().mean()

        if pd.api.types.is_bool_dtype(series):
            inferred = "boolean"
        elif pd.api.types.is_numeric_dtype(series):
            inferred = "numeric"
        elif series.nunique(dropna=True) <= max(20, int(0.05 * len(series))):
            inferred = "categorical"
        else:
            inferred = "free_text"

        rows.append(
            {
                "column": col,
                "inferred_type": inferred,
                "missing_fraction": round(missing_frac, 4),
                "n_unique": series.nunique(dropna=True),
            }
        )

    return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)


# --------------------------------------------------------------------------
# Task 2: categorical fields vs. target
# --------------------------------------------------------------------------

@dataclass
class CategoricalAssociation:
    column: str
    crosstab: pd.DataFrame          # counts, category x target class
    proportions: pd.DataFrame        # row-normalized (within-category class mix)
    chi2: float
    p_value: float
    dof: int


def categorical_vs_target(
    df: pd.DataFrame, column: str, target_col: str = "diagnosis_1"
) -> CategoricalAssociation:
    """
    Task 2/3 (categorical branch). Cross-tabulate a categorical metadata
    field against the target, and run a chi-square test of independence.

    Why chi-square: both variables are categorical, and we want to know
    whether the joint distribution differs from what we'd expect if the
    field and the diagnosis were independent. Chi-square is the standard
    test for exactly that (categorical x categorical) association.
    """
    subset = df[[column, target_col]].dropna()
    crosstab = pd.crosstab(subset[column], subset[target_col])
    proportions = crosstab.div(crosstab.sum(axis=1), axis=0)

    chi2, p_value, dof, _expected = stats.chi2_contingency(crosstab)

    return CategoricalAssociation(
        column=column,
        crosstab=crosstab,
        proportions=proportions,
        chi2=chi2,
        p_value=p_value,
        dof=dof,
    )


def plot_categorical_vs_target(assoc: CategoricalAssociation, ax=None):
    """Grouped/stacked bar chart of class proportions within each category."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    assoc.proportions.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Proportion of images")
    ax.set_title(
        f"{assoc.column} vs. target "
        f"(chi2={assoc.chi2:.1f}, p={assoc.p_value:.4f})"
    )
    ax.legend(title="diagnosis", bbox_to_anchor=(1.02, 1), loc="upper left")
    return ax


# --------------------------------------------------------------------------
# Task 3: numeric fields vs. target
# --------------------------------------------------------------------------

@dataclass
class NumericAssociation:
    column: str
    per_class_stats: pd.DataFrame   # mean/std/count per class
    f_stat: float
    p_value: float


def numeric_vs_target(
    df: pd.DataFrame, column: str, target_col: str = "diagnosis_1"
) -> NumericAssociation:
    """
    Task 3 (numeric branch). Compare a numeric field's distribution across
    target classes and run a one-way ANOVA.

    Why ANOVA: the field is numeric, the target is categorical with (2+)
    groups, and we want to know if the group means differ more than we'd
    expect by chance. One-way ANOVA is the standard test for a numeric
    variable across >=2 categorical groups (for exactly 2 groups this is
    equivalent to a two-sample t-test).
    """
    subset = df[[column, target_col]].dropna()
    groups = [g[column].values for _, g in subset.groupby(target_col) if len(g) > 0]

    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        raise ValueError(
            f"'{column}' has too few non-missing values per class to run ANOVA "
            f"(need >=2 classes with >=2 values each after dropping NaNs)."
        )

    f_stat, p_value = stats.f_oneway(*groups)

    per_class_stats = subset.groupby(target_col)[column].agg(
        ["count", "mean", "std", "median"]
    )

    return NumericAssociation(
        column=column,
        per_class_stats=per_class_stats,
        f_stat=f_stat,
        p_value=p_value,
    )


def plot_numeric_by_class(
    df: pd.DataFrame, column: str, target_col: str = "diagnosis_1", ax=None, kind="box"
):
    """Boxplot (default) or overlaid-histogram view of a numeric field per class."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    subset = df[[column, target_col]].dropna()

    if kind == "box":
        subset.boxplot(column=column, by=target_col, ax=ax)
        ax.set_title(f"{column} by {target_col}")
        ax.figure.suptitle("")
    elif kind == "hist":
        for cls, g in subset.groupby(target_col):
            ax.hist(g[column], bins=20, alpha=0.5, label=str(cls))
        ax.set_xlabel(column)
        ax.set_ylabel("count")
        ax.legend()
        ax.set_title(f"{column} distribution by {target_col}")
    else:
        raise ValueError("kind must be 'box' or 'hist'")

    return ax


# --------------------------------------------------------------------------
# Task 4: summary helper
# --------------------------------------------------------------------------

def rank_associations(
    categorical_results: list[CategoricalAssociation],
    numeric_results: list[NumericAssociation],
) -> pd.DataFrame:
    """
    Task 4. Combine every field's association test into one ranked table
    (lower p-value = stronger apparent association with the target).

    NOTE: p-values from chi-square and ANOVA aren't perfectly comparable
    (different tests, different assumptions, and both are sensitive to
    sample size), so treat this as a rough triage ranking, not a precise
    effect-size comparison -- say so in your report.
    """
    rows = []
    for r in categorical_results:
        rows.append({"column": r.column, "test": "chi-square", "p_value": r.p_value})
    for r in numeric_results:
        rows.append({"column": r.column, "test": "ANOVA", "p_value": r.p_value})

    out = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    out["likely_informative"] = out["p_value"] < 0.05
    return out
