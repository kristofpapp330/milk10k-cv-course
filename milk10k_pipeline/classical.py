"""
Milestone 2 -- Classical ML baseline (Session 5): models, model selection and evaluation.

Recipe (Session 5 "The baseline recipe"):
  1. split by lesion (Milestone 1 splits, re-used unchanged)
  2. majority-class model = the floor every real model must beat
  3. logistic regression on standardised features
  4. stronger models (kNN, random forest, SVM), compared on VALIDATION
  5. the chosen configuration is evaluated on TEST exactly once

Every model is a scikit-learn Pipeline:  median imputer -> StandardScaler -> classifier.
The imputer and the scaler are fitted inside .fit(), i.e. on the TRAIN split only, and then only
applied to validation/test (Session 3/5: no leakage).
"""
import itertools
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .features import FEATURE_SETS, feature_set_columns

# --------------------------------------------------------------------------------------------------
# models and the hyper-parameter values tried for each (small grids, chosen on VALIDATION)
# --------------------------------------------------------------------------------------------------
PARAM_GRIDS = {
    "majority":       {},                                                   # no parameters
    "logistic_reg":   {"C": [0.01, 0.1, 1, 10]},                            # C = 1 / regularisation strength
    "knn":            {"n_neighbors": [5, 15, 31, 51], "weights": ["uniform", "distance"]},
    "random_forest":  {"max_depth": [None, 12], "min_samples_leaf": [1, 5]},
    "svm_rbf":        {"C": [0.3, 1, 3, 10]},
}


def build_model(name, params, seed=config.SEED):
    """One untrained pipeline. class_weight='balanced' wherever the model supports it (kNN does not)."""
    if name == "majority":
        clf = DummyClassifier(strategy="most_frequent")          # always predicts the most common TRAIN class
    elif name == "logistic_reg":
        clf = LogisticRegression(class_weight="balanced", max_iter=5000, **params)
    elif name == "knn":
        clf = KNeighborsClassifier(**params)
    elif name == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1,
                                     random_state=seed, **params)
    elif name == "svm_rbf":
        clf = SVC(kernel="rbf", gamma="scale", class_weight="balanced", random_state=seed, **params)
    else:
        raise ValueError(f"unknown model {name}")
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)


def param_combinations(name):
    """All parameter dicts of the grid of one model, e.g. [{'C': 0.01}, {'C': 0.1}, ...]."""
    grid = PARAM_GRIDS[name]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def scores_of(model, X):
    """A continuous score per image (higher = more likely the positive / each class).

    binary: P(class 1) or the SVM decision value -> used for ROC-AUC and threshold tuning.
    multi-class: the class-probability matrix if available, else None.
    """
    n_classes = len(model.classes_)
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            return proba[:, 1] if n_classes == 2 else proba
        except AttributeError:       # SVC without probability=True
            pass
    if n_classes == 2:
        return model.decision_function(X)
    return None


# --------------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------------
def binary_metrics(y, pred, score=None):
    """Binary task: 1 = Malignant (positive = 'needs clinical action'), 0 = Benign.

    macro_f1      main metric (Session 5): plain average of the F1 of both classes
    balanced_acc  average recall of both classes (= (sensitivity + specificity) / 2)
    sensitivity   recall of Malignant: share of malignant lesions that are caught
    specificity   recall of Benign: share of benign lesions correctly left alone
    roc_auc       ranking quality of the score, independent of the 0.5 threshold
    """
    out = {
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "accuracy": accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y, pred, pos_label=0, zero_division=0),
    }
    out["roc_auc"] = roc_auc_score(y, score) if score is not None and len(np.unique(score)) > 1 else np.nan
    return out


def multiclass_metrics(y, pred, class_names):
    """3-class task: macro-F1, balanced accuracy, accuracy and the recall of every class."""
    labels = list(range(len(class_names)))
    out = {
        "macro_f1": f1_score(y, pred, labels=labels, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "accuracy": accuracy_score(y, pred),
    }
    rec = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    for name, r in zip(class_names, rec):
        out[f"recall_{name}"] = r
    return out


def evaluate(model, X, y, class_names):
    pred = model.predict(X)
    if len(class_names) == 2:
        return binary_metrics(y, pred, scores_of(model, X)), pred
    return multiclass_metrics(y, pred, class_names), pred


# --------------------------------------------------------------------------------------------------
# the comparison grid: every feature set x every model x every parameter value, scored on VALIDATION
# --------------------------------------------------------------------------------------------------
def run_grid(feats, train, val, label_col, class_names, feature_sets=None, models=None, verbose=True):
    """Train on TRAIN, score on VALIDATION, for every configuration. Never touches the test split.

    feats : feature table indexed by isic_id;  train/val : split tables with isic_id and label_col.
    Returns one row per configuration with all validation metrics.
    """
    feature_sets = feature_sets or list(FEATURE_SETS)
    models = models or list(PARAM_GRIDS)
    y_tr, y_va = train[label_col].to_numpy(), val[label_col].to_numpy()
    rows = []
    for fs in feature_sets:
        cols = feature_set_columns(fs, feats.columns)
        X_tr, X_va = feats.loc[train["isic_id"], cols].to_numpy(), feats.loc[val["isic_id"], cols].to_numpy()
        for name in models:
            for params in param_combinations(name):
                start = time.time()
                model = build_model(name, params).fit(X_tr, y_tr)
                metrics, _ = evaluate(model, X_va, y_va, class_names)
                rows.append({"feature_set": fs, "model": name, "params": params,
                             **metrics, "fit_seconds": time.time() - start})
            if verbose:
                best = max(r["macro_f1"] for r in rows if r["feature_set"] == fs and r["model"] == name)
                print(f"  {fs:<26} {name:<14} best val macro-F1 = {best:.3f}")
    return pd.DataFrame(rows)


def best_per_cell(results):
    """For each (feature set, model): the row with the best validation macro-F1 (its best parameters)."""
    idx = results.groupby(["feature_set", "model"], sort=False)["macro_f1"].idxmax()
    return results.loc[idx].reset_index(drop=True)


def select_best(results):
    """The single best configuration on VALIDATION: highest macro-F1, ties broken by balanced accuracy."""
    ranked = results.sort_values(["macro_f1", "balanced_acc"], ascending=False)
    return ranked.iloc[0]


def fit_config(row, feats, train, label_col):
    """Re-train one configuration (a row of the results table) on TRAIN. Returns (model, columns)."""
    cols = feature_set_columns(row["feature_set"], feats.columns)
    model = build_model(row["model"], row["params"])
    model.fit(feats.loc[train["isic_id"], cols].to_numpy(), train[label_col].to_numpy())
    return model, cols


# --------------------------------------------------------------------------------------------------
# extra evaluation tools
# --------------------------------------------------------------------------------------------------
def bootstrap_ci(y, pred, score=None, n_boot=1000, seed=config.SEED, alpha=0.05):
    """95% confidence intervals of the binary metrics by resampling the TEST lesions with replacement.

    Shows how much the test numbers could move by chance with ~1,000 test lesions.
    """
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    score = None if score is None else np.asarray(score)
    samples = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        samples.append(binary_metrics(y[i], pred[i], None if score is None else score[i]))
    samples = pd.DataFrame(samples)
    return pd.DataFrame({"low": samples.quantile(alpha / 2), "high": samples.quantile(1 - alpha / 2)})


def threshold_for_sensitivity(y, score, target=0.90):
    """Highest decision threshold that still catches >= target of the malignant lesions (on VALIDATION).

    The default rule 'score > 0.5' treats both errors the same; clinically a missed cancer is worse,
    so we also report an operating point with a guaranteed sensitivity (chosen on validation only).
    """
    y, score = np.asarray(y), np.asarray(score)
    malignant_scores = np.sort(score[y == 1])
    k = int(np.floor((1 - target) * len(malignant_scores)))   # number of malignant lesions we may miss
    return malignant_scores[k]


def per_group_recall(y, pred, groups):
    """Share of correct predictions inside each group (e.g. each of the 11 diagnoses)."""
    df = pd.DataFrame({"group": np.asarray(groups), "correct": np.asarray(y) == np.asarray(pred),
                       "true": np.asarray(y)})
    out = df.groupby("group").agg(n=("correct", "size"), correct=("correct", "mean"), label=("true", "first"))
    return out.sort_values("correct")


def confusion_table(y, pred, class_names):
    """Confusion matrix as a labelled table: rows = truth, columns = prediction (Session 5)."""
    cm = confusion_matrix(y, pred, labels=list(range(len(class_names))))
    return pd.DataFrame(cm, index=[f"true {c}" for c in class_names], columns=[f"pred {c}" for c in class_names])
