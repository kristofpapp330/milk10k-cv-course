"""
Milestones 4-5 -- Evaluation tools: macro-F1, the leaderboard metric, per-class thresholds, fairness.

Two ways of turning probabilities into decisions:
  * argmax     : the most probable class (what we used in the whole course; one decision per lesion)
  * leaderboard: the official MILK10k rule (Session 10): EVERY class whose probability is >= 0.5 is a
                 "yes", then F1 per class over all lesions, then the plain mean of the 11 F1s.
With softmax, the 11 probabilities sum to 1, so a rare class almost never reaches 0.5 -> its F1 is 0.
That is why we also tune one threshold per class on VALIDATION and rescale the probabilities so that
each threshold sits at 0.5 (the brief's "optional, advanced" step).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score


def macro_f1(y, pred, n_classes):
    return f1_score(y, pred, average="macro", labels=list(range(n_classes)), zero_division=0)


def per_class_table(y, pred, class_names):
    """Support, recall and F1 of every class (argmax decisions)."""
    labels = list(range(len(class_names)))
    return pd.DataFrame({
        "n": np.bincount(np.asarray(y), minlength=len(class_names)),
        "recall": recall_score(y, pred, labels=labels, average=None, zero_division=0),
        "F1": f1_score(y, pred, labels=labels, average=None, zero_division=0),
    }, index=class_names)


def confusion_df(y, pred, class_names):
    cm = confusion_matrix(y, pred, labels=list(range(len(class_names))))
    return pd.DataFrame(cm, index=[f"true {c}" for c in class_names], columns=[f"pred {c}" for c in class_names])


# --------------------------------------------------------------------------------------------------
# the official leaderboard metric
# --------------------------------------------------------------------------------------------------
def leaderboard_f1(y, proba, threshold=0.5):
    """Official rule: binarise every class at `threshold` (one value, or one per class), F1 per class
    (one-vs-rest over all lesions), then the plain mean. Returns (macro-F1, per-class F1 array)."""
    y, proba = np.asarray(y), np.asarray(proba)
    n_classes = proba.shape[1]
    t = np.broadcast_to(np.asarray(threshold, dtype=float), (n_classes,))
    per_class = np.array([f1_score(y == c, proba[:, c] >= t[c], zero_division=0) for c in range(n_classes)])
    return per_class.mean(), per_class


def tune_thresholds(y, proba, grid=np.round(np.arange(0.02, 0.99, 0.01), 2)):
    """For each class separately: the threshold (on VALIDATION) that gives the highest F1 for that class."""
    y, proba = np.asarray(y), np.asarray(proba)
    best = np.full(proba.shape[1], 0.5)
    for c in range(proba.shape[1]):
        if (y == c).sum() == 0:
            continue
        scores = [f1_score(y == c, proba[:, c] >= t, zero_division=0) for t in grid]
        best[c] = grid[int(np.argmax(scores))]
    return best


def rescale_to_half(proba, thresholds):
    """Monotone piece-wise linear map per class: [0, t] -> [0, 0.5] and [t, 1] -> [0.5, 1].
    After it, 'p >= 0.5' means exactly 'original p >= t', so the official 0.5 rule applies our thresholds.
    The ranking of the lesions inside each class does not change (values stay in [0, 1])."""
    p, t = np.asarray(proba, dtype=float), np.asarray(thresholds, dtype=float)
    low = 0.5 * p / t
    high = 0.5 + 0.5 * (p - t) / (1 - t)
    return np.clip(np.where(p < t, low, high), 0.0, 1.0)


# --------------------------------------------------------------------------------------------------
# uncertainty and fairness
# --------------------------------------------------------------------------------------------------
def bootstrap_macro_f1(y, pred, n_classes=None, n_boot=1000, seed=42):
    """95 % interval of macro-F1 by resampling the lesions with replacement.
    n_classes=None: average only over the classes that occur (used inside small groups)."""
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals.append(macro_f1(y[i], pred[i], n_classes) if n_classes else
                    f1_score(y[i], pred[i], average="macro", zero_division=0))
    return np.percentile(vals, [2.5, 97.5])


def group_report(y, pred, groups, min_n=50, n_boot=500):
    """Macro-F1 (with a 95 % bootstrap interval) and accuracy inside every group (e.g. each skin tone).
    Inside a group, macro-F1 averages over the classes that occur there (a group without any
    Indeterminate lesion is not punished for it). Groups smaller than min_n are flagged as too noisy."""
    df = pd.DataFrame({"y": np.asarray(y), "pred": np.asarray(pred), "group": np.asarray(groups)})
    rows = []
    for g, d in df.groupby("group", dropna=False):
        low, high = bootstrap_macro_f1(d["y"], d["pred"], None, n_boot=n_boot) if len(d) >= 10 else (np.nan, np.nan)
        rows.append({"group": "missing" if pd.isna(g) else g, "n lesions": len(d),
                     "classes present": int(d["y"].nunique()),
                     "macro-F1": f1_score(d["y"], d["pred"], average="macro", zero_division=0),
                     "95% CI": f"[{low:.2f}, {high:.2f}]" if not np.isnan(low) else "—",
                     "accuracy": (d["y"] == d["pred"]).mean(),
                     "note": "small group: do not conclude" if len(d) < min_n else ""})
    return pd.DataFrame(rows)
