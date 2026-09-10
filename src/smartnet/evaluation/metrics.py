"""Evaluation metrics.

Accuracy alone is not reportable here: the classes are imbalanced and the
operationally interesting behaviours (entering and exiting a net) are the
rarest. Per-class sensitivity and specificity are reported alongside macro
averages so that a model which ignores a small class cannot hide behind a high
overall score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def per_class_sensitivity_specificity(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list
) -> pd.DataFrame:
    """One-vs-rest sensitivity and specificity for every class."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    rows = []
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        rows.append(
            {
                "class": lab,
                "support": int(cm[i, :].sum()),
                "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
                "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
                "precision": tp / (tp + fp) if (tp + fp) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["f1"] = (
        2 * out["precision"] * out["sensitivity"] / (out["precision"] + out["sensitivity"])
    )
    return out


def macro_auc(y_true: np.ndarray, y_proba: np.ndarray, labels: list) -> float:
    """One-vs-rest macro ROC-AUC, tolerant of folds missing a class."""
    try:
        return float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels)
        )
    except ValueError:
        return float("nan")


def summarise(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None, labels: list
) -> dict[str, float]:
    """Headline metrics for one fold."""
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_proba is not None:
        out["macro_auc"] = macro_auc(y_true, y_proba, labels)
    return out


def aggregate_folds(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Mean and standard deviation across folds, flattened into one dict."""
    frame = pd.DataFrame(fold_metrics)
    out: dict[str, float] = {}
    for col in frame.columns:
        out[f"{col}_mean"] = float(frame[col].mean())
        out[f"{col}_std"] = float(frame[col].std(ddof=1)) if len(frame) > 1 else 0.0
    return out


def confusion_frame(y_true: np.ndarray, y_pred: np.ndarray, labels: list) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame (rows = truth)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
