"""Model interpretation.

Permutation importance is computed on held-out folds under the grouped split.
Impurity-based importances from a fitted forest are not used as the headline:
they are biased toward high-cardinality continuous features and are measured on
training data, which is exactly the confound this project is about.

Nothing here supports a causal claim. These are associations between engineered
signal features and a human-tagged label, not evidence about what causes a
behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from smartnet import config
from smartnet.evaluation.splits import make_splitter
from smartnet.models.zoo import build_model


def permutation_importances(
    df: pd.DataFrame,
    label_col: str,
    features: list[str] | None = None,
    model_name: str = "random_forest",
    strategy: str = "grouped_event",
    n_repeats: int = 10,
    seed: int = config.RANDOM_SEED,
    max_folds: int | None = None,
) -> pd.DataFrame:
    """Held-out permutation importance, averaged across folds.

    ``max_folds`` limits how many CV folds contribute. Permutation importance
    is expensive (n_features x n_repeats refits per fold); two folds is enough
    for a stable ranking on a dataset this size.
    """
    features = features or config.ALL_FEATURES
    X = df[features].to_numpy()
    y = df[label_col].to_numpy()

    acc = np.zeros(len(features))
    n_folds = 0
    for tr, te in make_splitter(df, y, strategy, seed=seed):
        if max_folds is not None and n_folds >= max_folds:
            break
        model = build_model(model_name, seed)
        model.fit(X[tr], y[tr])
        r = permutation_importance(
            model, X[te], y[te],
            n_repeats=n_repeats, random_state=seed,
            scoring="balanced_accuracy", n_jobs=-1,
        )
        acc += r.importances_mean
        n_folds += 1

    out = pd.DataFrame(
        {"feature": features, "importance": acc / n_folds}
    ).sort_values("importance", ascending=False, ignore_index=True)
    out["feature_block"] = out["feature"].map(_block_of)
    return out


def _block_of(feature: str) -> str:
    if feature in config.CURRENT_EPOCH_FEATURES:
        return "current epoch"
    if feature in config.WINDOW_AGGREGATE_FEATURES:
        return "10-epoch aggregate"
    if feature.startswith("back_"):
        return "backward lag"
    if feature.startswith("forward_"):
        return "forward lead"
    return "other"


def block_importance(imp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate importance by feature block, for a readable summary."""
    return (
        imp.groupby("feature_block")["importance"]
        .agg(["sum", "mean", "count"])
        .sort_values("sum", ascending=False)
        .round(5)
        .reset_index()
    )


def logistic_coefficients(
    df: pd.DataFrame, label_col: str, features: list[str] | None = None
) -> pd.DataFrame:
    """Standardised coefficients from the interpretable linear model."""
    features = features or config.ALL_FEATURES
    model = build_model("logistic_regression")
    model.fit(df[features].to_numpy(), df[label_col].to_numpy())
    clf = model[-1]
    names = config.LABEL_MAPS[label_col]
    coef = pd.DataFrame(clf.coef_, columns=features)
    coef.index = [names[int(c)] for c in clf.classes_]
    return coef.T
