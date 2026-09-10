"""Cross-validated experiment runner.

Runs every (model x split strategy) combination for a given label scheme and
returns tidy results. Out-of-fold predictions are retained so that confusion
matrices and error analysis use every observation exactly once.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from smartnet import config
from smartnet.evaluation import metrics as M
from smartnet.evaluation.splits import make_splitter
from smartnet.models.zoo import build_model

logger = logging.getLogger(__name__)


def run_cv(
    df: pd.DataFrame,
    label_col: str,
    model_name: str,
    strategy: str,
    features: list[str] | None = None,
    n_splits: int = config.N_SPLITS,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """Cross-validate one model under one split strategy.

    Returns a dict with aggregated metrics, per-fold metrics and out-of-fold
    predictions.
    """
    features = features or config.ALL_FEATURES
    X = df[features].to_numpy()
    y = df[label_col].to_numpy()
    labels = sorted(np.unique(y).tolist())

    oof_pred = np.full(len(y), np.nan)
    oof_proba = np.full((len(y), len(labels)), np.nan)
    fold_metrics: list[dict] = []

    t0 = time.time()
    for fold, (tr, te) in enumerate(make_splitter(df, y, strategy, n_splits, seed)):
        model = build_model(model_name, seed)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        oof_pred[te] = pred

        proba = None
        if hasattr(model[-1], "predict_proba"):
            p = model.predict_proba(X[te])
            # Align fold classes onto the global label ordering.
            cls = list(model[-1].classes_)
            aligned = np.zeros((len(te), len(labels)))
            for j, c in enumerate(cls):
                aligned[:, labels.index(c)] = p[:, j]
            oof_proba[te] = aligned
            proba = aligned

        fold_metrics.append(M.summarise(y[te], pred, proba, labels))

    elapsed = time.time() - t0
    agg = M.aggregate_folds(fold_metrics)

    valid = ~np.isnan(oof_pred)
    oof_summary = M.summarise(
        y[valid],
        oof_pred[valid],
        oof_proba[valid] if not np.isnan(oof_proba).all() else None,
        labels,
    )

    return {
        "label_col": label_col,
        "model": model_name,
        "strategy": strategy,
        "n_features": len(features),
        "n_obs": int(valid.sum()),
        "fit_seconds": round(elapsed, 2),
        **{f"cv_{k}": v for k, v in agg.items()},
        **{f"oof_{k}": v for k, v in oof_summary.items()},
        "_oof_pred": oof_pred,
        "_oof_proba": oof_proba,
        "_labels": labels,
        "_y": y,
    }


def run_sweep(
    df: pd.DataFrame,
    label_col: str = config.PRIMARY_LABEL,
    models: list[str] | None = None,
    strategies: list[str] | None = None,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run all combinations; return a results table and the raw artefacts."""
    models = models or ExperimentDefaults.models
    strategies = strategies or [s.name for s in config.SPLIT_STRATEGIES]

    rows, artefacts = [], {}
    for strategy in strategies:
        for model_name in models:
            logger.info("Running %s | %s | %s", label_col, model_name, strategy)
            res = run_cv(df, label_col, model_name, strategy, features)
            key = f"{label_col}|{model_name}|{strategy}"
            artefacts[key] = {
                "oof_pred": res.pop("_oof_pred"),
                "oof_proba": res.pop("_oof_proba"),
                "labels": res.pop("_labels"),
                "y": res.pop("_y"),
            }
            rows.append(res)
    return pd.DataFrame(rows), artefacts


class ExperimentDefaults:
    models = [
        "majority_baseline",
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
    ]


def leakage_table(results: pd.DataFrame, metric: str = "cv_accuracy_mean") -> pd.DataFrame:
    """Pivot results into models x split strategies, with the inflation gap."""
    piv = results.pivot_table(index="model", columns="strategy", values=metric)
    order = [s.name for s in config.SPLIT_STRATEGIES if s.name in piv.columns]
    piv = piv[order]
    if {"random_row", "grouped_event"} <= set(piv.columns):
        piv["inflation_random_vs_event"] = piv["random_row"] - piv["grouped_event"]
    return piv.round(4)
