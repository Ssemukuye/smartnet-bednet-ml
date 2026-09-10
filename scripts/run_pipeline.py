"""End-to-end pipeline: validate, evaluate, interpret, persist.

Usage
-----
    PYTHONPATH=src python scripts/run_pipeline.py
    PYTHONPATH=src python scripts/run_pipeline.py --label motion_5cat
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from smartnet import config
from smartnet.data import loader, validation
from smartnet.evaluation import metrics as M
from smartnet.evaluation.splits import summarise_strategies
from smartnet.models.experiment import leakage_table, run_sweep
from smartnet.models.interpret import permutation_importances

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pipeline")


def stage_quality(df: pd.DataFrame, label_col: str) -> None:
    """Stage 1: data-quality report and split leakage diagnostics."""
    loader.assert_events_are_label_pure(df)
    report = validation.run_all(df)
    report.to_csv(config.RESULTS_DIR / "data_quality_report.csv", index=False)
    if validation.has_blocking_failure(report):
        raise SystemExit("Blocking data-quality failure; see reports/model_results/")
    logger.info("Data quality: %d checks, all non-blocking", len(report))

    diag = summarise_strategies(df, label_col)
    diag.to_csv(config.RESULTS_DIR / "split_diagnostics.csv", index=False)
    logger.info(
        "Random split reuses %.1f%% of test epochs from training events",
        diag.loc[diag.strategy == "random_row", "pct_test_from_seen_event"].iloc[0],
    )


def stage_sweep(df: pd.DataFrame, label_col: str, feature_block: str) -> None:
    """Stage 2: model x split-strategy sweep, confusion matrix, per-class detail."""
    features = config.FEATURE_BLOCKS[feature_block]
    results, artefacts = run_sweep(df, label_col=label_col, features=features)
    results.to_csv(config.RESULTS_DIR / f"results_{label_col}_{feature_block}.csv", index=False)

    leak = leakage_table(results)
    leak.to_csv(config.RESULTS_DIR / f"leakage_table_{label_col}.csv")
    logger.info("Leakage table:\n%s", leak.to_string())

    names = config.LABEL_MAPS[label_col]
    ordered = [names[k] for k in sorted(names)]
    for strategy in ("random_row", "grouped_event", "grouped_day"):
        art = artefacts[f"{label_col}|random_forest|{strategy}"]
        valid = ~np.isnan(art["oof_pred"])
        y_true = np.array([names[int(v)] for v in art["y"][valid]])
        y_pred = np.array([names[int(v)] for v in art["oof_pred"][valid]])

        M.per_class_sensitivity_specificity(y_true, y_pred, ordered).to_csv(
            config.RESULTS_DIR / f"per_class_{label_col}_{strategy}.csv", index=False
        )
        cm = M.confusion_frame(y_true, y_pred, ordered)
        cm.to_csv(config.RESULTS_DIR / f"confusion_{label_col}_{strategy}.csv")
        np.save(config.RESULTS_DIR / f"oof_{label_col}_{strategy}.npy", art["oof_pred"])
        if strategy == "grouped_event":
            logger.info("Confusion (grouped_event, RF):\n%s", cm.to_string())


def stage_interpret(df: pd.DataFrame, label_col: str, feature_block: str) -> None:
    """Stage 3: held-out permutation importance."""
    features = config.FEATURE_BLOCKS[feature_block]
    imp = permutation_importances(df, label_col, features, n_repeats=5, max_folds=2)
    imp.to_csv(config.RESULTS_DIR / f"permutation_importance_{label_col}.csv", index=False)
    logger.info("Top features:\n%s", imp.head(12).to_string(index=False))


def stage_manifest(df: pd.DataFrame, label_col: str, feature_block: str) -> None:
    features = config.FEATURE_BLOCKS[feature_block]
    names = config.LABEL_MAPS[label_col]
    manifest = {
        "label_col": label_col,
        "feature_block": feature_block,
        "n_features": len(features),
        "n_observations": int(len(df)),
        "n_events": int(df["event_id"].nunique()),
        "n_recording_days": int(df["session_date"].nunique()),
        "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "class_counts": {
            names[int(k)]: int(v) for k, v in df[label_col].value_counts().items()
        },
        "seed": config.RANDOM_SEED,
        "n_splits": config.N_SPLITS,
    }
    (config.RESULTS_DIR / f"manifest_{label_col}.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote artefacts to %s", config.RESULTS_DIR)


STAGES = {
    "quality": lambda df, l, f: stage_quality(df, l),
    "sweep": stage_sweep,
    "interpret": stage_interpret,
    "manifest": stage_manifest,
}


def main(label_col: str, feature_block: str, stages: list[str]) -> None:
    df = loader.load_analysis_frame()
    for s in stages:
        logger.info("=== stage: %s ===", s)
        STAGES[s](df, label_col, feature_block)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=config.PRIMARY_LABEL, choices=config.LABEL_COLS)
    ap.add_argument("--features", default="all", choices=list(config.FEATURE_BLOCKS))
    ap.add_argument(
        "--stage", default="all",
        choices=list(STAGES) + ["all"],
        help="Run a single stage; stages persist artefacts so they can be resumed.",
    )
    a = ap.parse_args()
    chosen = list(STAGES) if a.stage == "all" else [a.stage]
    main(a.label, a.features, chosen)
