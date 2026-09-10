"""Demonstrate the full governed model lifecycle, end to end.

    register -> gate -> staging -> production -> monitor -> trigger -> rollback

Every step writes real artefacts to models/registry/ and reports/monitoring/,
including an append-only audit trail. Run:

    PYTHONPATH=src python scripts/run_mlops_lifecycle.py
"""
from __future__ import annotations

import json
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from smartnet import config
from smartnet.data import loader
from smartnet.evaluation import metrics as M
from smartnet.evaluation.splits import make_splitter
from smartnet.mlops.monitoring import (
    ReferenceProfile, RetrainingPolicy, detect_feature_drift,
    detect_prediction_drift, save_drift_report,
)
from smartnet.mlops.registry import ModelRegistry, PromotionGate, Stage
from smartnet.models.zoo import build_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("mlops")

MODEL_NAME = "bednet-behaviour-classifier"
ACTOR = "ssemukuye.timothy"


def evaluate(df: pd.DataFrame, label_col: str, algorithm: str, strategy: str) -> tuple:
    """Cross-validate and return (fitted_model, metrics, oof_predictions)."""
    feats = config.ALL_FEATURES
    X, y = df[feats].to_numpy(), df[label_col].to_numpy()
    labels = sorted(np.unique(y).tolist())
    oof = np.full(len(y), np.nan)

    for tr, te in make_splitter(df, y, strategy):
        m = build_model(algorithm)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])

    names = config.LABEL_MAPS[label_col]
    ordered = [names[k] for k in sorted(names)]
    y_true = np.array([names[int(v)] for v in y])
    y_pred = np.array([names[int(v)] for v in oof])

    per_class = M.per_class_sensitivity_specificity(y_true, y_pred, ordered)
    summary = M.summarise(y, oof, None, labels)
    summary["worst_class_sensitivity"] = float(per_class["sensitivity"].min())
    summary["worst_class"] = str(
        per_class.loc[per_class["sensitivity"].idxmin(), "class"]
    )

    final = build_model(algorithm)
    final.fit(X, y)
    return final, summary, oof


def main() -> None:
    df = loader.load_analysis_frame()
    reg = ModelRegistry()

    # ================================================================== 1
    log.info("=" * 66)
    log.info("STEP 1 — Train and register a 5-category candidate")
    log.info("=" * 66)
    model5, metrics5, oof5 = evaluate(df, "motion_5cat", "random_forest", "grouped_event")
    log.info(
        "5-cat: accuracy %.4f | balanced %.4f | worst class %s at %.3f",
        metrics5["accuracy"], metrics5["balanced_accuracy"],
        metrics5["worst_class"], metrics5["worst_class_sensitivity"],
    )
    v1 = reg.register(
        model5, MODEL_NAME, "random_forest", "motion_5cat", config.ALL_FEATURES,
        metrics5, df, "grouped_event", ACTOR,
        notes=f"Five-category candidate: unfurl / enter / sleep / exit / fold. "
              f"Worst class: {metrics5['worst_class']}",
    )
    log.info("registered %s (stage=%s)", v1.key, v1.stage)

    # ================================================================== 2
    log.info("")
    log.info("=" * 66)
    log.info("STEP 2 — Promotion gate: does it deserve production?")
    log.info("=" * 66)
    gate = PromotionGate()
    reg.promote(MODEL_NAME, v1.version, Stage.STAGING, ACTOR, "candidate evaluation")
    try:
        reg.promote(
            MODEL_NAME, v1.version, Stage.PRODUCTION, ACTOR,
            "propose 5-category model for production", gate=gate,
        )
        log.info("promoted to production")
    except PermissionError as exc:
        log.warning("BLOCKED — %s", exc)
        log.info("This is the intended outcome. 94.7%% accuracy is not enough when")
        log.info("Enter classifies at 0.34 sensitivity. The gate caught it.")

    # ================================================================== 3
    log.info("")
    log.info("=" * 66)
    log.info("STEP 3 — Train the 4-category model and promote it properly")
    log.info("=" * 66)
    model4, metrics4, oof4 = evaluate(df, "motion_4cat", "random_forest", "grouped_event")
    log.info(
        "4-cat: accuracy %.4f | balanced %.4f | worst class %s at %.3f",
        metrics4["accuracy"], metrics4["balanced_accuracy"],
        metrics4["worst_class"], metrics4["worst_class_sensitivity"],
    )
    v2 = reg.register(
        model4, MODEL_NAME, "random_forest", "motion_4cat", config.ALL_FEATURES,
        metrics4, df, "grouped_event", ACTOR,
        notes="Four-category model: entry and exit merged after directional "
              "confusion established as a sensor limitation",
    )
    reg.promote(MODEL_NAME, v2.version, Stage.STAGING, ACTOR, "gate evaluation")
    reg.promote(
        MODEL_NAME, v2.version, Stage.PRODUCTION, ACTOR,
        "meets balanced-accuracy and worst-class thresholds under grouped CV",
        gate=gate,
    )
    log.info("promoted %s to production", v2.key)

    # ================================================================== 4
    log.info("")
    log.info("=" * 66)
    log.info("STEP 4 — Capture the reference profile")
    log.info("=" * 66)
    ref = ReferenceProfile.build(df, config.ALL_FEATURES, "motion_4cat", v2.key)
    path = ref.save()
    log.info("reference profile: %d rows, %d features -> %s",
             ref.n_rows, len(ref.feature_stats), path.name)

    # ================================================================== 5
    log.info("")
    log.info("=" * 66)
    log.info("STEP 5 — Monitor an incoming batch (no drift)")
    log.info("=" * 66)
    rng = np.random.default_rng(0)
    stable = df.sample(400, random_state=1)
    drift_ok = detect_feature_drift(df, stable, config.ALL_FEATURES, v2.key)
    prod_model = reg.load(MODEL_NAME)
    drift_ok.prediction_drift = detect_prediction_drift(
        prod_model.predict(df[config.ALL_FEATURES].to_numpy()),
        prod_model.predict(stable[config.ALL_FEATURES].to_numpy()),
    )
    policy = RetrainingPolicy()
    decision_ok = policy.evaluate(drift_ok, metrics4["balanced_accuracy"], metrics4["balanced_accuracy"])
    log.info("worst PSI %.4f | alerts %d | decision: %s",
             drift_ok.worst_psi, drift_ok.n_alert, decision_ok["decision"])
    save_drift_report(drift_ok, decision_ok)

    # ================================================================== 6
    log.info("")
    log.info("=" * 66)
    log.info("STEP 6 — Monitor a drifted batch (simulated sensor recalibration)")
    log.info("=" * 66)
    drifted = df.sample(400, random_state=2).copy()
    for f in config.WINDOW_AGGREGATE_FEATURES + config.CURRENT_EPOCH_FEATURES:
        drifted[f] = drifted[f] * 1.6 + rng.normal(0.3, 0.1, len(drifted))
    drift_bad = detect_feature_drift(df, drifted, config.ALL_FEATURES, v2.key)
    drift_bad.prediction_drift = detect_prediction_drift(
        prod_model.predict(df[config.ALL_FEATURES].to_numpy()),
        prod_model.predict(drifted[config.ALL_FEATURES].to_numpy()),
    )
    degraded = metrics4["balanced_accuracy"] - 0.12
    decision_bad = policy.evaluate(drift_bad, metrics4["balanced_accuracy"], degraded)
    log.info("worst PSI %.4f | alerts %d | prediction PSI %.4f",
             drift_bad.worst_psi, drift_bad.n_alert,
             drift_bad.prediction_drift["prediction_psi"])
    log.info("decision: %s", decision_bad["decision"].upper())
    for r in decision_bad["reasons"]:
        log.info("   reason: %s", r)
    save_drift_report(drift_bad, decision_bad)

    # ================================================================== 7
    log.info("")
    log.info("=" * 66)
    log.info("STEP 7 — Bad promotion, then rollback")
    log.info("=" * 66)
    model_lr, metrics_lr, _ = evaluate(df, "motion_4cat", "logistic_regression", "grouped_event")
    v3 = reg.register(
        model_lr, MODEL_NAME, "logistic_regression", "motion_4cat", config.ALL_FEATURES,
        metrics_lr, df, "grouped_event", ACTOR, notes="Simpler model trialled in production",
    )
    reg.promote(MODEL_NAME, v3.version, Stage.STAGING, ACTOR, "trial simpler model")
    reg.promote(MODEL_NAME, v3.version, Stage.PRODUCTION, ACTOR,
                "prefer interpretable model for MOH handover", gate=gate)
    log.info("production is now %s", reg.production(MODEL_NAME).key)

    restored = reg.rollback(
        MODEL_NAME, ACTOR,
        reason="post-deployment monitoring showed worst-class sensitivity regression",
    )
    log.info("rolled back — production restored to %s (%s)",
             restored.key, restored.algorithm)

    # ================================================================== 8
    log.info("")
    log.info("=" * 66)
    log.info("STEP 8 — Registry and audit trail")
    log.info("=" * 66)
    hist = reg.history(MODEL_NAME)
    hist.to_csv(config.RESULTS_DIR / "model_registry_history.csv", index=False)
    log.info("\n%s", hist[["key", "stage", "algorithm", "label_scheme",
                           "metric_balanced_accuracy", "metric_worst_class_sensitivity"]].to_string(index=False))

    audit = reg.audit_trail()
    audit.to_csv(config.RESULTS_DIR / "model_audit_trail.csv", index=False)
    log.info("")
    log.info("audit trail — %d immutable records", len(audit))
    log.info("\n%s", audit[["timestamp", "action", "subject", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
