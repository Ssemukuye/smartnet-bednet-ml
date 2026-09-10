"""Unit tests for the SMARTNET pipeline.

Tests run against a synthetic fixture with the same schema as the study
extract, so the suite passes in CI without the real data present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smartnet import config
from smartnet.data import loader, validation
from smartnet.evaluation import metrics as M
from smartnet.evaluation.splits import make_splitter, split_diagnostics
from smartnet.models.zoo import build_model


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def synthetic() -> pd.DataFrame:
    """Schema-identical synthetic frame: 4 events/day over 6 days."""
    rng = np.random.default_rng(0)
    rows = []
    start = pd.Timestamp("2021-09-10 09:00:00")
    for day in range(6):
        for ev in range(4):
            label = ev % 4  # 0..3 across the 4cat scheme
            t0 = start + pd.Timedelta(days=day, minutes=17 * ev)
            for sec in range(rng.integers(3, 9)):
                row = {c: float(rng.normal(1 + label, 0.4)) for c in config.ALL_FEATURES}
                row[config.TIMESTAMP_COL] = str(t0 + pd.Timedelta(seconds=sec))
                row["motion_4cat"] = float(label)
                row["motion_5cat"] = float(label if label < 3 else 3 + (ev % 2))
                row["motion_3cat"] = float(0 if label == 0 else (2 if label == 2 else 1))
                row["motion_something"] = float(0 if label == 0 else 1)
                rows.append(row)
    return loader.add_time_structure(pd.DataFrame(rows))


# ---------------------------------------------------------------- structure
def test_time_structure_adds_expected_columns(synthetic):
    for col in ("timestamp", "event_id", "session_date", "gap_seconds", "hour"):
        assert col in synthetic.columns


def test_events_are_contiguous_and_pure(synthetic):
    loader.assert_events_are_label_pure(synthetic)
    assert synthetic["event_id"].nunique() == 24


def test_event_summary_durations_positive(synthetic):
    ev = loader.event_summary(synthetic, "motion_4cat")
    assert (ev["duration_s"] > 0).all()
    assert ev["n_distinct_labels"].max() == 1


def test_impure_events_are_detected():
    """The purity assertion must actually fire when violated."""
    rng = np.random.default_rng(1)
    rows = []
    t0 = pd.Timestamp("2021-09-10 09:00:00")
    for sec in range(4):
        row = {c: float(rng.normal()) for c in config.ALL_FEATURES}
        row[config.TIMESTAMP_COL] = str(t0 + pd.Timedelta(seconds=sec))
        for c in config.LABEL_COLS:
            row[c] = float(sec % 2)  # label flips mid-run
        rows.append(row)
    df = loader.add_time_structure(pd.DataFrame(rows))
    with pytest.raises(AssertionError):
        loader.assert_events_are_label_pure(df)


# ---------------------------------------------------------------- validation
def test_validation_report_shape(synthetic):
    rep = validation.run_all(synthetic)
    assert set(rep.columns) == {"name", "severity", "n_affected", "message"}
    assert not validation.has_blocking_failure(rep)


def test_missing_column_is_a_blocking_failure(synthetic):
    broken = synthetic.drop(columns=["stdvz"])
    rep = validation.run_all(broken)
    assert validation.has_blocking_failure(rep)


def test_label_inconsistency_detected(synthetic):
    broken = synthetic.copy()
    broken.loc[broken.index[0], "motion_something"] = 1 - broken.loc[broken.index[0], "motion_something"]
    result = validation.check_label_consistency(broken)
    assert result.severity == "fail"


# ---------------------------------------------------------------- splitting
def test_grouped_split_shares_no_events(synthetic):
    y = synthetic["motion_4cat"].to_numpy()
    for tr, te in make_splitter(synthetic, y, "grouped_event", n_splits=3):
        assert split_diagnostics(synthetic, tr, te)["shared_events"] == 0


def test_grouped_day_split_shares_no_days(synthetic):
    y = synthetic["motion_4cat"].to_numpy()
    for tr, te in make_splitter(synthetic, y, "grouped_day", n_splits=3):
        assert split_diagnostics(synthetic, tr, te)["shared_days"] == 0


def test_random_split_does_share_events(synthetic):
    """Guards the project's core claim: random splitting reuses events."""
    y = synthetic["motion_4cat"].to_numpy()
    shared = [
        split_diagnostics(synthetic, tr, te)["shared_events"]
        for tr, te in make_splitter(synthetic, y, "random_row", n_splits=3)
    ]
    assert max(shared) > 0


def test_unknown_strategy_raises(synthetic):
    y = synthetic["motion_4cat"].to_numpy()
    with pytest.raises(KeyError):
        list(make_splitter(synthetic, y, "not_a_strategy"))


# ---------------------------------------------------------------- metrics
def test_perfect_prediction_gives_unit_metrics():
    y = np.array(["a", "b", "a", "c"])
    pc = M.per_class_sensitivity_specificity(y, y, ["a", "b", "c"])
    assert np.allclose(pc["sensitivity"], 1.0)
    assert np.allclose(pc["specificity"], 1.0)


def test_sensitivity_matches_hand_computation():
    y_true = np.array(["pos", "pos", "pos", "neg"])
    y_pred = np.array(["pos", "neg", "pos", "neg"])
    pc = M.per_class_sensitivity_specificity(y_true, y_pred, ["pos", "neg"])
    assert pc.loc[pc["class"] == "pos", "sensitivity"].iloc[0] == pytest.approx(2 / 3)


def test_balanced_accuracy_penalises_majority_only_prediction():
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred = np.zeros(100, dtype=int)
    s = M.summarise(y_true, y_pred, None, [0, 1])
    assert s["accuracy"] == pytest.approx(0.9)
    assert s["balanced_accuracy"] == pytest.approx(0.5)


def test_aggregate_folds_reports_mean_and_std():
    agg = M.aggregate_folds([{"accuracy": 0.8}, {"accuracy": 0.9}])
    assert agg["accuracy_mean"] == pytest.approx(0.85)
    assert agg["accuracy_std"] > 0


# ---------------------------------------------------------------- models
@pytest.mark.parametrize("name", ["majority_baseline", "logistic_regression",
                                  "decision_tree", "random_forest", "gradient_boosting"])
def test_models_fit_and_predict(synthetic, name):
    X = synthetic[config.ALL_FEATURES].to_numpy()
    y = synthetic["motion_4cat"].to_numpy()
    model = build_model(name)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert set(np.unique(pred)) <= set(np.unique(y))


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        build_model("transformer_9000")


def test_feature_blocks_are_subsets_of_all():
    for name, block in config.FEATURE_BLOCKS.items():
        assert set(block) <= set(config.ALL_FEATURES), name


def test_no_label_leaks_into_features():
    """A label column must never appear in the feature list."""
    assert not (set(config.ALL_FEATURES) & set(config.LABEL_COLS))
