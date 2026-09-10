"""Tests for the MLOps lifecycle: registry, gates, drift, retraining policy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from smartnet import config
from smartnet.mlops.monitoring import (
    ReferenceProfile, RetrainingPolicy, detect_feature_drift,
    detect_prediction_drift, population_stability_index,
)
from smartnet.mlops.registry import (
    ModelRegistry, PromotionGate, Stage,
)


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(
        index_path=tmp_path / "index.json", audit_path=tmp_path / "audit.jsonl"
    )


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    d = {f: rng.normal(size=200) for f in config.ALL_FEATURES}
    d["motion_4cat"] = rng.integers(0, 4, size=200).astype(float)
    return pd.DataFrame(d)


def _register(registry, frame, name="m", **metric_overrides):
    metrics = {
        "accuracy": 0.95, "balanced_accuracy": 0.94,
        "worst_class_sensitivity": 0.90, **metric_overrides,
    }
    model = DummyClassifier(strategy="most_frequent").fit(
        frame[config.ALL_FEATURES], frame["motion_4cat"]
    )
    return registry.register(
        model, name, "dummy", "motion_4cat", config.ALL_FEATURES,
        metrics, frame, "grouped_event", "tester",
    )


# ------------------------------------------------------------- versioning
def test_versions_increment(registry, frame):
    assert _register(registry, frame).version == 1
    assert _register(registry, frame).version == 2


def test_registered_model_can_be_loaded(registry, frame):
    mv = _register(registry, frame)
    registry.promote("m", mv.version, Stage.STAGING, "tester", "t")
    registry.promote("m", mv.version, Stage.PRODUCTION, "tester", "t")
    model = registry.load("m")
    assert hasattr(model, "predict")


def test_data_hash_is_stable_and_sensitive(registry, frame):
    h1 = ModelRegistry.data_hash(frame, config.ALL_FEATURES, "motion_4cat")
    h2 = ModelRegistry.data_hash(frame.copy(), config.ALL_FEATURES, "motion_4cat")
    assert h1 == h2
    changed = frame.copy()
    changed.loc[0, config.ALL_FEATURES[0]] += 1.0
    assert ModelRegistry.data_hash(changed, config.ALL_FEATURES, "motion_4cat") != h1


def test_non_numeric_metrics_are_excluded(registry, frame):
    mv = _register(registry, frame, worst_class="Enter")
    assert "worst_class" not in mv.metrics
    assert "balanced_accuracy" in mv.metrics


# -------------------------------------------------------------- promotion
def test_illegal_transition_rejected(registry, frame):
    mv = _register(registry, frame)
    with pytest.raises(ValueError, match="Illegal transition"):
        registry.promote("m", mv.version, Stage.PRODUCTION, "tester", "skip staging")


def test_gate_blocks_low_worst_class_sensitivity(registry, frame):
    mv = _register(registry, frame, balanced_accuracy=0.78, worst_class_sensitivity=0.34)
    registry.promote("m", mv.version, Stage.STAGING, "tester", "t")
    with pytest.raises(PermissionError, match="worst_class_sensitivity"):
        registry.promote("m", mv.version, Stage.PRODUCTION, "tester", "t", gate=PromotionGate())


def test_gate_blocks_large_accuracy_balanced_gap(registry, frame):
    """The project's central finding, enforced as a control."""
    mv = _register(registry, frame, accuracy=0.947, balanced_accuracy=0.787,
                   worst_class_sensitivity=0.80)
    registry.promote("m", mv.version, Stage.STAGING, "tester", "t")
    with pytest.raises(PermissionError, match="masking minority-class failure"):
        registry.promote("m", mv.version, Stage.PRODUCTION, "tester", "t", gate=PromotionGate())


def test_gate_rejects_random_split_validation(registry, frame):
    model = DummyClassifier().fit(frame[config.ALL_FEATURES], frame["motion_4cat"])
    mv = registry.register(
        model, "m", "dummy", "motion_4cat", config.ALL_FEATURES,
        {"accuracy": 0.99, "balanced_accuracy": 0.98, "worst_class_sensitivity": 0.95},
        frame, "random_row", "tester",
    )
    registry.promote("m", mv.version, Stage.STAGING, "tester", "t")
    with pytest.raises(PermissionError, match="validation_strategy"):
        registry.promote("m", mv.version, Stage.PRODUCTION, "tester", "t", gate=PromotionGate())


def test_blocked_promotion_is_audited(registry, frame):
    mv = _register(registry, frame, worst_class_sensitivity=0.10)
    registry.promote("m", mv.version, Stage.STAGING, "tester", "t")
    with pytest.raises(PermissionError):
        registry.promote("m", mv.version, Stage.PRODUCTION, "tester", "t", gate=PromotionGate())
    assert "promotion_blocked" in set(registry.audit_trail()["action"])


def test_promotion_archives_previous_production(registry, frame):
    a = _register(registry, frame)
    b = _register(registry, frame)
    for mv in (a, b):
        registry.promote("m", mv.version, Stage.STAGING, "tester", "t")
        registry.promote("m", mv.version, Stage.PRODUCTION, "tester", "t")
    assert registry.get("m", a.version).stage == Stage.ARCHIVED.value
    assert registry.production("m").version == b.version


def test_only_one_production_version(registry, frame):
    for _ in range(3):
        mv = _register(registry, frame)
        registry.promote("m", mv.version, Stage.STAGING, "tester", "t")
        registry.promote("m", mv.version, Stage.PRODUCTION, "tester", "t")
    prod = [v for v in registry.history("m").itertuples() if v.stage == "production"]
    assert len(prod) == 1


# --------------------------------------------------------------- rollback
def test_rollback_restores_previous_version(registry, frame):
    a = _register(registry, frame)
    b = _register(registry, frame)
    for mv in (a, b):
        registry.promote("m", mv.version, Stage.STAGING, "tester", "t")
        registry.promote("m", mv.version, Stage.PRODUCTION, "tester", "t")
    restored = registry.rollback("m", "tester", "regression found")
    assert restored.version == a.version
    assert registry.production("m").version == a.version


def test_rollback_without_history_raises(registry, frame):
    mv = _register(registry, frame)
    registry.promote("m", mv.version, Stage.STAGING, "tester", "t")
    registry.promote("m", mv.version, Stage.PRODUCTION, "tester", "t")
    with pytest.raises(LookupError):
        registry.rollback("m", "tester", "nothing to go back to")


def test_audit_trail_records_actor_and_reason(registry, frame):
    _register(registry, frame)
    trail = registry.audit_trail()
    assert {"timestamp", "action", "subject", "actor", "reason"} <= set(trail.columns)
    assert (trail["actor"] == "tester").all()


# ------------------------------------------------------------------ drift
def test_psi_zero_for_identical_samples():
    x = np.random.default_rng(0).normal(size=1000)
    assert population_stability_index(x, x) < 0.01


def test_psi_detects_large_shift():
    rng = np.random.default_rng(0)
    assert population_stability_index(rng.normal(size=1000), rng.normal(4, size=1000)) > 0.25


def test_psi_ordered_by_shift_size():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=2000)
    small = population_stability_index(ref, rng.normal(0.15, size=2000))
    large = population_stability_index(ref, rng.normal(2.0, size=2000))
    assert small < large


def test_no_drift_flagged_on_resample(frame):
    d = detect_feature_drift(frame, frame.sample(120, random_state=1), config.ALL_FEATURES)
    assert d.n_alert == 0


def test_drift_flagged_on_shifted_batch(frame):
    shifted = frame.copy()
    for f in config.ALL_FEATURES[:10]:
        shifted[f] = shifted[f] * 3 + 5
    d = detect_feature_drift(frame, shifted, config.ALL_FEATURES)
    assert d.n_alert >= 5
    assert d.worst_psi > 0.25


def test_prediction_drift_detects_distribution_change():
    ref = np.array([0] * 80 + [1] * 20)
    cur = np.array([0] * 20 + [1] * 80)
    assert detect_prediction_drift(ref, cur)["prediction_psi"] > 0.25


def test_reference_profile_roundtrip(frame, tmp_path):
    ref = ReferenceProfile.build(frame, config.ALL_FEATURES, "motion_4cat", "m:v1")
    p = ref.save(tmp_path / "ref.json")
    assert ReferenceProfile.load(p).n_rows == len(frame)


# ------------------------------------------------------- retraining policy
def test_policy_no_action_when_stable(frame):
    d = detect_feature_drift(frame, frame.sample(150, random_state=3), config.ALL_FEATURES)
    d.prediction_drift = {"prediction_psi": 0.01}
    assert RetrainingPolicy().evaluate(d, 0.96, 0.96)["decision"] == "no_action"


def test_policy_investigates_drift_without_confirmed_loss(frame):
    shifted = frame.copy()
    for f in config.ALL_FEATURES[:20]:
        shifted[f] = shifted[f] * 4 + 8
    d = detect_feature_drift(frame, shifted, config.ALL_FEATURES)
    d.prediction_drift = {"prediction_psi": 0.02}
    r = RetrainingPolicy().evaluate(d)
    assert r["decision"] == "investigate" and r["retrain"] is False


def test_policy_triggers_retrain_on_confirmed_performance_loss(frame):
    shifted = frame.copy()
    for f in config.ALL_FEATURES[:20]:
        shifted[f] = shifted[f] * 4 + 8
    d = detect_feature_drift(frame, shifted, config.ALL_FEATURES)
    d.prediction_drift = {"prediction_psi": 0.5}
    r = RetrainingPolicy().evaluate(d, 0.96, 0.80)
    assert r["decision"] == "retrain" and r["retrain"] is True


def test_policy_refuses_to_judge_tiny_batch(frame):
    d = detect_feature_drift(frame, frame.sample(20, random_state=4), config.ALL_FEATURES)
    d.prediction_drift = {"prediction_psi": 0.9}
    assert RetrainingPolicy().evaluate(d, 0.96, 0.5)["decision"] == "insufficient_data"
