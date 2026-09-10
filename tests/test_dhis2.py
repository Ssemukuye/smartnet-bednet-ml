"""Tests for DHIS2 data-entry anomaly detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smartnet.dhis2.anomaly import (
    AnomalyPipeline, ConsistencyRule, DHIS2_COLUMNS, detect_consistency_violations,
    detect_digit_preference, detect_impossible_values, detect_missing_reports,
    detect_statistical_outliers, evaluate_against_truth,
)
from smartnet.dhis2.simulate import DEFAULT_RULES, generate


@pytest.fixture(scope="module")
def simulated():
    return generate(n_facilities=12, n_periods=18, error_rate=0.03, seed=7)


def _values(org, de, vals, start=202401):
    return pd.DataFrame([
        {"dataElement": de, "period": str(start + i), "orgUnit": org,
         "categoryOptionCombo": "default", "value": float(v)}
        for i, v in enumerate(vals)
    ])


# ---------------------------------------------------------------- schema
def test_simulated_data_is_dhis2_shaped(simulated):
    data, _ = simulated
    assert set(DHIS2_COLUMNS) <= set(data.columns)


def test_pipeline_rejects_non_dhis2_input():
    with pytest.raises(ValueError, match="not DHIS2-shaped"):
        AnomalyPipeline().run(pd.DataFrame({"a": [1], "b": [2]}))


def test_truth_table_records_error_types(simulated):
    _, truth = simulated
    assert len(truth) > 0
    assert {"orgUnit", "period", "dataElement", "error_type"} <= set(truth.columns)


# -------------------------------------------------------------- detectors
def test_negative_values_flagged():
    df = _values("F1", "ANC_1st_visit", [100, 110, -50, 105])
    hits = detect_impossible_values(df)
    assert len(hits) == 1 and hits[0].value == -50
    assert hits[0].confidence > 0.95


def test_absurd_magnitude_flagged():
    df = _values("F1", "ANC_1st_visit", [100, 110, 999_999, 105])
    assert any(h.detector == "impossible_value" for h in detect_impossible_values(df, max_plausible=100_000))


def test_extra_digit_caught_as_outlier():
    df = _values("F1", "Malaria_tested", [400, 410, 395, 405, 398, 402, 4000, 401])
    hits = detect_statistical_outliers(df)
    assert any(h.value == 4000 for h in hits)


def test_mad_not_fooled_by_the_outlier_itself():
    """A single extreme value must not inflate the spread enough to hide itself.

    This is the reason for median/MAD over mean/standard deviation.
    """
    df = _values("F1", "X", [50, 51, 49, 52, 50, 51, 50, 9000])
    assert any(h.value == 9000 for h in detect_statistical_outliers(df))


def test_short_series_not_flagged():
    df = _values("F1", "X", [10, 900])
    assert detect_statistical_outliers(df) == []


def test_consistency_violation_flagged():
    df = pd.concat([
        _values("F1", "ANC_1st_visit", [100, 100]),
        _values("F1", "ANC_2nd_visit", [80, 180]),
    ])
    rule = ConsistencyRule("anc", "ANC_2nd_visit", "ANC_1st_visit", "lte")
    hits = detect_consistency_violations(df, [rule])
    assert len(hits) == 1 and hits[0].value == 180


def test_consistency_ratio_rule():
    df = pd.concat([
        _values("F1", "Malaria_confirmed", [100]),
        _values("F1", "Malaria_treated", [200]),
    ])
    rule = ConsistencyRule("t", "Malaria_treated", "Malaria_confirmed", "ratio_max", max_ratio=1.15)
    assert len(detect_consistency_violations(df, [rule])) == 1


def test_consistency_ignores_missing_element():
    df = _values("F1", "ANC_1st_visit", [100, 100])
    rule = ConsistencyRule("anc", "ANC_2nd_visit", "ANC_1st_visit", "lte")
    assert detect_consistency_violations(df, [rule]) == []


def test_digit_preference_flagged():
    df = _values("F1", "X", [round(v / 5) * 5 for v in range(100, 160, 2)])
    assert len(detect_digit_preference(df, min_values=10)) == 1


def test_digit_preference_not_flagged_on_natural_counts():
    rng = np.random.default_rng(0)
    df = _values("F1", "X", rng.integers(100, 400, size=40))
    assert detect_digit_preference(df, min_values=10) == []


def test_missing_reports_flagged():
    df = _values("F1", "X", [10, 11, 12])
    hits = detect_missing_reports(df, expected_periods=["202401", "202402", "202403", "202404", "202405"])
    assert len(hits) == 1


# --------------------------------------------------------------- pipeline
def test_pipeline_deduplicates_cells(simulated):
    data, _ = simulated
    flagged = AnomalyPipeline(rules=DEFAULT_RULES, max_plausible=100_000).run(data)
    key = ["orgUnit", "period", "dataElement"]
    assert not flagged.duplicated(subset=key).any()


def test_pipeline_ranks_high_severity_first(simulated):
    data, _ = simulated
    flagged = AnomalyPipeline(rules=DEFAULT_RULES, max_plausible=100_000).run(data)
    ranks = flagged["severity"].map({"high": 0, "medium": 1, "low": 2}).tolist()
    assert ranks == sorted(ranks)


def test_pipeline_precision_at_top_of_list(simulated):
    """Precision-first design: the top of the list must be trustworthy."""
    data, truth = simulated
    flagged = AnomalyPipeline(rules=DEFAULT_RULES, max_plausible=100_000).run(data)
    cell = flagged[~flagged.detector.isin(["digit_preference", "missing_report"])]
    assert evaluate_against_truth(cell, truth, top_n=10)["precision"] >= 0.9


def test_pipeline_recall_is_reasonable(simulated):
    data, truth = simulated
    flagged = AnomalyPipeline(rules=DEFAULT_RULES, max_plausible=100_000).run(data)
    cell = flagged[~flagged.detector.isin(["digit_preference", "missing_report"])]
    assert evaluate_against_truth(cell, truth)["recall"] >= 0.7


def test_clean_data_produces_few_flags():
    rng = np.random.default_rng(1)
    rows = []
    for org in ("F1", "F2"):
        for i in range(24):
            rows.append({
                "dataElement": "ANC_1st_visit", "period": str(202401 + i),
                "orgUnit": org, "categoryOptionCombo": "default",
                "value": float(rng.integers(95, 115)),
            })
    flagged = AnomalyPipeline().run(pd.DataFrame(rows))
    assert len(flagged) <= 2


def test_evaluation_metrics_are_consistent(simulated):
    data, truth = simulated
    flagged = AnomalyPipeline(rules=DEFAULT_RULES, max_plausible=100_000).run(data)
    cell = flagged[~flagged.detector.isin(["digit_preference", "missing_report"])]
    r = evaluate_against_truth(cell, truth)
    assert r["true_positives"] + r["false_positives"] == r["n_flagged"]
    assert 0.0 <= r["precision"] <= 1.0 and 0.0 <= r["recall"] <= 1.0
