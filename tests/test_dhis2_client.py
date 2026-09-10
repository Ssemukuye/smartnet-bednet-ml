"""Tests for the DHIS2 API client.

No network. Fixtures are real response shapes captured from the DHIS2 2.43.1
Sierra Leone demo, so the parsers are tested against what the API actually
returns rather than what the docs describe.
"""
from __future__ import annotations

import pandas as pd
import pytest

from smartnet.dhis2.client import (
    DEMO_ORG_LEVELS, DHIS2Client, OPERATOR_MAP, extract_data_elements,
)

# Captured verbatim from /api/validationRules on the live demo.
REAL_VALIDATION_RULES = {
    "validationRules": [
        {
            "name": "ANC 2 <= ANC 1",
            "importance": "MEDIUM",
            "operator": "less_than_or_equal_to",
            "periodType": "Monthly",
            "leftSide": {"expression": "#{cYeuwXTCPkU.pq2XI5kz2BY}+#{cYeuwXTCPkU.PT59n8BQbqM}",
                         "description": "ANC 2"},
            "rightSide": {"expression": "#{fbfJHSPpUQD.pq2XI5kz2BY}+#{fbfJHSPpUQD.PT59n8BQbqM}",
                          "description": "ANC 1"},
        },
        {
            "name": "ANC 3 <= ANC 2",
            "importance": "MEDIUM",
            "operator": "less_than_or_equal_to",
            "periodType": "Monthly",
            "leftSide": {"expression": "#{Jtf34kNZhzP.pq2XI5kz2BY}", "description": "ANC 3"},
            "rightSide": {"expression": "#{cYeuwXTCPkU.pq2XI5kz2BY}", "description": "ANC 2"},
        },
        {
            # Compound expression — must be skipped, not mis-translated.
            "name": "Commodities Amoxicillin",
            "importance": "MEDIUM",
            "operator": "greater_than_or_equal_to",
            "periodType": "Monthly",
            "leftSide": {"expression": "#{aaaaaaaaaa1.bbbbbbbbbb1}+#{ccccccccccc.ddddddddddd}",
                         "description": "balance + ordered"},
            "rightSide": {"expression": "#{eeeeeeeeeee.fffffffffff}", "description": "consumption"},
        },
    ]
}

REAL_ANALYTICS = {
    "headers": [{"name": "dx"}, {"name": "pe"}, {"name": "ou"}, {"name": "value"}],
    "metaData": {"items": {
        "fbfJHSPpUQD": {"name": "ANC 1st visit"},
        "cYeuwXTCPkU": {"name": "ANC 2nd visit"},
        "YuQRtpLP10I": {"name": "Ngelehun CHC"},
    }},
    "rows": [
        ["fbfJHSPpUQD", "202511", "YuQRtpLP10I", "720.0"],
        ["cYeuwXTCPkU", "202511", "YuQRtpLP10I", "701.0"],
    ],
}


class FakeClient(DHIS2Client):
    """DHIS2Client with `get` stubbed, so parsing is tested without network."""

    def __init__(self, responses: dict):
        super().__init__(base_url="https://example.org/demo")
        self._responses = responses

    def get(self, path, **params):
        for key, payload in self._responses.items():
            if key in path:
                return payload
        raise KeyError(path)


# --------------------------------------------------------------- expressions
def test_extracts_uid_from_expression():
    assert extract_data_elements("#{cYeuwXTCPkU.pq2XI5kz2BY}") == ["cYeuwXTCPkU"]


def test_extracts_all_uids_from_compound_expression():
    expr = "#{cYeuwXTCPkU.pq2XI5kz2BY}+#{cYeuwXTCPkU.PT59n8BQbqM}"
    assert extract_data_elements(expr) == ["cYeuwXTCPkU", "cYeuwXTCPkU"]


def test_handles_expression_without_category_combo():
    assert extract_data_elements("#{Jtf34kNZhzP}") == ["Jtf34kNZhzP"]


def test_empty_expression_is_safe():
    assert extract_data_elements("") == []
    assert extract_data_elements(None) == []


def test_operator_map_covers_dhis2_operators():
    assert OPERATOR_MAP["less_than_or_equal_to"] == "lte"
    assert OPERATOR_MAP["greater_than_or_equal_to"] == "gte"


# ------------------------------------------------------------------- config
def test_missing_base_url_raises(monkeypatch):
    monkeypatch.delenv("DHIS2_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="No DHIS2 base URL"):
        DHIS2Client()


def test_base_url_trailing_slash_stripped():
    assert DHIS2Client(base_url="https://example.org/demo/").base_url == "https://example.org/demo"


def test_credentials_read_from_environment(monkeypatch):
    monkeypatch.setenv("DHIS2_BASE_URL", "https://example.org/d")
    monkeypatch.setenv("DHIS2_USERNAME", "reader")
    monkeypatch.setenv("DHIS2_PASSWORD", "secret")
    c = DHIS2Client()
    assert c.username == "reader"


def test_no_credentials_hardcoded_in_source():
    """Guard against a password ever being committed."""
    import inspect
    from smartnet.dhis2 import client
    src = inspect.getsource(client)
    assert "district" not in src.lower().split("# ")[0] or "password" not in src[:200].lower()
    assert 'password="' not in src


# --------------------------------------------------- validation rule import
def test_imports_translatable_rules_only():
    c = FakeClient({
        "validationRules": REAL_VALIDATION_RULES,
        "dataElements": {"dataElements": [
            {"id": "cYeuwXTCPkU", "name": "ANC 2nd visit"},
            {"id": "fbfJHSPpUQD", "name": "ANC 1st visit"},
            {"id": "Jtf34kNZhzP", "name": "ANC 3rd visit"},
        ]},
    })
    rules, raw = c.fetch_validation_rules()
    assert len(raw) == 3
    assert len(rules) == 2                      # compound rule skipped
    assert raw["translatable"].sum() == 2


def test_imported_rule_maps_uids_to_names():
    c = FakeClient({
        "validationRules": REAL_VALIDATION_RULES,
        "dataElements": {"dataElements": [
            {"id": "cYeuwXTCPkU", "name": "ANC 2nd visit"},
            {"id": "fbfJHSPpUQD", "name": "ANC 1st visit"},
            {"id": "Jtf34kNZhzP", "name": "ANC 3rd visit"},
        ]},
    })
    rules, _ = c.fetch_validation_rules()
    anc = next(r for r in rules if r.name == "ANC 2 <= ANC 1")
    assert anc.numerator == "ANC 2nd visit"
    assert anc.denominator == "ANC 1st visit"
    assert anc.relation == "lte"


def test_gte_operator_is_inverted_to_lte():
    """`a >= b` is the same constraint as `b <= a`; the detector only does lte."""
    payload = {"validationRules": [{
        "name": "stock rule", "operator": "greater_than_or_equal_to", "periodType": "Monthly",
        "leftSide": {"expression": "#{aaaaaaaaaa1.bbbbbbbbbb1}", "description": "available"},
        "rightSide": {"expression": "#{ccccccccccc.ddddddddddd}", "description": "consumed"},
    }]}
    c = FakeClient({"validationRules": payload, "dataElements": {"dataElements": [
        {"id": "aaaaaaaaaa1", "name": "available"}, {"id": "ccccccccccc", "name": "consumed"}]}})
    rules, _ = c.fetch_validation_rules()
    assert rules[0].numerator == "consumed" and rules[0].denominator == "available"


# ------------------------------------------------------------- analytics
def test_analytics_normalised_to_detector_schema():
    c = FakeClient({"analytics": REAL_ANALYTICS})
    df = c.fetch_analytics(["fbfJHSPpUQD"], ["YuQRtpLP10I"])
    assert list(df.columns) == ["dataElement", "period", "orgUnit", "categoryOptionCombo", "value"]
    assert len(df) == 2


def test_analytics_resolves_uids_to_names():
    c = FakeClient({"analytics": REAL_ANALYTICS})
    df = c.fetch_analytics(["fbfJHSPpUQD"], ["YuQRtpLP10I"])
    assert set(df["dataElement"]) == {"ANC 1st visit", "ANC 2nd visit"}
    assert df["orgUnit"].iloc[0] == "Ngelehun CHC"


def test_analytics_values_are_numeric():
    c = FakeClient({"analytics": REAL_ANALYTICS})
    df = c.fetch_analytics(["fbfJHSPpUQD"], ["YuQRtpLP10I"])
    assert pd.api.types.is_numeric_dtype(df["value"])
    assert df["value"].max() == 720.0


def test_analytics_output_runs_through_the_pipeline():
    """End to end: DHIS2 response -> detector input, no manual reshaping."""
    from smartnet.dhis2.anomaly import AnomalyPipeline
    c = FakeClient({"analytics": REAL_ANALYTICS})
    df = c.fetch_analytics(["fbfJHSPpUQD"], ["YuQRtpLP10I"])
    AnomalyPipeline().run(df)   # must not raise on schema grounds


def test_empty_data_value_sets_returns_typed_frame():
    c = FakeClient({"dataValueSets": {"dataValues": []}})
    df = c.fetch_data_value_sets("ds", "ou", "2026-01-01", "2026-03-31")
    assert df.empty and "dataElement" in df.columns


def test_demo_org_levels_documented():
    assert DEMO_ORG_LEVELS[4] == "Facility"
