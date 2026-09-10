"""Data-entry anomaly detection for DHIS2 aggregate data.

Built against the DHIS2 aggregate data model — a data value is identified by
``(dataElement, period, orgUnit, categoryOptionCombo)`` — so the detectors here
run unchanged on an export from the DHIS2 Web API
(``/api/dataValueSets``) or on the public demo database.

Design principle: **precision over recall.** A district health officer who is
sent fifty false flags a month stops opening the report, and then the tool has
made data quality worse rather than better. Every detector therefore emits a
confidence and a plain-language explanation, and the pipeline ranks rather than
dumps. Simple, explainable rules come first; a model is only justified where
rules demonstrably miss something.

The detector families mirror what DHIS2 itself offers (min-max, std-dev outlier
and follow-up analysis) and add consistency, level-shift and digit-preference
checks that catch different failure modes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Severity = Literal["low", "medium", "high"]

#: Columns of a DHIS2 dataValueSet export.
DHIS2_COLUMNS = [
    "dataElement",
    "period",
    "orgUnit",
    "categoryOptionCombo",
    "value",
]


@dataclass
class Anomaly:
    """One flagged data value, with the reason a human can act on."""

    dataElement: str
    period: str
    orgUnit: str
    value: float
    detector: str
    severity: Severity
    confidence: float
    explanation: str
    expected: float | None = None

    def as_dict(self) -> dict:
        return {
            "orgUnit": self.orgUnit,
            "period": self.period,
            "dataElement": self.dataElement,
            "value": self.value,
            "expected": self.expected,
            "detector": self.detector,
            "severity": self.severity,
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation,
        }


@dataclass
class ConsistencyRule:
    """A logical relationship that must hold between two data elements.

    Example: ANC 2nd visits can never exceed ANC 1st visits at the same
    facility and period. Violations are near-certain data-entry errors, which
    is why they carry the highest confidence in the pipeline.
    """

    name: str
    numerator: str
    denominator: str
    relation: Literal["lte", "ratio_max"] = "lte"
    max_ratio: float = 1.0
    explanation: str = ""


# --------------------------------------------------------------- detectors
def detect_impossible_values(
    df: pd.DataFrame, non_negative: bool = True, max_plausible: float | None = None
) -> list[Anomaly]:
    """Values that cannot be true regardless of context.

    Negative counts and absurd magnitudes are unambiguous entry errors —
    usually a stray minus sign or a repeated keystroke. Highest confidence of
    any detector because there is no legitimate explanation.
    """
    out: list[Anomaly] = []
    for _, r in df.iterrows():
        v = r["value"]
        if non_negative and v < 0:
            out.append(Anomaly(
                r["dataElement"], r["period"], r["orgUnit"], v,
                "impossible_value", "high", 0.99,
                "Negative value reported for a count data element",
                expected=0.0,
            ))
        elif max_plausible is not None and v > max_plausible:
            out.append(Anomaly(
                r["dataElement"], r["period"], r["orgUnit"], v,
                "impossible_value", "high", 0.95,
                f"Value exceeds the plausible ceiling of {max_plausible:,.0f}",
                expected=max_plausible,
            ))
    return out


def detect_statistical_outliers(
    df: pd.DataFrame, threshold: float = 3.5, min_history: int = 6
) -> list[Anomaly]:
    """Modified z-score outliers within each (orgUnit, dataElement) series.

    Uses median and MAD rather than mean and standard deviation: a single
    extreme mis-keyed value inflates the standard deviation enough to hide
    itself, which is the classic failure of naive z-score screening on exactly
    this kind of data.
    """
    out: list[Anomaly] = []
    for (org, de), grp in df.groupby(["orgUnit", "dataElement"]):
        vals = grp["value"].astype(float)
        if len(vals) < min_history:
            continue
        median = vals.median()
        mad = (vals - median).abs().median()
        if mad == 0:
            continue
        mz = 0.6745 * (vals - median) / mad
        for idx, score in mz.items():
            if abs(score) > threshold:
                r = grp.loc[idx]
                conf = float(min(0.95, 0.55 + 0.05 * (abs(score) - threshold)))
                out.append(Anomaly(
                    de, r["period"], org, float(r["value"]),
                    "statistical_outlier",
                    "high" if abs(score) > 2 * threshold else "medium",
                    conf,
                    f"Modified z-score {score:.1f} against a facility median of "
                    f"{median:,.0f} (MAD {mad:,.0f})",
                    expected=float(median),
                ))
    return out


def detect_consistency_violations(
    df: pd.DataFrame, rules: list[ConsistencyRule]
) -> list[Anomaly]:
    """Logical relationships between related data elements.

    These catch errors that survive every univariate check, because each value
    is individually plausible and only the pair is impossible.
    """
    out: list[Anomaly] = []
    wide = df.pivot_table(
        index=["orgUnit", "period"], columns="dataElement", values="value", aggfunc="sum"
    )
    for rule in rules:
        if rule.numerator not in wide.columns or rule.denominator not in wide.columns:
            continue
        num, den = wide[rule.numerator], wide[rule.denominator]
        if rule.relation == "lte":
            bad = num > den
        else:
            bad = (den > 0) & (num / den.replace(0, np.nan) > rule.max_ratio)

        for (org, period), is_bad in bad.items():
            if not bool(is_bad):
                continue
            n, d = num.loc[(org, period)], den.loc[(org, period)]
            out.append(Anomaly(
                rule.numerator, period, org, float(n),
                f"consistency:{rule.name}", "high", 0.92,
                rule.explanation
                or f"{rule.numerator} ({n:,.0f}) exceeds {rule.denominator} ({d:,.0f})",
                expected=float(d),
            ))
    return out


def detect_level_shift(
    df: pd.DataFrame, window: int = 3, ratio: float = 3.0, min_history: int = 6
) -> list[Anomaly]:
    """Sudden sustained change in a facility's reporting level.

    Distinct from a point outlier: a level shift usually means a denominator
    changed, a facility merged, or a reporting definition moved — a metadata
    problem rather than a typing error, and it needs a different response.
    """
    out: list[Anomaly] = []
    for (org, de), grp in df.groupby(["orgUnit", "dataElement"]):
        g = grp.sort_values("period").reset_index(drop=True)
        if len(g) < min_history + window:
            continue
        vals = g["value"].astype(float)
        for i in range(min_history, len(g) - window + 1):
            before = vals.iloc[max(0, i - window):i]
            after = vals.iloc[i:i + window]
            if before.mean() <= 0:
                continue
            r = after.mean() / before.mean()
            if r > ratio or r < 1 / ratio:
                out.append(Anomaly(
                    de, str(g.loc[i, "period"]), org, float(vals.iloc[i]),
                    "level_shift", "medium", 0.70,
                    f"Reporting level changed {r:.1f}x from a {window}-period mean of "
                    f"{before.mean():,.0f} to {after.mean():,.0f}",
                    expected=float(before.mean()),
                ))
                break  # one flag per series
    return out


def detect_digit_preference(
    df: pd.DataFrame, min_values: int = 20, threshold: float = 0.45
) -> list[Anomaly]:
    """Excess of round numbers, which indicates estimation rather than counting.

    If a facility's values end in 0 or 5 far more often than chance, the
    numbers are probably being guessed at the end of the month rather than
    tallied. It is a data *culture* problem, not a single bad cell, so it is
    reported at facility level.
    """
    out: list[Anomaly] = []
    for (org, de), grp in df.groupby(["orgUnit", "dataElement"]):
        vals = grp["value"].astype(float)
        vals = vals[vals >= 10]
        if len(vals) < min_values:
            continue
        last_digit = (vals.astype(int) % 10)
        round_share = float(last_digit.isin([0, 5]).mean())
        if round_share > threshold:
            out.append(Anomaly(
                de, "all", org, round_share,
                "digit_preference", "low", 0.60,
                f"{round_share:.0%} of values end in 0 or 5 (chance is 20%), "
                "suggesting estimation rather than counting",
                expected=0.20,
            ))
    return out


def detect_missing_reports(
    df: pd.DataFrame, expected_periods: list[str] | None = None
) -> list[Anomaly]:
    """Facilities that stopped reporting a data element they used to report."""
    out: list[Anomaly] = []
    periods = sorted(expected_periods or df["period"].unique().tolist())
    for (org, de), grp in df.groupby(["orgUnit", "dataElement"]):
        seen = set(grp["period"])
        if len(seen) < 2:
            continue
        first = min(seen)
        expected_after_first = [p for p in periods if p >= first]
        missing = [p for p in expected_after_first if p not in seen]
        if missing:
            out.append(Anomaly(
                de, ",".join(missing[:3]) + ("…" if len(missing) > 3 else ""),
                org, float("nan"), "missing_report",
                "high" if len(missing) > 2 else "medium",
                0.85,
                f"No value reported for {len(missing)} period(s) after the facility "
                f"began reporting in {first}",
            ))
    return out


# ---------------------------------------------------------------- pipeline
@dataclass
class AnomalyPipeline:
    """Runs the detector suite and returns a ranked, deduplicated report."""

    rules: list[ConsistencyRule] = field(default_factory=list)
    max_plausible: float | None = None
    outlier_threshold: float = 3.5
    min_confidence: float = 0.0

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_cols = set(DHIS2_COLUMNS[:3] + ["value"]) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Input is not DHIS2-shaped; missing {sorted(missing_cols)}")

        detectors: list[tuple[str, Callable[[], list[Anomaly]]]] = [
            ("impossible", lambda: detect_impossible_values(df, True, self.max_plausible)),
            ("outlier", lambda: detect_statistical_outliers(df, self.outlier_threshold)),
            ("consistency", lambda: detect_consistency_violations(df, self.rules)),
            ("level_shift", lambda: detect_level_shift(df)),
            ("digit_preference", lambda: detect_digit_preference(df)),
            ("missing", lambda: detect_missing_reports(df)),
        ]

        found: list[Anomaly] = []
        for name, fn in detectors:
            hits = fn()
            logger.info("detector %-18s flagged %3d", name, len(hits))
            found.extend(hits)

        if not found:
            return pd.DataFrame(columns=[*Anomaly.__annotations__])

        out = pd.DataFrame([a.as_dict() for a in found])
        out = out[out["confidence"] >= self.min_confidence]

        # One row per cell: keep the highest-confidence explanation.
        out = (
            out.sort_values("confidence", ascending=False)
            .drop_duplicates(subset=["orgUnit", "period", "dataElement"], keep="first")
        )
        sev_rank = {"high": 0, "medium": 1, "low": 2}
        out["_sev"] = out["severity"].map(sev_rank)
        return (
            out.sort_values(["_sev", "confidence"], ascending=[True, False])
            .drop(columns="_sev")
            .reset_index(drop=True)
        )


def evaluate_against_truth(
    flagged: pd.DataFrame, truth: pd.DataFrame, top_n: int | None = None
) -> dict[str, float]:
    """Precision, recall and F1 against a set of known injected errors.

    Without this a rule-based detector is just an assertion. ``top_n`` reports
    precision@k, which is what actually matters operationally: a district
    officer works the top of the list, not all of it.
    """
    key = ["orgUnit", "period", "dataElement"]
    sub = flagged.head(top_n) if top_n else flagged

    flagged_keys = set(map(tuple, sub[key].astype(str).values))
    truth_keys = set(map(tuple, truth[key].astype(str).values))

    tp = len(flagged_keys & truth_keys)
    fp = len(flagged_keys - truth_keys)
    fn = len(truth_keys - flagged_keys)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n_flagged": len(sub),
        "n_true_errors": len(truth_keys),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
