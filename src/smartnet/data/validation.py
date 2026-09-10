"""Data-quality validation for the tagged accelerometer extract.

Every check returns a structured result rather than raising, so a full quality
report can be produced in one pass. This mirrors the quality-control practice
used on the study itself: find everything, then decide what blocks analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
import pandas as pd

from smartnet import config

Severity = Literal["pass", "warn", "fail"]


@dataclass
class CheckResult:
    name: str
    severity: Severity
    message: str
    n_affected: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def check_required_columns(df: pd.DataFrame) -> CheckResult:
    expected = set(config.ALL_FEATURES + config.LABEL_COLS + [config.TIMESTAMP_COL])
    missing = sorted(expected - set(df.columns))
    if missing:
        return CheckResult(
            "required_columns", "fail", f"Missing columns: {missing}", len(missing)
        )
    return CheckResult("required_columns", "pass", f"All {len(expected)} expected columns present")


def check_missing_values(df: pd.DataFrame) -> CheckResult:
    cols = [c for c in config.ALL_FEATURES + config.LABEL_COLS if c in df.columns]
    n = int(df[cols].isna().sum().sum())
    if n == 0:
        return CheckResult("missing_values", "pass", "No missing values in features or labels")
    return CheckResult("missing_values", "warn", f"{n} missing cells across features/labels", n)


def check_duplicate_rows(df: pd.DataFrame) -> CheckResult:
    cols = [c for c in config.ALL_FEATURES if c in df.columns]
    n = int(df.duplicated(subset=cols).sum())
    sev: Severity = "pass" if n == 0 else "warn"
    return CheckResult("duplicate_feature_rows", sev, f"{n} rows duplicated on the feature vector", n)


def check_timestamp_validity(df: pd.DataFrame) -> CheckResult:
    ts = pd.to_datetime(df[config.TIMESTAMP_COL], errors="coerce")
    n_bad = int(ts.isna().sum())
    if n_bad:
        return CheckResult("timestamp_validity", "fail", f"{n_bad} unparseable timestamps", n_bad)
    n_dupe = int(ts.duplicated().sum())
    if n_dupe:
        return CheckResult(
            "timestamp_validity", "warn",
            f"Timestamps parse cleanly but {n_dupe} are duplicated", n_dupe,
        )
    return CheckResult(
        "timestamp_validity", "pass",
        f"All {len(ts)} timestamps parse and are unique "
        f"({ts.min():%Y-%m-%d} to {ts.max():%Y-%m-%d})",
    )


def check_label_consistency(df: pd.DataFrame) -> CheckResult:
    """The nested label schemes must agree with one another.

    ``motion_something`` is the binary collapse of the finer schemes, so every
    row labelled Nothing in one scheme must be Nothing in all of them.
    """
    problems = []
    nothing_binary = df["motion_something"] == 0
    for col in ("motion_3cat", "motion_4cat", "motion_5cat"):
        nothing_multi = df[col] == 0
        mismatch = int((nothing_binary != nothing_multi).sum())
        if mismatch:
            problems.append(f"{col}: {mismatch} rows disagree with motion_something")

    # 5cat Enter/Exit must collapse to 4cat "Enter or exit"
    ee = df["motion_5cat"].isin([3, 4])
    mismatch_ee = int((ee != (df["motion_4cat"] == 3)).sum())
    if mismatch_ee:
        problems.append(f"motion_5cat Enter/Exit does not collapse into motion_4cat ({mismatch_ee} rows)")

    if problems:
        return CheckResult("label_consistency", "fail", "; ".join(problems), len(problems))
    return CheckResult("label_consistency", "pass", "Nested label schemes are mutually consistent")


def check_value_ranges(df: pd.DataFrame) -> CheckResult:
    out_of_range = {}
    for col, (lo, hi) in config.PLAUSIBLE_RANGES.items():
        if col not in df.columns:
            continue
        n = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n:
            out_of_range[col] = n
    if out_of_range:
        return CheckResult(
            "value_ranges", "warn",
            f"Values outside plausible bounds: {out_of_range}",
            sum(out_of_range.values()),
        )
    return CheckResult("value_ranges", "pass", "All checked features within plausible bounds")


def check_class_support(df: pd.DataFrame, label_col: str = config.PRIMARY_LABEL) -> CheckResult:
    counts = df[label_col].value_counts()
    smallest = int(counts.min())
    name = config.LABEL_MAPS[label_col].get(int(counts.idxmin()), "?")
    if smallest < 30:
        return CheckResult(
            "class_support", "warn",
            f"Rarest class in {label_col} is {name!r} with only {smallest} epochs; "
            "per-class estimates will be unstable",
            smallest,
        )
    return CheckResult(
        "class_support", "pass",
        f"Rarest class in {label_col} is {name!r} with {smallest} epochs",
        smallest,
    )


def check_event_purity(df: pd.DataFrame) -> CheckResult:
    if "event_id" not in df.columns:
        return CheckResult("event_purity", "warn", "event_id not derived; run add_time_structure first")
    bad = 0
    for col in config.LABEL_COLS:
        bad += int((df.groupby("event_id")[col].nunique() > 1).sum())
    if bad:
        return CheckResult("event_purity", "fail", f"{bad} events span multiple classes", bad)
    n_events = df["event_id"].nunique()
    return CheckResult("event_purity", "pass", f"All {n_events} derived events are label-pure", n_events)


ALL_CHECKS = [
    check_required_columns,
    check_missing_values,
    check_duplicate_rows,
    check_timestamp_validity,
    check_label_consistency,
    check_value_ranges,
    check_class_support,
    check_event_purity,
]


def run_all(df: pd.DataFrame) -> pd.DataFrame:
    """Run every check and return a tidy report."""
    rows = []
    for check in ALL_CHECKS:
        try:
            rows.append(check(df).as_dict())
        except Exception as exc:  # a broken check must not hide the others
            rows.append(
                CheckResult(check.__name__, "fail", f"Check errored: {exc}").as_dict()
            )
    return pd.DataFrame(rows)[["name", "severity", "n_affected", "message"]]


def has_blocking_failure(report: pd.DataFrame) -> bool:
    return bool((report["severity"] == "fail").any())
