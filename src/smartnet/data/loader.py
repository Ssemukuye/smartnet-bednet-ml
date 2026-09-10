"""Loading and structuring the tagged SMARTNET accelerometer extract.

The raw artefact is a Stata ``.dta`` file produced by the study team. This
module reads it, normalises timestamps, and derives the grouping keys that make
leakage-free validation possible.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from smartnet import config

logger = logging.getLogger(__name__)


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read the tagged Stata extract without applying value labels.

    Categorical conversion is deliberately disabled so that label columns stay
    numeric and the mapping in :data:`config.LABEL_MAPS` remains the single
    source of truth.
    """
    path = Path(path) if path else config.RAW_DIR / config.RAW_STATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Tagged extract not found at {path}.\n"
            "The study dataset is not distributed with this repository. "
            "See data/README.md for how to supply it, or run "
            "`python scripts/make_synthetic_sample.py` to generate a "
            "structurally identical synthetic file for demonstration."
        )
    df = pd.read_stata(path, convert_categoricals=False)
    logger.info("Loaded %s rows x %s columns from %s", *df.shape, path.name)
    return df


def add_time_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps and derive event / session grouping keys.

    Adds:

    ``timestamp``
        Parsed datetime.
    ``event_id``
        Index of the contiguous run of epochs spaced ``EVENT_GAP_SECONDS``
        apart. Each labelled motion is recorded as such a run, so this is the
        natural unit that must not be split across train and test.
    ``session_date``
        Calendar date of recording, used as a stricter grouping level.
    ``gap_seconds``
        Seconds since the previous epoch in time order.
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out[config.TIMESTAMP_COL])
    out = out.sort_values("timestamp").reset_index(drop=True)

    gap = out["timestamp"].diff().dt.total_seconds()
    out["gap_seconds"] = gap
    is_new_event = (gap > config.EVENT_GAP_SECONDS) | gap.isna()
    out["event_id"] = is_new_event.cumsum().astype(int)
    out["session_date"] = out["timestamp"].dt.date.astype(str)
    out["hour"] = out["timestamp"].dt.hour

    logger.info(
        "Derived %d events across %d recording days",
        out["event_id"].nunique(),
        out["session_date"].nunique(),
    )
    return out


def label_series(df: pd.DataFrame, label_col: str) -> pd.Series:
    """Return a label column mapped to human-readable class names."""
    if label_col not in config.LABEL_MAPS:
        raise KeyError(f"Unknown label column {label_col!r}")
    mapping = config.LABEL_MAPS[label_col]
    return df[label_col].astype("Int64").map(mapping).astype("string")


def load_analysis_frame(path: str | Path | None = None) -> pd.DataFrame:
    """Convenience loader: raw file plus derived time structure."""
    return add_time_structure(load_raw(path))


def event_summary(df: pd.DataFrame, label_col: str = config.PRIMARY_LABEL) -> pd.DataFrame:
    """One row per contiguous motion event, for inspection and reporting."""
    grp = df.groupby("event_id")
    summary = pd.DataFrame(
        {
            "n_epochs": grp.size(),
            "start": grp["timestamp"].min(),
            "end": grp["timestamp"].max(),
            "session_date": grp["session_date"].first(),
            "n_distinct_labels": grp[label_col].nunique(),
            "label": grp[label_col].first().map(config.LABEL_MAPS[label_col]),
        }
    )
    summary["duration_s"] = (
        summary["end"] - summary["start"]
    ).dt.total_seconds() + 1
    return summary.reset_index()


def assert_events_are_label_pure(df: pd.DataFrame) -> None:
    """Fail loudly if any derived event spans more than one class.

    Event-level grouping is only defensible if events are internally
    homogeneous. This is asserted rather than assumed.
    """
    impure = {}
    for label_col in config.LABEL_COLS:
        n = df.groupby("event_id")[label_col].nunique()
        bad = int((n > 1).sum())
        if bad:
            impure[label_col] = bad
    if impure:
        raise AssertionError(
            f"Derived events are not label-pure: {impure}. "
            "Revisit EVENT_GAP_SECONDS before grouping on event_id."
        )


def describe_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missingness table, sorted by severity."""
    n = len(df)
    miss = df.isna().sum()
    return (
        pd.DataFrame({"n_missing": miss, "pct_missing": (100 * miss / n).round(3)})
        .sort_values("n_missing", ascending=False)
        .reset_index(names="column")
    )
