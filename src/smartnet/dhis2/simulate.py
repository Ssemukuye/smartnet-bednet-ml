"""Generate a DHIS2-shaped dataset with known, labelled data-entry errors.

Two reasons this exists:

1. The public DHIS2 demo database has no ground truth about which values are
   wrong, so detector precision and recall cannot be measured on it.
2. Real MOH data cannot be used for a public portfolio.

Injecting errors of known type and location makes the detectors *evaluable*.
Every error is recorded in a truth table, so precision, recall and precision@k
are measured rather than claimed.

Data elements and monthly periods follow real Uganda HMIS reporting patterns
(ANC visits, malaria testing and treatment, ITN distribution) but every value
is synthetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from smartnet.dhis2.anomaly import ConsistencyRule

DATA_ELEMENTS = {
    "ANC_1st_visit": (140, 40),
    "ANC_2nd_visit": (110, 35),
    "Malaria_tested": (420, 120),
    "Malaria_confirmed": (180, 60),
    "Malaria_treated": (175, 58),
    "ITN_distributed": (95, 40),
}

#: Relationships that must hold in valid HMIS data.
DEFAULT_RULES = [
    ConsistencyRule(
        "anc2_lte_anc1", "ANC_2nd_visit", "ANC_1st_visit", "lte",
        explanation="ANC 2nd visits exceed ANC 1st visits at the same facility and period",
    ),
    ConsistencyRule(
        "confirmed_lte_tested", "Malaria_confirmed", "Malaria_tested", "lte",
        explanation="Confirmed malaria cases exceed the number of tests performed",
    ),
    ConsistencyRule(
        "treated_lte_confirmed", "Malaria_treated", "Malaria_confirmed", "ratio_max",
        max_ratio=1.15,
        explanation="Malaria treatments exceed confirmed cases by more than 15%",
    ),
]


def _periods(n: int = 24, start_year: int = 2024) -> list[str]:
    """DHIS2 monthly period identifiers, e.g. 202401."""
    out, y, m = [], start_year, 1
    for _ in range(n):
        out.append(f"{y}{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def generate(
    n_facilities: int = 40,
    n_periods: int = 24,
    error_rate: float = 0.03,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(data, truth)`` — DHIS2-shaped values plus the injected errors."""
    rng = np.random.default_rng(seed)
    periods = _periods(n_periods)
    org_units = [f"Facility_{i:03d}" for i in range(1, n_facilities + 1)]

    # Facility scale factor: a big hospital and a small HC II differ hugely.
    scale = {o: float(rng.lognormal(0.0, 0.55)) for o in org_units}
    # Seasonal malaria pattern, peaking in the rainy months.
    season = {p: 1.0 + 0.35 * np.sin(2 * np.pi * (int(p[-2:]) - 3) / 12) for p in periods}

    rows = []
    for org in org_units:
        for de, (base, sd) in DATA_ELEMENTS.items():
            level = base * scale[org]
            for p in periods:
                mult = season[p] if de.startswith("Malaria") else 1.0
                v = rng.normal(level * mult, sd * scale[org] * 0.5)
                rows.append({
                    "dataElement": de, "period": p, "orgUnit": org,
                    "categoryOptionCombo": "default",
                    "value": float(max(0, round(v))),
                })
    df = pd.DataFrame(rows)

    # Enforce logical consistency in the clean baseline, so that any violation
    # later is definitely an injected error and not generator noise.
    wide = df.pivot_table(index=["orgUnit", "period"], columns="dataElement", values="value")
    wide["ANC_2nd_visit"] = np.minimum(wide["ANC_2nd_visit"], wide["ANC_1st_visit"])
    wide["Malaria_confirmed"] = np.minimum(wide["Malaria_confirmed"], wide["Malaria_tested"])
    wide["Malaria_treated"] = np.minimum(wide["Malaria_treated"], wide["Malaria_confirmed"])
    df = wide.stack().reset_index().rename(columns={0: "value"})
    df["categoryOptionCombo"] = "default"
    df = df[["dataElement", "period", "orgUnit", "categoryOptionCombo", "value"]]

    # ------------------------------------------------------------ inject
    truth_rows: list[dict] = []
    n_errors = max(1, int(len(df) * error_rate))
    targets = rng.choice(df.index, size=n_errors, replace=False)

    error_types = [
        "extra_digit", "negative", "transposition",
        "consistency_break", "impossible_magnitude",
    ]
    weights = [0.30, 0.15, 0.20, 0.25, 0.10]

    for idx in targets:
        kind = rng.choice(error_types, p=weights)
        row = df.loc[idx]
        original = row["value"]

        if kind == "extra_digit":
            new = original * 10                      # keystroke repeated
        elif kind == "negative":
            new = -abs(original) if original else -5.0
        elif kind == "transposition":
            s = str(int(original))
            new = float(s[1] + s[0] + s[2:]) if len(s) >= 2 else original + 9
        elif kind == "impossible_magnitude":
            new = 250_000.0
        else:  # consistency_break — value plausible alone, impossible in context
            if row["dataElement"] == "ANC_2nd_visit":
                partner = df[
                    (df.orgUnit == row.orgUnit) & (df.period == row.period)
                    & (df.dataElement == "ANC_1st_visit")
                ]["value"]
                new = float(partner.iloc[0]) * 1.8 if len(partner) else original * 2
            else:
                partner = df[
                    (df.orgUnit == row.orgUnit) & (df.period == row.period)
                    & (df.dataElement == "Malaria_tested")
                ]["value"]
                new = float(partner.iloc[0]) * 1.6 if len(partner) else original * 2

        if new == original:
            continue
        df.loc[idx, "value"] = float(round(new))
        truth_rows.append({
            "orgUnit": row["orgUnit"], "period": row["period"],
            "dataElement": row["dataElement"],
            "original_value": original, "corrupted_value": float(round(new)),
            "error_type": kind,
        })

    # A facility that estimates rather than counts — digit preference.
    rounder = org_units[0]
    mask = df.orgUnit == rounder
    df.loc[mask, "value"] = (df.loc[mask, "value"] / 5).round() * 5

    # A facility that stops reporting halfway through — missing reports.
    dropper = org_units[1]
    drop_mask = (df.orgUnit == dropper) & (df.period > periods[n_periods // 2])
    df = df[~drop_mask].reset_index(drop=True)

    truth = pd.DataFrame(truth_rows)
    return df, truth
