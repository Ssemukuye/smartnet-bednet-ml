"""Explore a live DHIS2 instance and run the anomaly detectors against it.

Reproduces, in Python, the exploration originally done by hand against the
DHIS2 2.43.1 Sierra Leone demo — so the findings in ``docs/DHIS2_NOTES.md`` can
be re-derived rather than taken on trust.

Usage
-----
    export DHIS2_BASE_URL="https://play.im.dhis2.org/stable-2-43-1"
    export DHIS2_USERNAME="admin"
    export DHIS2_PASSWORD="..."          # never commit this
    PYTHONPATH=src python scripts/explore_dhis2.py

Options
-------
    --district O6uvpzGd5pu    organisation unit to pull facilities from (default: Bo)
    --period LAST_12_MONTHS   analytics relative period
    --no-detect               metadata exploration only, skip the detectors

Everything is read-only. Nothing is written back to DHIS2.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from smartnet import config
from smartnet.dhis2.anomaly import AnomalyPipeline
from smartnet.dhis2.client import DHIS2Client

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("explore")

# ANC data elements in the Sierra Leone demo. On another instance, discover
# these with `client.data_elements(name_filter="ANC")` instead of hard-coding.
DEMO_ANC = {
    "fbfJHSPpUQD": "ANC 1st visit",
    "cYeuwXTCPkU": "ANC 2nd visit",
    "Jtf34kNZhzP": "ANC 3rd visit",
}
DEMO_BO_DISTRICT = "O6uvpzGd5pu"


def rule(title: str) -> None:
    log.info("")
    log.info("=" * 72)
    log.info(title)
    log.info("=" * 72)


def main(district: str, period: str, detect: bool) -> int:
    if not os.environ.get("DHIS2_BASE_URL"):
        log.error("DHIS2_BASE_URL is not set. See the docstring at the top of this file.")
        return 2

    client = DHIS2Client()
    log.info("connecting to %s", client.base_url)

    # ------------------------------------------------------------------ 1
    rule("STEP 1 — What is in this instance?")
    info = client.system_info()
    log.info("DHIS2 version %s (revision %s), %s calendar",
             info.get("version"), info.get("revision"), info.get("calendar"))

    levels = client.org_unit_levels()
    log.info("organisation hierarchy: %s",
             " -> ".join(f"{k}:{v}" for k, v in sorted(levels.items())))

    counts = {}
    for resource in ("dataElements", "dataSets", "organisationUnits",
                     "indicators", "validationRules"):
        j = client.get(resource, pageSize=1, fields="id")
        counts[resource] = (j.get("pager") or {}).get("total", "n/a")
    for k, v in counts.items():
        log.info("  %-20s %s", k, v)

    # ------------------------------------------------------------------ 2
    rule("STEP 2 — Import DHIS2's own validation rules")
    log.info("DHIS2 already encodes consistency logic such as 'ANC 2 <= ANC 1'.")
    log.info("Importing beats hand-coding: the detector then inherits whatever")
    log.info("the Ministry has configured, and stays in step when it changes.")
    log.info("")

    rules, raw = client.fetch_validation_rules()
    log.info("%d rules on the server, %d translatable into consistency checks",
             len(raw), len(rules))
    log.info("")
    log.info("translated (first 8):")
    for r in rules[:8]:
        log.info("   %-46s  %s <= %s", r.name[:44], r.numerator, r.denominator)

    skipped = raw[~raw["translatable"]]
    if len(skipped):
        log.info("")
        log.info("skipped %d compound rules — a rule summing several data elements", len(skipped))
        log.info("is not evaluable as a single comparison, and guessing would be worse")
        log.info("than skipping. Examples:")
        for _, s in skipped.head(3).iterrows():
            log.info("   %s", s["name"])

    raw.to_csv(config.RESULTS_DIR / "dhis2_live_validation_rules.csv", index=False)

    if not detect:
        log.info("\n--no-detect given; stopping after metadata.")
        return 0

    # ------------------------------------------------------------------ 3
    rule("STEP 3 — Pull real facility data")
    df = client.fetch_analytics(
        data_element_uids=list(DEMO_ANC),
        org_unit_uids=[district],
        period=period,
        org_unit_level=4,          # facilities beneath the named org unit
    )
    if df.empty:
        log.warning("No data returned. The demo database moves its data window —")
        log.warning("try a different --period, or check the org unit UID.")
        return 1

    log.info("%d values | %d facilities | %d periods | %d data elements",
             len(df), df.orgUnit.nunique(), df.period.nunique(), df.dataElement.nunique())
    log.info("period range: %s to %s", df.period.min(), df.period.max())

    # ------------------------------------------------------------------ 4
    rule("STEP 4 — Run the detectors on real data")
    pipeline = AnomalyPipeline(rules=rules, max_plausible=100_000)
    flagged = pipeline.run(df)
    log.info("")
    log.info("%d values flagged", len(flagged))

    if len(flagged):
        log.info("")
        log.info("by detector:")
        for det, n in flagged.detector.value_counts().items():
            log.info("   %-34s %3d", det, n)

        cols = ["orgUnit", "period", "dataElement", "value", "expected",
                "severity", "confidence", "explanation"]
        log.info("")
        log.info("top flags a district officer would work first:")
        log.info("\n%s", flagged[cols].head(8).to_string(index=False, max_colwidth=52))

        out = config.RESULTS_DIR / "dhis2_live_anomalies.csv"
        flagged.to_csv(out, index=False)
        log.info("")
        log.info("written to %s", out.name)

    # ------------------------------------------------------------------ 5
    rule("STEP 5 — Read the result honestly")
    log.info("Large outliers here are usually genuine data-entry signatures: a")
    log.info("facility reporting ~24 visits a month suddenly reporting ~1,900 is")
    log.info("an extra digit, not an outbreak.")
    log.info("")
    log.info("Consistency violations are a different matter on THIS server. The")
    log.info("demo database is randomly generated, so it breaks its own validation")
    log.info("rules constantly. A high violation count here says the detector")
    log.info("fires correctly. It says nothing about real-world data quality, and")
    log.info("must not be reported as if it did.")

    manifest = {
        "base_url": client.base_url,
        "dhis2_version": info.get("version"),
        "counts": counts,
        "rules_total": int(len(raw)),
        "rules_translated": int(len(rules)),
        "values_pulled": int(len(df)),
        "facilities": int(df.orgUnit.nunique()),
        "periods": int(df.period.nunique()),
        "flagged": int(len(flagged)),
        "detector_counts": flagged.detector.value_counts().to_dict() if len(flagged) else {},
    }
    (config.RESULTS_DIR / "dhis2_live_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("")
    log.info("manifest written to reports/model_results/dhis2_live_manifest.json")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--district", default=DEMO_BO_DISTRICT,
                    help="organisation unit UID to pull facilities beneath")
    ap.add_argument("--period", default="LAST_12_MONTHS")
    ap.add_argument("--no-detect", action="store_true",
                    help="metadata exploration only")
    a = ap.parse_args()
    sys.exit(main(a.district, a.period, not a.no_detect))
