"""Run and evaluate DHIS2 data-entry anomaly detection.

    PYTHONPATH=src python scripts/run_dhis2_anomaly.py

Generates a DHIS2-shaped dataset with known injected errors, runs the detector
suite, and measures precision, recall and precision@k against ground truth.
"""
from __future__ import annotations

import json
import logging
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from smartnet import config
from smartnet.dhis2.anomaly import AnomalyPipeline, evaluate_against_truth
from smartnet.dhis2.simulate import DEFAULT_RULES, generate

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("dhis2")

OUT = config.RESULTS_DIR


def main() -> None:
    log.info("=" * 70)
    log.info("DHIS2 DATA-ENTRY ANOMALY DETECTION")
    log.info("=" * 70)

    data, truth = generate(n_facilities=40, n_periods=24, error_rate=0.03, seed=42)
    log.info(
        "dataset: %s values | %s facilities | %s periods | %s data elements",
        f"{len(data):,}", data.orgUnit.nunique(),
        data.period.nunique(), data.dataElement.nunique(),
    )
    log.info("injected errors: %d (%.1f%% of values)", len(truth), 100 * len(truth) / len(data))
    log.info("\nerror types injected:")
    for k, v in truth.error_type.value_counts().items():
        log.info("   %-24s %3d", k, v)

    log.info("\n" + "-" * 70)
    log.info("running detectors")
    log.info("-" * 70)
    pipeline = AnomalyPipeline(rules=DEFAULT_RULES, max_plausible=100_000)
    flagged = pipeline.run(data)
    log.info("total flagged (deduplicated): %d", len(flagged))

    log.info("\n" + "-" * 70)
    log.info("performance against ground truth")
    log.info("-" * 70)

    # Cell-level detectors only; facility-level flags (digit preference,
    # missing reports) are excluded because they have no single true cell.
    cell_level = flagged[~flagged.detector.isin(["digit_preference", "missing_report"])]

    overall = evaluate_against_truth(cell_level, truth)
    log.info("all cell-level flags:")
    for k, v in overall.items():
        log.info("   %-18s %s", k, v)

    log.info("\nprecision@k — what a district officer sees at the top of the list:")
    rows = []
    for k in (10, 25, 50, 100):
        r = evaluate_against_truth(cell_level, truth, top_n=k)
        rows.append({"k": k, "precision": r["precision"], "recall": r["recall"]})
        log.info("   top %-4d precision %.3f   recall %.3f", k, r["precision"], r["recall"])
    pd.DataFrame(rows).to_csv(OUT / "dhis2_precision_at_k.csv", index=False)

    log.info("\nrecall by injected error type:")
    key = ["orgUnit", "period", "dataElement"]
    fk = set(map(tuple, cell_level[key].astype(str).values))
    by_type = []
    for kind, grp in truth.groupby("error_type"):
        keys = set(map(tuple, grp[key].astype(str).values))
        caught = len(keys & fk)
        by_type.append({
            "error_type": kind, "injected": len(keys),
            "caught": caught, "recall": round(caught / len(keys), 3),
        })
        log.info("   %-24s %2d/%2d  recall %.2f", kind, caught, len(keys), caught / len(keys))
    pd.DataFrame(by_type).to_csv(OUT / "dhis2_recall_by_error_type.csv", index=False)

    log.info("\ncontribution by detector:")
    for det, n in flagged.detector.value_counts().items():
        log.info("   %-28s %3d", det, n)

    log.info("\n" + "-" * 70)
    log.info("sample of the top flags a reviewer would work")
    log.info("-" * 70)
    cols = ["orgUnit", "period", "dataElement", "value", "expected", "severity", "confidence", "explanation"]
    log.info("\n%s", flagged[cols].head(8).to_string(index=False, max_colwidth=58))

    flagged.to_csv(OUT / "dhis2_anomalies.csv", index=False)
    truth.to_csv(OUT / "dhis2_injected_errors.csv", index=False)
    (OUT / "dhis2_evaluation.json").write_text(json.dumps({
        "dataset": {
            "n_values": int(len(data)), "n_facilities": int(data.orgUnit.nunique()),
            "n_periods": int(data.period.nunique()),
            "n_injected_errors": int(len(truth)),
        },
        "overall": overall,
        "precision_at_k": rows,
        "recall_by_error_type": by_type,
        "detector_counts": flagged.detector.value_counts().to_dict(),
    }, indent=2))
    log.info("\nartefacts written to %s", OUT)


if __name__ == "__main__":
    main()
