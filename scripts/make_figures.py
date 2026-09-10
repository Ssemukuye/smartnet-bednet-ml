"""Regenerate every figure in reports/figures from persisted results."""
from __future__ import annotations

import logging
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from smartnet import config
from smartnet.data import loader
from smartnet.visualization import plots as P

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("figures")
R = config.RESULTS_DIR


def main() -> None:
    df = loader.load_analysis_frame()
    events = loader.event_summary(df)

    P._save(P.plot_class_balance(df, "motion_5cat"), "01_class_balance")
    P._save(P.plot_event_duration(events), "02_event_duration")
    P._save(P.plot_temporal_coverage(df, "motion_5cat"), "03_temporal_coverage")
    P._save(P.plot_signal_examples(df, "motion_5cat"), "04_signal_examples")

    diag = pd.read_csv(R / "split_diagnostics.csv")
    P._save(P.plot_leakage_diagnostic(diag), "05_leakage_diagnostic")

    for lab in ("motion_4cat", "motion_5cat"):
        res = pd.read_csv(R / f"results_{lab}_all.csv")
        P._save(P.plot_model_comparison(res), f"06_model_comparison_{lab}")
        P._save(P.plot_accuracy_vs_balanced(res), f"07_accuracy_vs_balanced_{lab}")

        cm = pd.read_csv(R / f"confusion_{lab}_grouped_event.csv", index_col=0)
        P._save(
            P.plot_confusion(cm, f"Random forest, grouped-event CV — {lab}"),
            f"08_confusion_{lab}",
        )
        pc = pd.read_csv(R / f"per_class_{lab}_grouped_event.csv")
        P._save(
            P.plot_per_class(pc, f"Per-class performance — {lab} (grouped-event CV)"),
            f"09_per_class_{lab}",
        )

    imp = pd.read_csv(R / "permutation_importance_motion_5cat.csv")
    P._save(P.plot_feature_importance(imp), "10_feature_importance")

    n = len(list(config.FIGURES_DIR.glob("*.png")))
    log.info("Wrote %d figures to %s", n, config.FIGURES_DIR)


if __name__ == "__main__":
    main()
