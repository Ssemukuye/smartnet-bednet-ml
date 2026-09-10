"""Drift detection, performance monitoring and retraining triggers.

Three questions a deployed model has to keep answering:

1. **Has the input changed?** — feature drift, detectable without labels.
2. **Has the output changed?** — prediction drift, also label-free.
3. **Has it got worse?** — performance drift, needs ground truth.

In a health setting labels arrive late or not at all, so (1) and (2) are the
early-warning system and (3) is the confirmation. The retraining trigger
combines them with an explicit rule rather than a calendar, because retraining
on a schedule wastes effort when nothing has changed and is too slow when
something has.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from smartnet import config

logger = logging.getLogger(__name__)

MONITORING_DIR = config.REPORTS_DIR / "monitoring"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)

Severity = Literal["ok", "warn", "alert"]


# --------------------------------------------------------------- reference
@dataclass
class ReferenceProfile:
    """Statistical fingerprint of the training data, captured at registration.

    This is what "normal" means for a given model version. Drift is measured
    against this, not against the previous batch — otherwise slow drift never
    trips because each batch resembles the last.
    """

    model_key: str
    created_at: str
    n_rows: int
    feature_stats: dict[str, dict[str, float]]
    class_distribution: dict[str, float]

    @classmethod
    def build(
        cls, df: pd.DataFrame, features: list[str], label_col: str, model_key: str
    ) -> "ReferenceProfile":
        feature_stats = {}
        for f in features:
            s = df[f].astype(float)
            feature_stats[f] = {
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "p05": float(s.quantile(0.05)),
                "p50": float(s.quantile(0.50)),
                "p95": float(s.quantile(0.95)),
                "min": float(s.min()),
                "max": float(s.max()),
            }
        counts = df[label_col].value_counts(normalize=True)
        names = config.LABEL_MAPS.get(label_col, {})
        return cls(
            model_key=model_key,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            n_rows=int(len(df)),
            feature_stats=feature_stats,
            class_distribution={
                str(names.get(int(k), k)): round(float(v), 5) for k, v in counts.items()
            },
        )

    def save(self, path: Path | None = None) -> Path:
        path = path or MONITORING_DIR / f"reference_{self.model_key.replace(':', '_')}.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> "ReferenceProfile":
        return cls(**json.loads(Path(path).read_text()))


# -------------------------------------------------------------------- PSI
def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10
) -> float:
    """Population Stability Index between two samples of one feature.

    Industry convention, and the reason it is used here rather than a p-value:
    PSI measures *magnitude* of shift and does not become significant merely
    because the sample is large.

        PSI < 0.10   no meaningful shift
        0.10-0.25    moderate shift, investigate
        PSI > 0.25   major shift, act

    Bins come from the reference quantiles so the reference is the yardstick.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        return float("nan")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:  # near-constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=edges)[0] / reference.size
    cur_pct = np.histogram(current, bins=edges)[0] / current.size

    eps = 1e-6  # avoid log(0) on empty bins
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _psi_severity(psi: float) -> Severity:
    if np.isnan(psi):
        return "warn"
    if psi > 0.25:
        return "alert"
    if psi > 0.10:
        return "warn"
    return "ok"


# ------------------------------------------------------------ drift report
@dataclass
class DriftReport:
    model_key: str
    checked_at: str
    n_reference: int
    n_current: int
    feature_drift: pd.DataFrame
    prediction_drift: dict[str, float] = field(default_factory=dict)
    performance: dict[str, float] | None = None

    @property
    def n_alert(self) -> int:
        return int((self.feature_drift["severity"] == "alert").sum())

    @property
    def n_warn(self) -> int:
        return int((self.feature_drift["severity"] == "warn").sum())

    @property
    def worst_psi(self) -> float:
        return float(self.feature_drift["psi"].max())

    def summary(self) -> dict:
        return {
            "model_key": self.model_key,
            "checked_at": self.checked_at,
            "n_current": self.n_current,
            "features_alert": self.n_alert,
            "features_warn": self.n_warn,
            "worst_psi": round(self.worst_psi, 4),
            "worst_feature": str(
                self.feature_drift.sort_values("psi", ascending=False).iloc[0]["feature"]
            ),
            **({"performance": self.performance} if self.performance else {}),
        }


def detect_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    model_key: str = "unknown",
) -> DriftReport:
    """Per-feature PSI plus a Kolmogorov-Smirnov test, for every feature."""
    rows = []
    for f in features:
        ref = reference_df[f].to_numpy(dtype=float)
        cur = current_df[f].to_numpy(dtype=float)
        psi = population_stability_index(ref, cur)
        try:
            ks_stat, ks_p = stats.ks_2samp(ref, cur)
        except Exception:
            ks_stat, ks_p = float("nan"), float("nan")
        rows.append({
            "feature": f,
            "psi": psi,
            "severity": _psi_severity(psi),
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
            "ref_mean": float(np.mean(ref)),
            "cur_mean": float(np.mean(cur)),
            "mean_shift": float(np.mean(cur) - np.mean(ref)),
        })
    frame = pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
    return DriftReport(
        model_key=model_key,
        checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_reference=len(reference_df),
        n_current=len(current_df),
        feature_drift=frame,
    )


def detect_prediction_drift(
    reference_preds: np.ndarray, current_preds: np.ndarray
) -> dict[str, float]:
    """Shift in the predicted class distribution.

    Available without ground truth, which is why it is the practical
    early-warning signal in health deployments where labels lag by weeks.
    """
    ref = pd.Series(reference_preds).value_counts(normalize=True)
    cur = pd.Series(current_preds).value_counts(normalize=True)
    classes = sorted(set(ref.index) | set(cur.index))
    ref_v = np.array([ref.get(c, 0.0) for c in classes])
    cur_v = np.array([cur.get(c, 0.0) for c in classes])

    eps = 1e-6
    psi = float(np.sum((cur_v - ref_v) * np.log(np.clip(cur_v, eps, None) / np.clip(ref_v, eps, None))))
    return {
        "prediction_psi": psi,
        "severity": _psi_severity(psi),
        "max_class_shift": float(np.max(np.abs(cur_v - ref_v))),
        **{f"shift_{c}": float(cur_v[i] - ref_v[i]) for i, c in enumerate(classes)},
    }


# ------------------------------------------------------- retraining policy
@dataclass
class RetrainingPolicy:
    """Explicit, auditable rule for when a model must be retrained.

    Deliberately not a calendar. Retraining on a fixed schedule is wasted work
    when nothing has changed, and far too slow when something has.
    """

    psi_alert_features: int = 3
    worst_psi: float = 0.25
    prediction_psi: float = 0.20
    balanced_accuracy_floor: float = 0.85
    balanced_accuracy_drop: float = 0.05
    min_rows_to_judge: int = 100

    def evaluate(
        self,
        drift: DriftReport,
        baseline_balanced_accuracy: float | None = None,
        current_balanced_accuracy: float | None = None,
    ) -> dict:
        """Return a decision with the reasons that produced it."""
        reasons: list[str] = []

        if drift.n_current < self.min_rows_to_judge:
            return {
                "decision": "insufficient_data",
                "retrain": False,
                "reasons": [
                    f"only {drift.n_current} rows, need {self.min_rows_to_judge} to judge"
                ],
            }

        if drift.n_alert >= self.psi_alert_features:
            reasons.append(
                f"{drift.n_alert} features at PSI alert level "
                f"(threshold {self.psi_alert_features})"
            )
        if drift.worst_psi > self.worst_psi:
            reasons.append(
                f"worst feature PSI {drift.worst_psi:.3f} > {self.worst_psi}"
            )
        pred_psi = drift.prediction_drift.get("prediction_psi")
        if pred_psi is not None and pred_psi > self.prediction_psi:
            reasons.append(f"prediction PSI {pred_psi:.3f} > {self.prediction_psi}")

        if current_balanced_accuracy is not None:
            if current_balanced_accuracy < self.balanced_accuracy_floor:
                reasons.append(
                    f"balanced accuracy {current_balanced_accuracy:.3f} below floor "
                    f"{self.balanced_accuracy_floor}"
                )
            if baseline_balanced_accuracy is not None:
                drop = baseline_balanced_accuracy - current_balanced_accuracy
                if drop > self.balanced_accuracy_drop:
                    reasons.append(
                        f"balanced accuracy dropped {drop:.3f} from baseline "
                        f"(threshold {self.balanced_accuracy_drop})"
                    )

        if not reasons:
            return {"decision": "no_action", "retrain": False, "reasons": ["all monitors within threshold"]}

        # Confirmed performance loss escalates from investigate to retrain.
        performance_confirmed = current_balanced_accuracy is not None and any(
            "balanced accuracy" in r for r in reasons
        )
        if performance_confirmed:
            return {"decision": "retrain", "retrain": True, "reasons": reasons}
        return {"decision": "investigate", "retrain": False, "reasons": reasons}


def save_drift_report(drift: DriftReport, decision: dict) -> Path:
    """Persist the full report plus the decision it produced."""
    stamp = drift.checked_at.replace(":", "").replace("-", "")
    base = MONITORING_DIR / f"drift_{drift.model_key.replace(':', '_')}_{stamp}"
    drift.feature_drift.to_csv(f"{base}_features.csv", index=False)
    Path(f"{base}_summary.json").write_text(
        json.dumps(
            {**drift.summary(), "prediction_drift": drift.prediction_drift, "decision": decision},
            indent=2,
        )
    )
    return Path(f"{base}_summary.json")
