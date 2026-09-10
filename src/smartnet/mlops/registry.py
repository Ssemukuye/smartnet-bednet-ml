"""Model registry with versioning, staged promotion and an audit trail.

Implements the lifecycle a governed deployment needs:

    register -> validate -> promote(staging) -> promote(production) -> rollback

Every state change is appended to an immutable audit log with a reason, an
actor and a timestamp. Nothing is promoted without a recorded justification,
because in a Ministry-of-Health setting "who approved this model, on what
evidence, and when" has to be answerable after the fact.

Storage is a JSON index plus joblib artefacts on disk. That is deliberate: it
runs anywhere, needs no service, and the same interface would sit over MLflow
or a cloud registry without changing calling code.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from smartnet import config

logger = logging.getLogger(__name__)

REGISTRY_DIR = config.MODELS_DIR / "registry"
ARTEFACT_DIR = REGISTRY_DIR / "artefacts"
INDEX_PATH = REGISTRY_DIR / "index.json"
AUDIT_PATH = REGISTRY_DIR / "audit_log.jsonl"

for _d in (REGISTRY_DIR, ARTEFACT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class Stage(str, Enum):
    """Lifecycle stages. Promotion may only move one step at a time."""

    REGISTERED = "registered"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


#: Legal transitions. Anything else raises.
ALLOWED_TRANSITIONS: dict[Stage, set[Stage]] = {
    Stage.REGISTERED: {Stage.STAGING, Stage.ARCHIVED},
    Stage.STAGING: {Stage.PRODUCTION, Stage.ARCHIVED, Stage.REGISTERED},
    Stage.PRODUCTION: {Stage.ARCHIVED},
    Stage.ARCHIVED: {Stage.STAGING},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ModelVersion:
    """One immutable registered model version."""

    name: str
    version: int
    stage: str
    created_at: str
    created_by: str
    algorithm: str
    label_scheme: str
    features: list[str]
    metrics: dict[str, float]
    training_data_hash: str
    n_training_rows: int
    validation_strategy: str
    seed: int
    python_version: str
    artefact_path: str
    notes: str = ""
    approved_by: str | None = None
    approved_at: str | None = None

    @property
    def key(self) -> str:
        return f"{self.name}:v{self.version}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """File-backed model registry with an append-only audit trail."""

    def __init__(
        self,
        index_path: Path = INDEX_PATH,
        audit_path: Path = AUDIT_PATH,
        artefact_dir: Path | None = None,
    ):
        self.index_path = Path(index_path)
        self.audit_path = Path(audit_path)
        # Artefacts live beside their index. Without this a registry pointed at
        # a temporary directory — as every test does — would still write model
        # binaries into the real one.
        self.artefact_dir = Path(artefact_dir) if artefact_dir else self.index_path.parent / "artefacts"
        self.artefact_dir.mkdir(parents=True, exist_ok=True)
        self._versions: list[ModelVersion] = []
        self._load()

    # ------------------------------------------------------------------ io
    def _load(self) -> None:
        if self.index_path.exists():
            raw = json.loads(self.index_path.read_text())
            self._versions = [ModelVersion(**v) for v in raw]
        else:
            self._versions = []

    def _save(self) -> None:
        self.index_path.write_text(
            json.dumps([v.as_dict() for v in self._versions], indent=2)
        )

    def _audit(self, action: str, subject: str, actor: str, reason: str, **extra) -> None:
        """Append one immutable audit record."""
        record = {
            "timestamp": _utc_now(),
            "action": action,
            "subject": subject,
            "actor": actor,
            "reason": reason,
            **extra,
        }
        with self.audit_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.info("AUDIT %s %s by %s — %s", action, subject, actor, reason)

    # ------------------------------------------------------------ register
    @staticmethod
    def data_hash(df: pd.DataFrame, features: list[str], label_col: str) -> str:
        """Stable content hash of the exact training matrix.

        Lets you prove later which data a model was trained on, and detect
        silently changed inputs between runs.
        """
        arr = np.ascontiguousarray(df[features + [label_col]].to_numpy(dtype=float))
        return hashlib.sha256(arr.tobytes()).hexdigest()[:16]

    def register(
        self,
        model: Any,
        name: str,
        algorithm: str,
        label_scheme: str,
        features: list[str],
        metrics: dict[str, float],
        training_frame: pd.DataFrame,
        validation_strategy: str,
        created_by: str,
        notes: str = "",
        seed: int = config.RANDOM_SEED,
    ) -> ModelVersion:
        """Persist a fitted model as a new immutable version."""
        version = 1 + max(
            [v.version for v in self._versions if v.name == name], default=0
        )
        artefact = self.artefact_dir / f"{name}_v{version}.joblib"
        joblib.dump(model, artefact, compress=3)

        mv = ModelVersion(
            name=name,
            version=version,
            stage=Stage.REGISTERED.value,
            created_at=_utc_now(),
            created_by=created_by,
            algorithm=algorithm,
            label_scheme=label_scheme,
            features=list(features),
            # Non-numeric entries (e.g. the name of the worst class) are kept
            # in notes rather than the metrics dict, which stays float-only so
            # gates and comparisons never have to type-check.
            metrics={
                k: round(float(v), 5)
                for k, v in metrics.items()
                if isinstance(v, (int, float, np.floating, np.integer))
                and not isinstance(v, bool)
            },
            training_data_hash=self.data_hash(training_frame, features, label_scheme),
            n_training_rows=int(len(training_frame)),
            validation_strategy=validation_strategy,
            seed=seed,
            python_version=platform.python_version(),
            artefact_path=self._relative(artefact),
            notes=notes,
        )
        self._versions.append(mv)
        self._save()
        self._audit(
            "register", mv.key, created_by,
            reason=notes or "initial registration",
            metrics=mv.metrics, data_hash=mv.training_data_hash,
        )
        return mv

    # ------------------------------------------------------------- promote
    def promote(
        self, name: str, version: int, to: Stage, actor: str, reason: str,
        gate: "PromotionGate | None" = None,
    ) -> ModelVersion:
        """Move a version between stages, enforcing legal transitions.

        If a ``gate`` is supplied it must pass, otherwise the promotion is
        refused and the refusal is itself audited. A blocked promotion is a
        governance event worth recording.
        """
        mv = self.get(name, version)
        current = Stage(mv.stage)
        if to not in ALLOWED_TRANSITIONS[current]:
            raise ValueError(
                f"Illegal transition {current.value} -> {to.value} for {mv.key}. "
                f"Allowed: {sorted(s.value for s in ALLOWED_TRANSITIONS[current])}"
            )

        if gate is not None:
            verdict = gate.evaluate(mv)
            if not verdict.passed:
                self._audit(
                    "promotion_blocked", mv.key, actor,
                    reason=reason, target_stage=to.value, failures=verdict.failures,
                )
                raise PermissionError(
                    f"Promotion of {mv.key} to {to.value} blocked by gate: "
                    + "; ".join(verdict.failures)
                )

        # Only one production version per model name.
        if to is Stage.PRODUCTION:
            for other in self._versions:
                if (
                    other.name == name
                    and other.version != version
                    and other.stage == Stage.PRODUCTION.value
                ):
                    other.stage = Stage.ARCHIVED.value
                    self._audit(
                        "auto_archive", f"{other.name}:v{other.version}", actor,
                        reason=f"superseded by v{version}",
                    )
            mv.approved_by = actor
            mv.approved_at = _utc_now()

        mv.stage = to.value
        self._save()
        self._audit(
            "promote", mv.key, actor, reason=reason,
            from_stage=current.value, to_stage=to.value,
        )
        return mv

    def rollback(self, name: str, actor: str, reason: str) -> ModelVersion:
        """Return the previously archived production version to production.

        Chooses the most recently approved archived version, so a bad
        promotion can be reversed without guessing.
        """
        archived = [
            v for v in self._versions
            if v.name == name and v.stage == Stage.ARCHIVED.value and v.approved_at
        ]
        if not archived:
            raise LookupError(f"No previously-approved version of {name!r} to roll back to")
        target = max(archived, key=lambda v: v.approved_at or "")

        for v in self._versions:
            if v.name == name and v.stage == Stage.PRODUCTION.value:
                v.stage = Stage.ARCHIVED.value
                self._audit("rollback_archive", v.key, actor, reason=reason)

        target.stage = Stage.PRODUCTION.value
        self._save()
        self._audit(
            "rollback", target.key, actor, reason=reason,
            restored_version=target.version,
        )
        return target

    # --------------------------------------------------------------- query
    def get(self, name: str, version: int) -> ModelVersion:
        for v in self._versions:
            if v.name == name and v.version == version:
                return v
        raise LookupError(f"{name}:v{version} not found in registry")

    def production(self, name: str) -> ModelVersion | None:
        for v in self._versions:
            if v.name == name and v.stage == Stage.PRODUCTION.value:
                return v
        return None

    def _relative(self, path: Path) -> str:
        """Store paths relative to the project root when possible.

        A registry under a temporary directory (tests) has no meaningful
        relative path, so the absolute one is kept.
        """
        try:
            return str(path.relative_to(config.PROJECT_ROOT))
        except ValueError:
            return str(path)

    def load(self, name: str, version: int | None = None) -> Any:
        """Load the fitted artefact. Defaults to the production version."""
        mv = self.get(name, version) if version else self.production(name)
        if mv is None:
            raise LookupError(f"No production version of {name!r}")
        p = Path(mv.artefact_path)
        return joblib.load(p if p.is_absolute() else config.PROJECT_ROOT / p)

    def history(self, name: str | None = None) -> pd.DataFrame:
        rows = [
            {
                "key": v.key, "stage": v.stage, "algorithm": v.algorithm,
                "label_scheme": v.label_scheme, "created_at": v.created_at,
                "created_by": v.created_by, "approved_by": v.approved_by,
                "data_hash": v.training_data_hash, "n_rows": v.n_training_rows,
                **{f"metric_{k}": val for k, val in v.metrics.items()},
            }
            for v in self._versions
            if name is None or v.name == name
        ]
        return pd.DataFrame(rows)

    def audit_trail(self, limit: int | None = None) -> pd.DataFrame:
        if not self.audit_path.exists():
            return pd.DataFrame()
        records = [json.loads(l) for l in self.audit_path.read_text().splitlines() if l.strip()]
        df = pd.DataFrame(records)
        return df.tail(limit) if limit else df


# ------------------------------------------------------------------- gates
@dataclass
class GateVerdict:
    passed: bool
    failures: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromotionGate:
    """Evidence a model must produce before it may be promoted.

    Thresholds are set on *balanced accuracy and worst-class sensitivity*, not
    overall accuracy — the whole point of this project is that accuracy hides
    minority-class failure, so the gate is written to catch exactly that.
    """

    min_balanced_accuracy: float = 0.90
    min_worst_class_sensitivity: float = 0.75
    max_accuracy_minus_balanced: float = 0.10
    required_validation_strategies: tuple[str, ...] = ("grouped_event", "grouped_day")

    def evaluate(self, mv: ModelVersion) -> GateVerdict:
        failures: list[str] = []
        m = mv.metrics

        bal = m.get("balanced_accuracy")
        if bal is None:
            failures.append("balanced_accuracy not reported")
        elif bal < self.min_balanced_accuracy:
            failures.append(
                f"balanced_accuracy {bal:.3f} < {self.min_balanced_accuracy:.2f}"
            )

        worst = m.get("worst_class_sensitivity")
        if worst is None:
            failures.append("worst_class_sensitivity not reported")
        elif worst < self.min_worst_class_sensitivity:
            failures.append(
                f"worst_class_sensitivity {worst:.3f} < {self.min_worst_class_sensitivity:.2f}"
            )

        acc = m.get("accuracy")
        if acc is not None and bal is not None:
            gap = acc - bal
            if gap > self.max_accuracy_minus_balanced:
                failures.append(
                    f"accuracy-balanced gap {gap:.3f} > {self.max_accuracy_minus_balanced:.2f} "
                    "(headline metric is masking minority-class failure)"
                )

        if mv.validation_strategy not in self.required_validation_strategies:
            failures.append(
                f"validation_strategy {mv.validation_strategy!r} not in "
                f"{self.required_validation_strategies}"
            )

        return GateVerdict(
            passed=not failures,
            failures=failures,
            checks={"balanced_accuracy": bal, "worst_class_sensitivity": worst},
        )
