"""Central configuration for the SMARTNET bed-net ML project.

All paths, column groups, label schemes and modelling constants live here so
that notebooks, scripts and tests share one source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "model_results"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

#: Default name of the tagged Stata extract supplied by the study team.
RAW_STATA_FILENAME = "tagged train data.dta"

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
TIMESTAMP_COL = "date_str"

LABEL_COLS = ["motion_something", "motion_3cat", "motion_4cat", "motion_5cat"]

#: Value labels exactly as stored in the Stata value-label sets.
LABEL_MAPS: dict[str, dict[int, str]] = {
    "motion_something": {0: "Nothing", 1: "Something"},
    "motion_3cat": {0: "Nothing", 1: "Down event", 2: "Put up"},
    "motion_4cat": {0: "Nothing", 1: "Put down", 2: "Put up", 3: "Enter or exit"},
    "motion_5cat": {0: "Nothing", 1: "Put down", 2: "Put up", 3: "Enter", 4: "Exit"},
}

#: Features describing the epoch being classified.
CURRENT_EPOCH_FEATURES = ["sum_vectormagnitudes", "stdvz"]

#: Lagged context: the 10 epochs preceding the labelled epoch.
BACK_LAG_FEATURES = [f"back_{stat}{i}" for i in range(1, 11) for stat in ("stdvz", "sumvector")]

#: Lead context: the 10 epochs following the labelled epoch.
FORWARD_LAG_FEATURES = [
    f"forward_{stat}{i}" for i in range(1, 11) for stat in ("stdvz", "sumvector")
]

#: Rolling aggregates already computed over the 10-epoch back/forward windows.
WINDOW_AGGREGATE_FEATURES = [
    "sumover10vector_back",
    "meanzover10vector_back",
    "stdvzover10vector_back",
    "sumover10vector_forward",
    "meanzover10vector_forward",
    "stdvzover10vector_forward",
]

ALL_FEATURES = (
    CURRENT_EPOCH_FEATURES
    + BACK_LAG_FEATURES
    + FORWARD_LAG_FEATURES
    + WINDOW_AGGREGATE_FEATURES
)

#: Named feature blocks, used for ablation experiments.
FEATURE_BLOCKS: dict[str, list[str]] = {
    "current_only": CURRENT_EPOCH_FEATURES,
    "current_plus_aggregates": CURRENT_EPOCH_FEATURES + WINDOW_AGGREGATE_FEATURES,
    "backward_only": CURRENT_EPOCH_FEATURES + BACK_LAG_FEATURES,
    "all": ALL_FEATURES,
}

# --------------------------------------------------------------------------
# Data-quality thresholds
# --------------------------------------------------------------------------
#: Maximum gap (seconds) between consecutive epochs still considered the same
#: contiguous recording run. Derived empirically: 676/1339 observed gaps are
#: exactly 1s and every 1s run is label-pure.
EVENT_GAP_SECONDS = 1

#: Physically plausible bounds, used for range validation rather than for
#: silently clipping values.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "sum_vectormagnitudes": (-5.0, 50.0),
    "stdvz": (0.0, 10.0),
}

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
RANDOM_SEED = 42
N_SPLITS = 5

#: Label scheme used for headline results.
PRIMARY_LABEL = "motion_4cat"


@dataclass(frozen=True)
class SplitConfig:
    """Describes one validation strategy."""

    name: str
    group_col: str | None
    description: str


SPLIT_STRATEGIES: list[SplitConfig] = [
    SplitConfig(
        name="random_row",
        group_col=None,
        description=(
            "Naive random split of individual epochs. Reproduces the published "
            "approach. Leaks because overlapping context windows from the same "
            "motion event appear in both train and test."
        ),
    ),
    SplitConfig(
        name="grouped_event",
        group_col="event_id",
        description=(
            "GroupKFold on contiguous 1-second motion events. No epoch from an "
            "event appears in both folds. Estimates performance on unseen events."
        ),
    ),
    SplitConfig(
        name="grouped_day",
        group_col="session_date",
        description=(
            "GroupKFold on calendar recording day. Strictest setting; estimates "
            "performance on an unseen monitoring session."
        ),
    ),
]


@dataclass
class ExperimentConfig:
    """Runtime configuration for a full experiment sweep."""

    label_col: str = PRIMARY_LABEL
    feature_block: str = "all"
    n_splits: int = N_SPLITS
    seed: int = RANDOM_SEED
    models: list[str] = field(
        default_factory=lambda: [
            "majority_baseline",
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "gradient_boosting",
        ]
    )

    @property
    def features(self) -> list[str]:
        return FEATURE_BLOCKS[self.feature_block]
