"""Model definitions.

The progression runs from a trivial baseline to an interpretable linear model,
a single tree, and finally two ensembles. Nothing more exotic is included: with
1,340 labelled epochs and 664 events, deep sequence models cannot be trained
credibly, and saying so is more useful than demonstrating otherwise.
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from smartnet import config


def build_model(name: str, seed: int = config.RANDOM_SEED) -> Pipeline:
    """Return an unfitted pipeline for the named model.

    Scaling is included for the linear model only; tree ensembles are invariant
    to monotone feature transforms. Wrapping everything in a ``Pipeline`` keeps
    all preprocessing inside the cross-validation loop, which is itself a
    leakage guard.
    """
    if name == "majority_baseline":
        est = DummyClassifier(strategy="most_frequent")
        return Pipeline([("model", est)])

    if name == "logistic_regression":
        est = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("model", est)])

    if name == "decision_tree":
        est = DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=seed,
        )
        return Pipeline([("model", est)])

    if name == "random_forest":
        # Mirrors the published configuration: 1000 trees, sqrt(p) features.
        est = RandomForestClassifier(
            n_estimators=1000,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        return Pipeline([("model", est)])

    if name == "gradient_boosting":
        # Histogram-based boosting: same family as XGBoost/LightGBM, ships with
        # scikit-learn, and trains in seconds on a dataset this size.
        est = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=3,
            early_stopping=False,
            random_state=seed,
        )
        return Pipeline([("model", est)])

    raise KeyError(f"Unknown model {name!r}")


MODEL_NOTES: dict[str, str] = {
    "majority_baseline": "Always predicts the most frequent class. Floor for every other number.",
    "logistic_regression": "Interpretable linear baseline; coefficients are directly readable.",
    "decision_tree": "Single depth-limited tree; shows how far one set of thresholds gets you.",
    "random_forest": "Matches the published configuration (1000 trees, sqrt(p) candidates per split).",
    "gradient_boosting": "Sequential boosting; tests whether ensembling beyond bagging adds anything.",
}
