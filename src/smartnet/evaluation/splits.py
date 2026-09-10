"""Cross-validation strategies, including the leakage demonstration.

The central methodological point of this project lives here. Each labelled
motion is recorded as a contiguous run of 1-second epochs, and every row
carries features describing the 10 epochs before and after it. Two adjacent
rows from the same event therefore share most of their input signal. Splitting
those rows at random puts near-duplicate windows on both sides of the split and
inflates the score.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from smartnet import config


def make_splitter(
    df: pd.DataFrame,
    y: np.ndarray,
    strategy: str,
    n_splits: int = config.N_SPLITS,
    seed: int = config.RANDOM_SEED,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` for the named strategy.

    Parameters
    ----------
    strategy
        One of ``random_row``, ``grouped_event`` or ``grouped_day``.
    """
    known = {s.name: s for s in config.SPLIT_STRATEGIES}
    if strategy not in known:
        raise KeyError(f"Unknown split strategy {strategy!r}; expected one of {sorted(known)}")

    cfg = known[strategy]

    if cfg.group_col is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from cv.split(np.zeros(len(y)), y)
        return

    groups = df[cfg.group_col].to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(
            f"Strategy {strategy!r} has only {n_groups} groups but {n_splits} folds requested."
        )
    cv = GroupKFold(n_splits=n_splits)
    yield from cv.split(np.zeros(len(y)), y, groups=groups)


def split_diagnostics(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> dict[str, float | int]:
    """Quantify how much structure a split shares between train and test.

    ``shared_events`` above zero means the split cannot estimate performance on
    unseen motion events.
    """
    tr, te = df.iloc[train_idx], df.iloc[test_idx]
    shared_events = len(set(tr["event_id"]) & set(te["event_id"]))
    shared_days = len(set(tr["session_date"]) & set(te["session_date"]))
    return {
        "n_train": len(tr),
        "n_test": len(te),
        "shared_events": shared_events,
        "shared_days": shared_days,
        "pct_test_epochs_from_seen_event": round(
            100 * te["event_id"].isin(set(tr["event_id"])).mean(), 2
        ),
    }


def summarise_strategies(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Table of leakage diagnostics for every strategy, averaged over folds."""
    y = df[label_col].to_numpy()
    rows = []
    for cfg in config.SPLIT_STRATEGIES:
        diags = [
            split_diagnostics(df, tr, te)
            for tr, te in make_splitter(df, y, cfg.name)
        ]
        mean = pd.DataFrame(diags).mean(numeric_only=True)
        rows.append(
            {
                "strategy": cfg.name,
                "grouping": cfg.group_col or "none (row-level)",
                "mean_train_n": int(mean["n_train"]),
                "mean_test_n": int(mean["n_test"]),
                "shared_events": round(float(mean["shared_events"]), 1),
                "pct_test_from_seen_event": round(
                    float(mean["pct_test_epochs_from_seen_event"]), 1
                ),
                "description": cfg.description,
            }
        )
    return pd.DataFrame(rows)
