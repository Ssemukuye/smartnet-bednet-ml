"""Generate a schema-identical synthetic extract.

Lets anyone run the full pipeline without access to the study data. Output is
clearly named and is not study data. Class-conditional means are invented, so
results from this file describe the generator, not bed nets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from smartnet import config

N_DAYS = 20
EVENTS_PER_DAY = 34
SEED = 7


def make_frame(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Class-conditional signal levels, loosely ordered so classes are separable
    # but overlapping — enough to exercise the pipeline, not to imitate results.
    level = {0: 0.05, 1: 1.6, 2: 1.9, 3: 1.2, 4: 1.25}
    spread = {0: 0.03, 1: 0.9, 2: 1.0, 3: 0.8, 4: 0.8}

    rows = []
    base = pd.Timestamp("2021-09-10 00:00:00")
    # Events are placed on a 60-second grid sampled without replacement, so no
    # two events can ever merge into one contiguous run. Event purity is a
    # precondition of the grouped-CV design and must hold for synthetic data too.
    daytime_slots = np.array([h * 60 + m for h in (9, 10, 12, 14, 15, 16) for m in range(60)])
    all_slots = np.arange(24 * 60)

    for day in range(N_DAYS):
        classes = rng.choice([0, 1, 2, 3, 4], size=EVENTS_PER_DAY, p=[0.39, 0.17, 0.37, 0.035, 0.035])
        n_motion = int((classes != 0).sum())
        n_nothing = EVENTS_PER_DAY - n_motion
        motion_slots = rng.choice(daytime_slots, size=n_motion, replace=False)
        remaining = np.setdiff1d(all_slots, motion_slots)
        nothing_slots = rng.choice(remaining, size=n_nothing, replace=False)

        slots = {"motion": list(motion_slots), "nothing": list(nothing_slots)}
        for cls5 in classes:
            cls5 = int(cls5)
            slot = slots["nothing"].pop() if cls5 == 0 else slots["motion"].pop()
            t0 = base + pd.Timedelta(days=day, minutes=int(slot))
            n_epochs = 1 if cls5 == 0 else int(rng.integers(2, 12))

            for sec in range(n_epochs):
                mu, sd = level[cls5], spread[cls5]
                row: dict[str, object] = {}
                row["sum_vectormagnitudes"] = float(abs(rng.normal(mu, sd)))
                row["stdvz"] = float(abs(rng.normal(mu * 0.15, sd * 0.15)))
                for i in range(1, 11):
                    decay = 1.0 - 0.06 * i
                    row[f"back_sumvector{i}"] = float(abs(rng.normal(mu * decay, sd)))
                    row[f"back_stdvz{i}"] = float(abs(rng.normal(mu * 0.15 * decay, sd * 0.15)))
                    row[f"forward_sumvector{i}"] = float(abs(rng.normal(mu * decay, sd)))
                    row[f"forward_stdvz{i}"] = float(abs(rng.normal(mu * 0.15 * decay, sd * 0.15)))
                for side in ("back", "forward"):
                    row[f"sumover10vector_{side}"] = float(
                        sum(row[f"{side}_sumvector{i}"] for i in range(1, 11))
                    )
                    row[f"meanzover10vector_{side}"] = float(rng.normal(mu * 0.4, sd * 0.3))
                    row[f"stdvzover10vector_{side}"] = float(
                        np.std([row[f"{side}_stdvz{i}"] for i in range(1, 11)])
                    )
                row[config.TIMESTAMP_COL] = str(t0 + pd.Timedelta(seconds=sec))
                row["motion_5cat"] = float(cls5)
                row["motion_4cat"] = float(3 if cls5 in (3, 4) else cls5)
                row["motion_3cat"] = float(0 if cls5 == 0 else (2 if cls5 == 2 else 1))
                row["motion_something"] = float(0 if cls5 == 0 else 1)
                rows.append(row)

    df = pd.DataFrame(rows)
    # Guarantee unique timestamps, as in the real extract.
    df = df.drop_duplicates(subset=[config.TIMESTAMP_COL]).reset_index(drop=True)
    return df[
        config.LABEL_COLS
        + [config.TIMESTAMP_COL]
        + [c for c in config.ALL_FEATURES]
    ]


if __name__ == "__main__":
    out = config.RAW_DIR / "synthetic_sample.dta"
    frame = make_frame()
    frame.to_stata(out, write_index=False, version=118)
    print(f"Wrote {len(frame)} synthetic rows x {frame.shape[1]} columns -> {out}")
    print("NOTE: synthetic data. Any results derived from it are not study results.")
