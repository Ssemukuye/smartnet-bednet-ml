"""Merge the chapter notebooks into one continuous analysis notebook.

Notebooks 01–07 are themed chapters that each re-import and reload the data so
they can be run independently. Merging them naively would repeat that setup
seven times. This script:

  * emits a single title, context block and table of contents
  * hoists all imports into one consolidated setup cell
  * strips the repeated per-chapter setup and reload cells
  * renumbers the chapters as parts of one document
  * appends the results summary and limitations from notebook 00

Output: notebooks/SMARTNET_full_analysis.ipynb
"""
from __future__ import annotations

import json
import re
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
OUT = NB_DIR / "SMARTNET_full_analysis.ipynb"

CHAPTERS = [
    ("01_data_understanding.ipynb", "Data and study understanding"),
    ("02_data_quality.ipynb", "Data quality"),
    ("03_exploratory_analysis.ipynb", "Exploratory analysis"),
    ("04_feature_engineering.ipynb", "Feature engineering and selection"),
    ("05_baseline_models.ipynb", "Validation design and baseline models"),
    ("06_model_comparison.ipynb", "Model comparison and error analysis"),
    ("07_model_interpretation.ipynb", "Interpretation"),
]

SETUP = """# ---------------------------------------------------------------------------
# Setup — run once. Every later cell depends on this.
# ---------------------------------------------------------------------------
import sys, warnings
sys.path.insert(0, '../src')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.width', 200)

from smartnet import config
from smartnet.data import loader, validation
from smartnet.evaluation.splits import summarise_strategies, make_splitter, split_diagnostics
from smartnet.evaluation import metrics as M
from smartnet.models.experiment import run_cv, run_sweep, leakage_table
from smartnet.models.interpret import block_importance, logistic_coefficients
from smartnet.visualization import plots as P

R = config.RESULTS_DIR

# Loaded once and reused throughout.
df = loader.load_analysis_frame()
events = loader.event_summary(df, 'motion_5cat')

print(f'{df.shape[0]:,} labelled epochs x {df.shape[1]} columns')
print(f'{df.event_id.nunique()} motion events across {df.session_date.nunique()} recording days')
print(f'{df.timestamp.min():%Y-%m-%d} to {df.timestamp.max():%Y-%m-%d}')"""

TITLE = """# Bed-Net Use Behaviour Classification from Accelerometer Data
## Complete analysis

Classifying how insecticide-treated bed nets are actually used, from a single accelerometer fixed to the net.

This notebook is the full analysis in one document: data understanding, quality control, exploratory analysis, feature selection, validation design, model comparison, error analysis and interpretation. It is the merged form of the seven chapter notebooks in this repository.

**Runtime:** roughly 5–8 minutes. The model sweeps in Parts 5 and 6 and the permutation importance in Part 7 are the slow cells.

---

### Headline results

| Finding | Value |
|---|---|
| Test epochs sharing a motion event with training, under a random split | **59.6%** |
| Leakage inflation of accuracy | **0.3–1.0 pp** — hypothesis not supported |
| Five-category accuracy / balanced accuracy | **0.947 / 0.787** |
| Enter / Exit sensitivity | **0.34 / 0.61** |
| Four-category accuracy / balanced accuracy | **0.978 / 0.965** |
| Combined enter-or-exit sensitivity | **0.934** |
| Best feature block | **8 aggregates**, beating all 48 |
| Balanced accuracy without forward context | **0.593** |

The five-category model reaches 94.7% accuracy, and that number is misleading. Balanced accuracy is 78.7%, because entering and exiting a net — the behaviours that matter most for malaria exposure — are classified at 0.34 and 0.61 sensitivity and are confused with each other. This notebook is built around measuring and explaining that gap.

---

### Contents

1. [Data and study understanding](#part-1)
2. [Data quality](#part-2)
3. [Exploratory analysis](#part-3)
4. [Feature engineering and selection](#part-4)
5. [Validation design and baseline models](#part-5)
6. [Model comparison and error analysis](#part-6)
7. [Interpretation](#part-7)
8. [Summary and limitations](#part-8)

---

### Scientific context

The methodology follows Koudou et al. (2022), *Evaluation of an accelerometer-based monitor for detecting bed net use and human entry/exit using a machine learning algorithm*, Malaria Journal 21:85. That study, run under controlled conditions in Liverpool in 2018, reported 96.2% accuracy for three categories, 94.8% for four and 82.9% for five, with entry/exit sensitivity of 0.681 and 0.632.

**This analysis uses a different, later tagged dataset** (September–October 2021; 1,340 epochs; a richer 48-feature representation). It is not a re-analysis of the published data and the numbers are not directly comparable. What it tests is whether the published *limitation* reappears independently. It does.

### Data governance

The tagged extract is human-subjects research data belonging to the study investigators. It is excluded from version control. This notebook runs on synthetic data of identical schema if the real extract is absent — see `data/README.md` and run `python scripts/make_synthetic_sample.py`."""

SUMMARY = """<a id="part-8"></a>
# Part 8 — Summary and limitations

## What this analysis established

| Finding | Value |
|---|---|
| Test epochs sharing an event with training, random split | 59.6% |
| Leakage inflation of accuracy | 0.3–1.0 pp *(hypothesis not supported)* |
| 5-category accuracy / balanced accuracy | 0.947 / 0.787 |
| Enter / Exit sensitivity | 0.34 / 0.61 |
| 4-category accuracy / balanced accuracy | 0.978 / 0.965 |
| Combined enter-or-exit sensitivity | 0.934 |
| Best feature block | 8 aggregates, beating all 48 |
| Balanced accuracy without forward context | 0.593 |

## Recommendation

**Use the four-category model.** It detects when a net goes up or down and counts net crossings, which is enough to measure net-use timing and duration. It does not support directional inference — distinguishing entering from exiting — and must not be used for it.

A calibrated deployment should be allowed to **abstain**: report *net crossed* with high confidence and decline to guess direction. Matching the output to the model's real capability is more useful than forcing a five-way decision it cannot support.

## Limitations

1. **No participant identifier.** Grouping is by event and recording day, not by person, so **performance on a new participant cannot be estimated** — the question that matters most for deployment. This is the single biggest limitation.

2. **No demographics**, so the adult/child fairness comparison reported in the published study (87.8% vs 70.0%) cannot be reproduced. Children are a priority malaria population, which makes this a material gap.

3. **Entry/exit direction is not recoverable** from this sensor placement and feature set at these sample sizes.

4. **Small minority classes.** 47 *Enter* and 44 *Exit* epochs means roughly 9 test examples per fold; those per-class figures carry wide uncertainty.

5. **Staged conditions.** Motions were recorded in daytime sessions, not during natural overnight net use.

6. **Single sensor placement.** One accelerometer on one side panel.

7. **`Nothing` classifies perfectly (517/517), which deserves scrutiny.** The likely explanation is benign — near-zero movement is trivially separable — but perfect separation always warrants a check, and with a participant identifier I would test whether it reflects a recording-context artefact.

Limitations 1 and 2 are blocked by the extract, not by method. **A participant-linked extract from the study PI would be the single biggest upgrade to this work.**

## Next steps

1. Obtain participant identifiers; re-run with participant-level grouping and add the adult/child fairness analysis.
2. Test a second sensor or placement to establish whether entry/exit direction is recoverable at all. This is a data-collection question — no modelling will extract information the signal does not contain.
3. Sequence models over raw 10 Hz signal, once materially more labelled events exist.
4. Reduce to the ~8 features that carry the signal, for low-power on-device inference.
5. Prospective validation in real household conditions, where any programmatic claim has to be earned.

---

*Research and portfolio project. Not a medical device; not validated for clinical or programmatic use.*"""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def is_setup_cell(src: str) -> bool:
    """Chapter setup cells: imports plus a reload of df/events."""
    return "sys.path.insert" in src or src.strip().startswith("import sys")


#: Lines the consolidated setup cell already covers.
_REDUNDANT = re.compile(
    r"""^(
        import\s
      | from\s(smartnet|sklearn)
      | sys\.path\.insert
      | warnings\.filterwarnings
      | pd\.set_option
      | (df|events|R)\s*=\s*(loader\.|config\.)
    )""",
    re.VERBOSE,
)


def strip_reload(src: str) -> str:
    """Drop lines that re-import or re-load state the merged setup already holds."""
    keep = [ln for ln in src.splitlines() if not _REDUNDANT.match(ln.strip())]
    return "\n".join(keep).strip()


def demote_heading(text: str) -> str:
    """Chapter H1 titles become part-level H1s handled by the injected banner."""
    lines = text.splitlines()
    out = []
    for line in lines:
        if line.startswith("# ") and re.match(r"^# \d{2} — ", line):
            continue  # replaced by the part banner
        out.append(line)
    return "\n".join(out).strip()


def main() -> None:
    cells: list[dict] = [md(TITLE), code(SETUP)]
    dropped = 0

    for idx, (fname, title) in enumerate(CHAPTERS, start=1):
        nb = json.loads((NB_DIR / fname).read_text())
        cells.append(md(f'<a id="part-{idx}"></a>\n# Part {idx} — {title}'))

        for cell in nb["cells"]:
            src = "".join(cell["source"])

            if cell["cell_type"] == "code":
                if is_setup_cell(src):
                    remainder = strip_reload(src)
                    if not remainder:
                        dropped += 1
                        continue
                    cells.append(code(remainder))
                else:
                    cells.append(code(src))
            else:
                text = demote_heading(src)
                if text:
                    cells.append(md(text))
                else:
                    dropped += 1

    cells.append(md(SUMMARY))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "title": "Bed-Net Use Behaviour Classification — Complete Analysis",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1))

    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"wrote {OUT.relative_to(OUT.parents[1])}")
    print(f"  {len(cells)} cells ({n_code} code, {len(cells)-n_code} markdown)")
    print(f"  merged {len(CHAPTERS)} chapters, dropped {dropped} duplicated setup cells")


if __name__ == "__main__":
    main()
