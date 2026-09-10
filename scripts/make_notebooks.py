"""Generate the analysis notebooks.

Notebooks are generated from this single source so that narrative and code stay
in sync with the package. Each notebook imports from ``smartnet`` rather than
redefining logic, so a change in the library propagates everywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NB_DIR.mkdir(exist_ok=True)

HEADER = """import sys, warnings
sys.path.insert(0, '../src')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.width', 200)
from smartnet import config"""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


NOTEBOOKS: dict[str, list[dict]] = {}

# ---------------------------------------------------------------- 01
NOTEBOOKS["01_data_understanding.ipynb"] = [
    md("""# 01 — Data and study understanding

**Question this notebook answers:** what is in this dataset, what does each variable mean, and what analysis does it make possible?

The scientific context comes from Koudou et al. (2022), *Malaria Journal* 21:85, which established the accelerometer methodology. This extract is a separate, later tagged dataset — the numbers here are not a re-analysis of that paper."""),
    code(HEADER + "\nfrom smartnet.data import loader\n\ndf = loader.load_analysis_frame()\nprint(df.shape)\ndf.head(3)"),
    md("""## Label schemes

Four nested schemes. Coarser schemes collapse classes of finer ones, so they answer progressively easier questions."""),
    code("""for col, names in config.LABEL_MAPS.items():
    vc = df[col].value_counts().sort_index()
    print(f'--- {col}')
    for k, v in vc.items():
        print(f'   {names[int(k)]:<16} {v:5d}  ({100*v/len(df):5.1f}%)')"""),
    md("""The imbalance is the central design constraint: **Enter (47) and Exit (44) are the rarest classes and the most operationally interesting ones.** Any evaluation reported as overall accuracy will be dominated by Nothing and Put up."""),
    md("""## Feature structure

Every row carries a 21-second context window: the tagged second, the 10 preceding seconds, the 10 following seconds, plus aggregates over those windows."""),
    code("""for block, cols in config.FEATURE_BLOCKS.items():
    print(f'{block:<26} {len(cols):3d} features')
print()
print('Current epoch :', config.CURRENT_EPOCH_FEATURES)
print('Aggregates    :', config.WINDOW_AGGREGATE_FEATURES)"""),
    md("""## Temporal structure

This determines the entire experimental design, so it is worth establishing carefully."""),
    code("""d = df.sort_values('timestamp')
gaps = d['gap_seconds'].dropna()
print('Period      :', df.timestamp.min(), '->', df.timestamp.max())
print('Recording days:', df.session_date.nunique())
print()
print('Gap between consecutive epochs (seconds):')
print(gaps.value_counts().head(6).to_string())
print(f'\\n{(gaps==1).sum()} of {len(gaps)} gaps are exactly 1 second')"""),
    code("""events = loader.event_summary(df, 'motion_5cat')
print(f'{len(events)} contiguous events')
print(events.n_epochs.describe().to_string())
events.head()"""),
    md("""**Key finding.** Labelled motions are recorded as contiguous runs of 1-second epochs, up to 16 seconds long. Every run is label-pure — no event spans two classes. This gives a legitimate grouping unit for cross-validation, which notebook 05 relies on."""),
    code("loader.assert_events_are_label_pure(df)\nprint('All events label-pure.')"),
    md("""## What this dataset does *not* contain

There is **no participant, device or bed-net identifier**. Consequences:

1. Grouping can be done by event and by recording day, but **not by person**.
2. Performance on a *new participant* — the deployment-relevant question — cannot be estimated.
3. The adult/child subgroup comparison reported in the published study **cannot be reproduced**.

These are stated as limitations throughout rather than worked around."""),
    code("""missing_ids = [c for c in ['participant_id','device_id','net_id','subject'] if c in df.columns]
print('Identifier columns present:', missing_ids or 'none')"""),
]

# ---------------------------------------------------------------- 02
NOTEBOOKS["02_data_quality.ipynb"] = [
    md("""# 02 — Data quality

**Question:** is this extract fit for modelling, and what would block it?

Eight automated checks run on every pipeline execution. Each returns a severity rather than raising, so one pass surfaces every problem."""),
    code(HEADER + "\nfrom smartnet.data import loader, validation\n\ndf = loader.load_analysis_frame()\nreport = validation.run_all(df)\nreport"),
    code("print('Blocking failure:', validation.has_blocking_failure(report))"),
    md("""## Missingness"""),
    code("miss = loader.describe_missingness(df)\nprint(miss.head(10).to_string(index=False))"),
    md("""## Label consistency

The nested schemes must agree. Every row that is *Nothing* in the binary scheme must be *Nothing* in all schemes, and 5-category Enter/Exit must collapse into 4-category *Enter or exit*. This is a real integrity test of the tagging process, not a formality."""),
    code("""print(pd.crosstab(df.motion_5cat.map(config.LABEL_MAPS['motion_5cat']),
                  df.motion_4cat.map(config.LABEL_MAPS['motion_4cat'])).to_string())"""),
    md("""## Class support

The rarest class carries 44 epochs. Under 5-fold cross-validation that is roughly 9 test examples per fold, so per-class estimates for Enter and Exit carry wide uncertainty. Flagged, not fixed — no amount of resampling creates information that is not there."""),
    code("""for col in ['motion_4cat','motion_5cat']:
    vc = df[col].value_counts()
    names = config.LABEL_MAPS[col]
    print(f'{col}: rarest = {names[int(vc.idxmin())]} at {vc.min()} epochs '
          f'(~{vc.min()//config.N_SPLITS} per CV fold)')"""),
    md("""## Deliberate non-decisions

Three things this pipeline does **not** do, each on purpose:

- **No outlier removal.** Large accelerations are the signal, not noise. Values are range-checked and reported, never clipped.
- **No imputation.** There is nothing missing.
- **No resampling of minority classes.** Class weighting is used inside the models instead, so reported support stays honest."""),
]

# ---------------------------------------------------------------- 03
NOTEBOOKS["03_exploratory_analysis.ipynb"] = [
    md("""# 03 — Exploratory analysis

**Question:** what does the movement signal actually look like for each behaviour, and is there visible separation before any model is fitted?"""),
    code(HEADER + "\nfrom smartnet.data import loader\nfrom smartnet.visualization import plots as P\n\ndf = loader.load_analysis_frame()\nevents = loader.event_summary(df, 'motion_5cat')"),
    code("fig = P.plot_class_balance(df, 'motion_5cat'); plt.show()"),
    code("fig = P.plot_event_duration(events); plt.show()"),
    md("""Events are short. Most are a handful of seconds, and adjacent epochs within one event share ±10 seconds of context — so they are near-duplicates of each other. Notebook 05 quantifies what that does to a random train/test split."""),
    code("fig = P.plot_temporal_coverage(df, 'motion_5cat'); plt.show()"),
    md("""Motions were staged during daytime sessions; *Nothing* was sampled across all 24 hours. This is a property of the data-collection protocol, and it means hour-of-day would be a leaky feature — it is deliberately excluded from the model."""),
    code("fig = P.plot_signal_examples(df, 'motion_5cat'); plt.show()"),
    md("""## Feature distributions by class"""),
    code("""names = config.LABEL_MAPS['motion_5cat']
key = ['sum_vectormagnitudes','stdvz','meanzover10vector_back','sumover10vector_forward']
summary = df.groupby(df.motion_5cat.map(names))[key].agg(['mean','std']).round(3)
summary"""),
    code("""fig, axes = plt.subplots(2, 2, figsize=(11, 6))
for ax, col in zip(axes.ravel(), key):
    for i, (name, sub) in enumerate(df.groupby(df.motion_5cat.map(names))):
        ax.hist(sub[col], bins=30, alpha=0.55, label=name, color=P.PALETTE[i % len(P.PALETTE)])
    ax.set_title(col, fontsize=9); ax.set_yscale('log')
axes[0,0].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.show()"""),
    md("""*Nothing* separates cleanly on every feature — it is near-zero movement. The four motion classes overlap heavily with one another, which previews the result: detecting *that* something happened is easy; identifying *which* motion is not."""),
    md("""## Correlation structure"""),
    code("""corr = df[config.WINDOW_AGGREGATE_FEATURES + config.CURRENT_EPOCH_FEATURES].corr()
fig, ax = plt.subplots(figsize=(6.5,5.2))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha='right', fontsize=7)
ax.set_yticks(range(len(corr)), corr.columns, fontsize=7); ax.grid(False)
fig.colorbar(im, shrink=0.8); plt.title('Aggregate feature correlations'); plt.show()"""),
    md("""The lag features are strongly autocorrelated by construction. This is why permutation importance (notebook 07) is read at the level of feature *blocks* rather than trusting individual rankings among correlated columns."""),
]

# ---------------------------------------------------------------- 04
NOTEBOOKS["04_feature_engineering.ipynb"] = [
    md("""# 04 — Feature engineering and selection

**Question:** which parts of the 21-second context window actually carry signal?

The features were engineered by the study team before this extract: per-second summary statistics, ±10-second lags and leads, and aggregates over those windows. Rather than invent more, this notebook tests which existing blocks matter — an ablation, not an expansion."""),
    code(HEADER + "\nfrom smartnet.data import loader\nfrom smartnet.models.experiment import run_cv\n\ndf = loader.load_analysis_frame()\nfor b, c in config.FEATURE_BLOCKS.items():\n    print(f'{b:<26} {len(c):3d}')"),
    md("""## Ablation

Each block is evaluated under grouped-event CV with the same random forest. If the 40 individual lag columns add nothing over the 6 aggregates, the feature set can be cut by 85% with no cost — which matters for on-device inference."""),
    code("""rows = []
for block in config.FEATURE_BLOCKS:
    r = run_cv(df, 'motion_5cat', 'random_forest', 'grouped_event',
               features=config.FEATURE_BLOCKS[block])
    rows.append({'block': block, 'n_features': r['n_features'],
                 'accuracy': round(r['cv_accuracy_mean'], 4),
                 'balanced_accuracy': round(r['cv_balanced_accuracy_mean'], 4),
                 'macro_f1': round(r['cv_macro_f1_mean'], 4)})
ablation = pd.DataFrame(rows).sort_values('balanced_accuracy', ascending=False)
ablation"""),
    code("""fig, ax = plt.subplots(figsize=(7,3.2))
ax.barh(ablation.block, ablation.balanced_accuracy, color='#17365D')
for i,(b,v,n) in enumerate(zip(ablation.block, ablation.balanced_accuracy, ablation.n_features)):
    ax.text(v+0.005, i, f'{v:.3f}  ({n} features)', va='center', fontsize=8)
ax.set_xlim(0,1.05); ax.set_xlabel('Balanced accuracy (grouped-event CV)')
ax.set_title('Feature block ablation'); ax.grid(axis='y', visible=False); plt.show()"""),
    md("""## Why no additional features were engineered

Frequency-domain features (FFT, spectral energy) were considered and rejected. They require the raw 10 Hz waveform; this extract contains only per-second summary statistics, so the underlying signal needed to compute them is not present. Claiming spectral features here would mean fabricating them.

Hour-of-day was excluded deliberately: notebook 03 shows motions were staged in daytime sessions, so time of day would separate the classes for reasons that have nothing to do with movement and would not survive deployment."""),
]

# ---------------------------------------------------------------- 05
NOTEBOOKS["05_baseline_models.ipynb"] = [
    md("""# 05 — Validation design and baseline models

**Question:** how should this dataset be split, and what does a sensible model achieve?

This is the most important notebook in the project."""),
    code(HEADER + "\nfrom smartnet.data import loader\nfrom smartnet.evaluation.splits import summarise_strategies, make_splitter, split_diagnostics\nfrom smartnet.models.experiment import run_sweep, leakage_table\n\ndf = loader.load_analysis_frame()"),
    md("""## The leakage risk

Each labelled motion is a contiguous run of 1-second epochs, and each row carries features for the 10 seconds either side. Adjacent rows within an event overlap in most of their input. Splitting rows at random puts near-duplicates on both sides."""),
    code("diag = summarise_strategies(df, 'motion_5cat')\ndiag[['strategy','grouping','shared_events','pct_test_from_seen_event']]"),
    md("""**Under a random split, 59.6% of test epochs come from a motion event that also appears in training.** Grouped splits reduce this to zero by construction."""),
    code("from smartnet.visualization import plots as P\nfig = P.plot_leakage_diagnostic(diag); plt.show()"),
    md("""## Does it actually matter?

The diagnostic establishes the *risk*. Whether it inflates scores is an empirical question."""),
    code("results, artefacts = run_sweep(df, label_col='motion_5cat')\nleakage_table(results)"),
    md("""### The hypothesis was wrong, and that is the finding

Inflation is **0.3 to 1.0 percentage points** — small. The structural risk is real and measurable, but on this dataset it does not materially change the score.

The likely reason: the classes are separated by broad differences in signal energy that generalise across events and days, not by event-specific quirks a model could memorise.

Reporting this honestly is worth more than the tidier story. The design is still correct — the diagnostic is cheap, the risk was genuine, and on a dataset with longer events or subtler classes the answer could easily have gone the other way. **Grouped-event CV is used for every subsequent result.**"""),
    code("""base = results[results.model=='majority_baseline']
print('Majority baseline (grouped-event):')
print(base[base.strategy=='grouped_event'][['cv_accuracy_mean','cv_balanced_accuracy_mean']].round(4).to_string(index=False))
print('\\nA model predicting only the most common class reaches 37% accuracy.')
print('Every number below must be read against that floor.')"""),
    code("fig = P.plot_model_comparison(results); plt.show()"),
]

# ---------------------------------------------------------------- 06
NOTEBOOKS["06_model_comparison.ipynb"] = [
    md("""# 06 — Model comparison and error analysis

**Question:** which model, which label scheme, and where does it fail?"""),
    code(HEADER + "\nfrom smartnet.data import loader\nfrom smartnet.evaluation import metrics as M\nfrom smartnet.visualization import plots as P\n\ndf = loader.load_analysis_frame()\nR = config.RESULTS_DIR"),
    code("""res5 = pd.read_csv(R/'results_motion_5cat_all.csv')
res5[res5.strategy=='grouped_event'][
    ['model','cv_accuracy_mean','cv_balanced_accuracy_mean','cv_macro_f1_mean','cv_macro_auc_mean']
].round(4)"""),
    md("""All four real models land within ~1.5 points. **Logistic regression reaches 0.934 against the random forest's 0.947** — the ensemble buys about one point. On a project where a government team maintains the result, the simpler model is genuinely competitive.

Deep learning was rejected, not overlooked: 664 independent events and 91 minority-class examples cannot train or validate a 1D CNN or LSTM credibly."""),
    md("""## Accuracy versus balanced accuracy"""),
    code("fig = P.plot_accuracy_vs_balanced(res5); plt.show()"),
    code("""cm5 = pd.read_csv(R/'confusion_motion_5cat_grouped_event.csv', index_col=0)
fig = P.plot_confusion(cm5, 'Random forest, grouped-event CV — 5 categories'); plt.show()
pd.read_csv(R/'per_class_motion_5cat_grouped_event.csv').round(3)"""),
    md("""## Error analysis: entering versus exiting"""),
    code("""m = cm5.to_numpy()
labels = [c.replace('pred_','') for c in cm5.columns]
i_en, i_ex = labels.index('Enter'), labels.index('Exit')
en_err, ex_err = m[i_en].sum()-m[i_en,i_en], m[i_ex].sum()-m[i_ex,i_ex]
print(f'Enter: {en_err} errors, {m[i_en,i_ex]} of them predicted as Exit ({100*m[i_en,i_ex]/en_err:.0f}%)')
print(f'Exit : {ex_err} errors, {m[i_ex,i_en]} of them predicted as Enter ({100*m[i_ex,i_en]/ex_err:.0f}%)')"""),
    md("""**Entering and exiting are confused with each other**, not with the other behaviours.

This independently reproduces the published limitation. Koudou et al. reported entry/exit confusion accounting for 83.3% and 85.7% of errors on those classes. The same failure appears here, on different data, three years later, with a richer feature set.

That convergence suggests a physical limit rather than a dataset artefact: a single accelerometer on a net's side panel registers a similar disturbance whether a body moves in or out. **Direction is largely absent from the signal**, and no model can recover information the sensor never captured."""),
    md("""## Collapsing the classes"""),
    code("""res4 = pd.read_csv(R/'results_motion_4cat_all.csv')
comp = pd.DataFrame({
 '5-category': res5[(res5.strategy=='grouped_event')&(res5.model=='random_forest')].iloc[0][
     ['cv_accuracy_mean','cv_balanced_accuracy_mean','cv_macro_f1_mean']].values,
 '4-category': res4[(res4.strategy=='grouped_event')&(res4.model=='random_forest')].iloc[0][
     ['cv_accuracy_mean','cv_balanced_accuracy_mean','cv_macro_f1_mean']].values,
}, index=['accuracy','balanced_accuracy','macro_f1']).astype(float).round(4)
comp"""),
    code("""fig = P.plot_per_class(pd.read_csv(R/'per_class_motion_4cat_grouped_event.csv'),
                       'Per-class performance — 4 categories (grouped-event CV)'); plt.show()"""),
    md("""Merging entry and exit lifts sensitivity on that behaviour from 0.34/0.61 to **0.934**, and balanced accuracy from 0.787 to 0.965.

**Recommended operating point: the four-category model.** It supports counting net crossings and detecting when a net goes up or down — enough for net-use duration and timing. It does not support directional inference and should not be used for it."""),
]

# ---------------------------------------------------------------- 07
NOTEBOOKS["07_model_interpretation.ipynb"] = [
    md("""# 07 — Interpretation

**Question:** what is the model relying on, and does it correspond to anything physically sensible?

For a health application, a model nobody can interrogate is hard to govern. Permutation importance is computed on held-out folds under the grouped split — impurity importance from a fitted forest is measured on training data and biased toward high-cardinality features, which is the confound this project is about."""),
    code(HEADER + """
from smartnet.data import loader
from smartnet.models.interpret import permutation_importances, block_importance, logistic_coefficients
from smartnet.visualization import plots as P

df = loader.load_analysis_frame()

# Computed here rather than loaded from disk, so the interpretability step is
# reproduced in front of the reader. ~15s: 48 features x 5 repeats x 2 folds.
imp = permutation_importances(df, 'motion_5cat', n_repeats=5, max_folds=2)
imp.head(12).round(4)"""),
    code("fig = P.plot_feature_importance(imp); plt.show()"),
    code("block_importance(imp)"),
    md("""**The four window-level aggregates dominate; the 40 individual per-second lag features contribute almost nothing.** Mean z-axis displacement over the preceding 10 seconds is the strongest feature by nearly a factor of two.

This matches the published importance analysis, which found the top variables were averages over a single dimension across the two 10-second periods. Both point the same way: the discriminating information is the **aggregate orientation change of the net over a window**, not the fine structure within it.

Physically, that is coherent. Raising or lowering a net produces a sustained displacement; a body crossing the threshold produces a burst. Those are separable. The direction of the crossing is not."""),
    md("""## The interpretable model"""),
    code("""coef = logistic_coefficients(df, 'motion_5cat')
top = coef.abs().max(axis=1).sort_values(ascending=False).head(10).index
coef.loc[top].round(3)"""),
    md("""## What these numbers are not

Permutation importance measures what a model **relies on**, not what **causes** a behaviour. These are associations between engineered signal features and a human-applied label.

`meanzover10vector_back` being important does not mean z-axis displacement causes someone to enter a net. It means that, in this dataset, with this sensor placement, that statistic helps separate labels a human assigned from video.

**Prediction ≠ causation.** Any operational use must rest on prospective validation, not on feature rankings."""),
    md("""## Deployment implication

The feature set could likely be cut from 48 columns to fewer than 10 with little loss — relevant for on-device or low-bandwidth inference in field conditions.

A calibrated model should also be allowed to **abstain**: output *net crossed* with high confidence, and decline to guess direction. Matching the output to the model's real capability is more useful than forcing a five-way decision it cannot support."""),
]


# ---------------------------------------------------------------- 00
# Mirrors, step for step, the exploratory session that produced the results in
# the README. Kept as a single runnable notebook so every number can be checked.
NOTEBOOKS["00_full_analysis.ipynb"] = [
    md("""# 00 — Full analysis, start to finish

This notebook reproduces **every step actually run** to produce the results in the README, in the order they were run. Notebooks 01–07 break the same work into themed chapters; this one is the complete record in a single file.

Runtime is roughly 4–6 minutes, dominated by the model sweep and permutation importance.

**Expected outputs are stated in the markdown before each step**, so you can tell immediately if a result has drifted."""),

    md("""## 0. Setup"""),
    code(HEADER + """
from smartnet.data import loader, validation
from smartnet.evaluation.splits import summarise_strategies
from smartnet.evaluation import metrics as M
from smartnet.models.experiment import run_cv, run_sweep, leakage_table
from smartnet.models.interpret import permutation_importances, block_importance
from smartnet.visualization import plots as P

print('Label schemes :', list(config.LABEL_MAPS))
print('Features      :', len(config.ALL_FEATURES))
print('Seed          :', config.RANDOM_SEED)"""),

    md("""## 1. Load and derive time structure

Expect **1,340 rows x 53 columns**, resolving to **664 events across 25 recording days**."""),
    code("""df = loader.load_analysis_frame()
print(df.shape)
print('Period :', df.timestamp.min(), '->', df.timestamp.max())
print('Events :', df.event_id.nunique(), '| Days :', df.session_date.nunique())"""),

    md("""### Why events matter

Labelled motions are contiguous runs of 1-second epochs. This is the grouping unit for cross-validation, and it is only defensible if every run carries a single label — so that is asserted, not assumed."""),
    code("""gaps = df.sort_values('timestamp')['gap_seconds'].dropna()
print('Gaps of exactly 1s :', int((gaps == 1).sum()), 'of', len(gaps))
loader.assert_events_are_label_pure(df)
print('All events label-pure.')

events = loader.event_summary(df, 'motion_5cat')
print('\\nEvent length (epochs):')
print(events.n_epochs.describe()[['count','50%','max']].to_string())"""),

    md("""## 2. Data quality

Eight checks. Expect **all pass**, with `class_support` reporting the rarest class at 91 epochs for the 4-category scheme."""),
    code("""report = validation.run_all(df)
print(report.to_string(index=False))
print('\\nBlocking failure:', validation.has_blocking_failure(report))"""),

    md("""## 3. Class balance

The imbalance is the central constraint: **Enter (47) and Exit (44)** are the rarest classes and the most operationally interesting."""),
    code("""for col in ['motion_4cat', 'motion_5cat']:
    names = config.LABEL_MAPS[col]
    vc = df[col].value_counts().sort_index()
    print(f'--- {col}')
    for k, v in vc.items():
        print(f'    {names[int(k)]:<16} {v:5d}  ({100*v/len(df):5.1f}%)')"""),

    md("""## 4. Leakage diagnostic

Each row carries features for the 10 seconds either side, so adjacent rows within an event are near-duplicates.

Expect: **59.6%** of test epochs under a random split come from an event that is also in training. Grouped splits: **0.0%**."""),
    code("""diag = summarise_strategies(df, 'motion_5cat')
print(diag[['strategy','grouping','shared_events','pct_test_from_seen_event']].to_string(index=False))"""),
    code("fig = P.plot_leakage_diagnostic(diag); plt.show()"),

    md("""## 5. Model sweep

Five models x three validation designs. Slowest cell in the notebook (~90s).

Expect inflation of **0.003 to 0.010** — i.e. random splitting inflates accuracy by well under a percentage point."""),
    code("""results5, artefacts5 = run_sweep(df, label_col='motion_5cat')
leakage_table(results5)"""),

    md("""### The hypothesis was wrong

I built this expecting the random split to inflate results substantially. It does not. The structural risk is real and measurable — 59.6% shared events — but the classes are separated by broad signal-energy differences that generalise across events and days, not by event-specific quirks a model could memorise.

This is reported as a negative result rather than quietly dropped. The design is still correct: the diagnostic is cheap, the risk was genuine, and on a dataset with longer events or subtler classes the answer could have gone the other way.

**Grouped-event CV is used from here on.**"""),
    code("""print(results5[results5.strategy=='grouped_event'][
    ['model','cv_accuracy_mean','cv_balanced_accuracy_mean','cv_macro_f1_mean','cv_macro_auc_mean']
].round(4).to_string(index=False))"""),

    md("""## 6. The headline gap

Expect **94.7% accuracy** against **78.7% balanced accuracy** for the random forest — a 16-point gap."""),
    code("fig = P.plot_accuracy_vs_balanced(results5); plt.show()"),
    code("""names = config.LABEL_MAPS['motion_5cat']
ordered = [names[k] for k in sorted(names)]
art = artefacts5['motion_5cat|random_forest|grouped_event']
valid = ~np.isnan(art['oof_pred'])
y_true = np.array([names[int(v)] for v in art['y'][valid]])
y_pred = np.array([names[int(v)] for v in art['oof_pred'][valid]])

per_class = M.per_class_sensitivity_specificity(y_true, y_pred, ordered)
print(per_class.round(3).to_string(index=False))"""),

    md("""**Enter 0.34, Exit 0.61.** The two behaviours that matter most for malaria exposure are the ones the model handles worst, and overall accuracy conceals it entirely."""),
    code("""cm5 = M.confusion_frame(y_true, y_pred, ordered)
fig = P.plot_confusion(cm5, 'Random forest, grouped-event CV — 5 categories'); plt.show()"""),

    md("""## 7. Error analysis

Expect entry/exit confusion to account for the large majority of errors on those two classes."""),
    code("""m = cm5.to_numpy()
i_en, i_ex = ordered.index('Enter'), ordered.index('Exit')
en_err = m[i_en].sum() - m[i_en, i_en]
ex_err = m[i_ex].sum() - m[i_ex, i_ex]
print(f'Enter: {en_err} errors, {m[i_en,i_ex]} predicted as Exit  ({100*m[i_en,i_ex]/en_err:.0f}%)')
print(f'Exit : {ex_err} errors, {m[i_ex,i_en]} predicted as Enter ({100*m[i_ex,i_en]/ex_err:.0f}%)')"""),

    md("""Entering and exiting are confused **with each other**, not with the other behaviours.

This independently reproduces the limitation in Koudou et al. (2022), who reported entry/exit confusion accounting for 83.3% and 85.7% of errors on those classes — on different data, three years earlier, with a different feature set.

That convergence points to a physical limit rather than a modelling problem: one accelerometer on a net's side panel registers a similar disturbance whether a body moves in or out. **Direction is largely absent from the signal.**"""),

    md("""## 8. Collapsing entry and exit

Expect the 4-category model to reach **0.978 accuracy / 0.965 balanced accuracy**, with the combined class at **0.934 sensitivity**."""),
    code("""results4, artefacts4 = run_sweep(df, label_col='motion_4cat')
rf4 = results4[(results4.strategy=='grouped_event') & (results4.model=='random_forest')].iloc[0]
rf5 = results5[(results5.strategy=='grouped_event') & (results5.model=='random_forest')].iloc[0]

pd.DataFrame({
    '5-category': [rf5.cv_accuracy_mean, rf5.cv_balanced_accuracy_mean, rf5.cv_macro_f1_mean],
    '4-category': [rf4.cv_accuracy_mean, rf4.cv_balanced_accuracy_mean, rf4.cv_macro_f1_mean],
}, index=['accuracy','balanced_accuracy','macro_f1']).round(4)"""),
    code("""names4 = config.LABEL_MAPS['motion_4cat']
ordered4 = [names4[k] for k in sorted(names4)]
a4 = artefacts4['motion_4cat|random_forest|grouped_event']
v4 = ~np.isnan(a4['oof_pred'])
pc4 = M.per_class_sensitivity_specificity(
    np.array([names4[int(v)] for v in a4['y'][v4]]),
    np.array([names4[int(v)] for v in a4['oof_pred'][v4]]), ordered4)
print(pc4.round(3).to_string(index=False))
fig = P.plot_per_class(pc4, 'Per-class — 4 categories (grouped-event CV)'); plt.show()"""),

    md("""Merging entry and exit lifts sensitivity on that behaviour from 0.34/0.61 to **0.934**.

**The four-category model is the recommended operating point.** It supports counting net crossings and detecting when a net goes up or down — enough for net-use duration and timing. It does not support directional inference and must not be used for it."""),

    md("""## 9. Feature ablation

Expect **8 features (current + aggregates) to beat all 48** on balanced accuracy, and backward-only to collapse to ~0.59."""),
    code("""rows = []
for block in config.FEATURE_BLOCKS:
    r = run_cv(df, 'motion_5cat', 'random_forest', 'grouped_event',
               features=config.FEATURE_BLOCKS[block])
    rows.append({'block': block, 'n_features': r['n_features'],
                 'accuracy': round(r['cv_accuracy_mean'], 4),
                 'balanced_accuracy': round(r['cv_balanced_accuracy_mean'], 4)})
pd.DataFrame(rows).sort_values('balanced_accuracy', ascending=False)"""),

    md("""Two consequences.

**Eight features outperform forty-eight.** The individual per-second lag columns add nothing — the feature set can be cut by 83%, which matters for on-device inference in field conditions.

**Forward context is essential.** Dropping it collapses balanced accuracy from 0.797 to 0.593. Knowing what happens *after* the tagged second is what separates a net going up from a net going down — so this classifier is inherently retrospective and needs ~10 seconds of lookahead. It cannot run in true real time. That constraint would otherwise only have surfaced in production."""),

    md("""## 10. Interpretation

Slow cell (~15s). Expect the **10-epoch aggregates to dominate**, led by `meanzover10vector_back`."""),
    code("""imp = permutation_importances(df, 'motion_5cat', n_repeats=5, max_folds=2)
print(imp.head(10).round(4).to_string(index=False))
print()
print(block_importance(imp).to_string(index=False))"""),
    code("fig = P.plot_feature_importance(imp); plt.show()"),

    md("""The window aggregates carry the signal; the 40 individual lag features contribute nothing (permuting them slightly *improves* held-out balanced accuracy, consistent with noise).

This matches the published importance analysis, which found averages over a single dimension across the two 10-second periods to be the top variables. Both point the same way: the discriminating information is the **aggregate orientation change of the net over a window**, not the fine structure within it. Raising or lowering a net produces a sustained displacement; a body crossing produces a burst. Those separate. Direction does not.

**Prediction is not causation.** These are associations between engineered signal features and a human-applied label, nothing more."""),

    md("""## 11. Summary of what this run established

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

### Limitations this run cannot address

1. **No participant identifier**, so performance on a *new person* cannot be estimated — the question that matters most for deployment.
2. **No demographics**, so the adult/child fairness comparison reported in the published study cannot be reproduced.
3. **44–47 epochs** in the minority classes means roughly 9 test examples per fold; those per-class figures carry wide uncertainty.
4. **Staged daytime conditions**, not natural overnight net use.

The first two are blocked by the extract, not by method. A participant-linked extract from the PI would be the single biggest upgrade to this project."""),
]


def main() -> None:
    for name, cells in NOTEBOOKS.items():
        (NB_DIR / name).write_text(json.dumps(notebook(cells), indent=1))
        print(f"wrote notebooks/{name}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
