"""Publication-quality figures.

Every function answers one question and returns a Matplotlib figure. Styling is
deliberately restrained: no chartjunk, colour used to encode meaning only.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from smartnet import config

NAVY = "#17365D"
TEAL = "#1F6F6B"
CORAL = "#C0504D"
GREY = "#8C8C8C"
PALETTE = [NAVY, TEAL, "#E8A33D", CORAL, "#7A5195"]

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(config.FIGURES_DIR / f"{name}.png")
    plt.close(fig)


def plot_class_balance(df: pd.DataFrame, label_col: str) -> plt.Figure:
    """How many labelled epochs exist per behaviour?"""
    names = config.LABEL_MAPS[label_col]
    counts = df[label_col].value_counts().sort_index()
    labels = [names[int(i)] for i in counts.index]

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    bars = ax.barh(labels, counts.values, color=PALETTE[: len(labels)])
    for b, v in zip(bars, counts.values):
        ax.text(v + 8, b.get_y() + b.get_height() / 2,
                f"{v}  ({100*v/len(df):.1f}%)", va="center", fontsize=8)
    ax.set_xlabel("Labelled epochs")
    ax.set_title(f"Class balance — {label_col}")
    ax.set_xlim(0, counts.max() * 1.28)
    ax.grid(axis="y", visible=False)
    return fig


def plot_event_duration(events: pd.DataFrame) -> plt.Figure:
    """How long is a labelled motion event, in seconds?"""
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.hist(events["n_epochs"], bins=np.arange(0.5, events["n_epochs"].max() + 1.5),
            color=NAVY, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Epochs in event (seconds)")
    ax.set_ylabel("Number of events")
    ax.set_title("Contiguous motion events are short\n"
                 "Adjacent epochs within an event share ±10s of context features")
    return fig


def plot_temporal_coverage(df: pd.DataFrame, label_col: str) -> plt.Figure:
    """When during the day was each behaviour recorded?"""
    names = config.LABEL_MAPS[label_col]
    ct = pd.crosstab(df["hour"], df[label_col].map(names))
    ordered = [names[k] for k in sorted(names) if names[k] in ct.columns]
    ct = ct[ordered]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bottom = np.zeros(len(ct))
    for i, col in enumerate(ct.columns):
        ax.bar(ct.index, ct[col], bottom=bottom, label=col,
               color=PALETTE[i % len(PALETTE)], width=0.85)
        bottom += ct[col].values
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Labelled epochs")
    ax.set_title("Motion events were staged during daytime sessions;\n"
                 "'Nothing' was sampled across all 24 hours")
    ax.legend(frameon=False, fontsize=8, ncol=len(ct.columns))
    ax.set_xticks(range(0, 24, 2))
    return fig


def plot_leakage_diagnostic(diag: pd.DataFrame) -> plt.Figure:
    """How much event structure does each split share between train and test?"""
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    colors = [CORAL if v > 1 else TEAL for v in diag["pct_test_from_seen_event"]]
    bars = ax.barh(diag["strategy"], diag["pct_test_from_seen_event"], color=colors)
    for b, v in zip(bars, diag["pct_test_from_seen_event"]):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("% of test epochs whose motion event also appears in training")
    ax.set_title("Random splitting reuses motion events across the split")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    return fig


def plot_model_comparison(results: pd.DataFrame, metric: str = "cv_balanced_accuracy_mean") -> plt.Figure:
    """Model performance across split strategies."""
    piv = results.pivot_table(index="model", columns="strategy", values=metric)
    order = [s.name for s in config.SPLIT_STRATEGIES if s.name in piv.columns]
    piv = piv[order].sort_values(order[-1])

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(piv))
    w = 0.26
    for i, col in enumerate(piv.columns):
        ax.bar(x + (i - 1) * w, piv[col], w, label=col, color=PALETTE[i])
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in piv.index], fontsize=8)
    ax.set_ylabel(metric.replace("cv_", "").replace("_mean", "").replace("_", " "))
    ax.set_title("Model comparison under three validation designs")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0, 1.05)
    return fig


def plot_accuracy_vs_balanced(results: pd.DataFrame, strategy: str = "grouped_event") -> plt.Figure:
    """The headline honesty plot: accuracy hides minority-class failure."""
    sub = results[(results.strategy == strategy) & (results.model != "majority_baseline")]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub["cv_accuracy_mean"], 0.4, label="Overall accuracy", color=NAVY)
    ax.bar(x + 0.2, sub["cv_balanced_accuracy_mean"], 0.4, label="Balanced accuracy", color=CORAL)
    for i, (a, b) in enumerate(zip(sub["cv_accuracy_mean"], sub["cv_balanced_accuracy_mean"])):
        ax.annotate("", xy=(i + 0.2, b), xytext=(i - 0.2, a),
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8, ls=":"))
        ax.text(i, max(a, b) + 0.03, f"−{100*(a-b):.1f}pp", ha="center", fontsize=7.5, color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in sub["model"]], fontsize=8)
    ax.set_ylim(0, 1.18)
    ax.set_title("Overall accuracy overstates performance\nbecause the rare classes are the hard ones")
    ax.legend(frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.42))
    return fig


def plot_confusion(cm: pd.DataFrame, title: str, normalise: bool = True) -> plt.Figure:
    """Confusion matrix heatmap; rows are ground truth."""
    m = cm.to_numpy(dtype=float)
    display = m / m.sum(axis=1, keepdims=True) if normalise else m

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(display, cmap="Blues", vmin=0, vmax=1 if normalise else display.max())
    labels = [c.replace("pred_", "") for c in cm.columns]
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            val = display[i, j]
            ax.text(j, i, f"{val:.2f}\n({int(m[i,j])})", ha="center", va="center",
                    fontsize=7.5, color="white" if val > 0.55 else "#333333")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Row-normalised rate" if normalise else "Count")
    return fig


def plot_per_class(per_class: pd.DataFrame, title: str) -> plt.Figure:
    """Sensitivity and specificity side by side, with support annotated."""
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    x = np.arange(len(per_class))
    ax.bar(x - 0.2, per_class["sensitivity"], 0.4, label="Sensitivity (recall)", color=TEAL)
    ax.bar(x + 0.2, per_class["specificity"], 0.4, label="Specificity", color=NAVY)
    ax.axhline(0.8, ls="--", lw=0.8, color=CORAL)
    ax.text(len(per_class) - 0.45, 0.815, "0.80", fontsize=7, color=CORAL)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n(n={n})" for c, n in zip(per_class["class"], per_class["support"])], fontsize=8
    )
    ax.set_ylim(0, 1.08)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    return fig


def plot_feature_importance(imp: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Held-out permutation importance for the top features."""
    top = imp.head(top_n).iloc[::-1]
    block_colors = {
        "10-epoch aggregate": NAVY,
        "forward lead": TEAL,
        "backward lag": "#E8A33D",
        "current epoch": CORAL,
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.barh(top["feature"], top["importance"],
            color=[block_colors.get(b, GREY) for b in top["feature_block"]])
    ax.set_xlabel("Drop in balanced accuracy when permuted")
    ax.set_title("Permutation importance (held-out folds, grouped by event)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in block_colors.values()]
    ax.legend(handles, block_colors.keys(), frameon=False, fontsize=7.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    return fig


def plot_signal_examples(df: pd.DataFrame, label_col: str, seed: int = 0) -> plt.Figure:
    """Reconstructed ±10-epoch signal traces for one example of each behaviour."""
    names = config.LABEL_MAPS[label_col]
    rng = np.random.default_rng(seed)
    classes = sorted(names)
    fig, axes = plt.subplots(len(classes), 1, figsize=(7.0, 1.55 * len(classes)), sharex=True)

    back = [f"back_sumvector{i}" for i in range(10, 0, -1)]
    fwd = [f"forward_sumvector{i}" for i in range(1, 11)]
    offsets = list(range(-10, 11))

    for ax, k in zip(np.atleast_1d(axes), classes):
        sub = df[df[label_col] == k]
        if sub.empty:
            continue
        row = sub.iloc[rng.integers(len(sub))]
        trace = [row[c] for c in back] + [row["sum_vectormagnitudes"]] + [row[c] for c in fwd]
        ax.plot(offsets, trace, color=NAVY, lw=1.4)
        ax.axvline(0, color=CORAL, lw=1.0, ls="--")
        ax.set_ylabel(names[k], fontsize=8, rotation=0, ha="right", va="center")
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[-1].set_xlabel("Seconds relative to the labelled epoch (0 = tagged second)")
    fig.suptitle("Sum of vector magnitudes around one example of each behaviour",
                 fontsize=10, fontweight="bold", y=0.995)
    fig.tight_layout()
    return fig
