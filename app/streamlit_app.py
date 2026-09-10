"""Research demonstration app.

    streamlit run app/streamlit_app.py

NOT A MEDICAL DEVICE. This is a portfolio demonstration of a research
classifier. It has not been clinically validated, is not approved for any
diagnostic or programmatic use, and must not be used to make decisions about
individuals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smartnet import config
from smartnet.data import loader, validation
from smartnet.models.zoo import build_model

st.set_page_config(page_title="Bed-net behaviour classifier", layout="wide")

st.title("Bed-net use behaviour classification")
st.caption("Accelerometer-based behaviour classification — research demonstration")

st.warning(
    "**Research demonstration only.** This is not a medical device and has not "
    "been clinically or programmatically validated. Entering and exiting a net "
    "are classified poorly (sensitivity 0.34 and 0.61 respectively); see the "
    "Limitations section of the repository README before interpreting anything.",
    icon="⚠️",
)

LABEL = st.sidebar.selectbox(
    "Label scheme", config.LABEL_COLS,
    index=config.LABEL_COLS.index(config.PRIMARY_LABEL),
    help="The 4-category scheme is the recommended operating point.",
)
MODEL = st.sidebar.selectbox(
    "Model", ["random_forest", "gradient_boosting", "logistic_regression", "decision_tree"]
)
st.sidebar.divider()
st.sidebar.caption(
    "Training data stays local. Nothing uploaded here is transmitted anywhere."
)


@st.cache_data(show_spinner=False)
def _load(path: str | None):
    return loader.load_analysis_frame(path)


@st.cache_resource(show_spinner=False)
def _fit(label: str, model_name: str, path: str | None):
    df = _load(path)
    model = build_model(model_name)
    model.fit(df[config.ALL_FEATURES].to_numpy(), df[label].to_numpy())
    return model


tab_predict, tab_explore, tab_quality = st.tabs(
    ["Classify", "Explore training data", "Data quality"]
)

# --------------------------------------------------------------- reference
default_path = config.RAW_DIR / config.RAW_STATA_FILENAME
if not default_path.exists():
    default_path = config.RAW_DIR / "synthetic_sample.dta"
ref_path = str(default_path) if default_path.exists() else None

if ref_path is None:
    st.error(
        "No reference dataset found. Add the study extract to data/raw/ or run "
        "`python scripts/make_synthetic_sample.py`."
    )
    st.stop()

if "synthetic" in str(ref_path):
    st.info("Running on **synthetic** data — results describe the generator, not bed nets.")

# --------------------------------------------------------------- classify
with tab_predict:
    st.subheader("Classify epochs")
    st.write(
        "Upload a `.dta` or `.csv` with the same 48 feature columns as the study "
        "extract, or classify a held-out sample of the reference data."
    )
    up = st.file_uploader("Accelerometer feature file", type=["dta", "csv"])

    if st.button("Run classification", type="primary"):
        model = _fit(LABEL, MODEL, ref_path)
        names = config.LABEL_MAPS[LABEL]

        if up is not None:
            new = pd.read_stata(up) if up.name.endswith(".dta") else pd.read_csv(up)
            missing = sorted(set(config.ALL_FEATURES) - set(new.columns))
            if missing:
                st.error(f"Missing {len(missing)} required columns, e.g. {missing[:5]}")
                st.stop()
        else:
            new = _load(ref_path).sample(200, random_state=0)

        X = new[config.ALL_FEATURES].to_numpy()
        pred = model.predict(X)
        proba = model.predict_proba(X)
        classes = list(model[-1].classes_)

        out = pd.DataFrame({
            "predicted": [names[int(p)] for p in pred],
            "confidence": proba.max(axis=1).round(3),
        })
        if LABEL in new.columns:
            out.insert(0, "actual", [names[int(v)] for v in new[LABEL]])
            acc = float((out["actual"] == out["predicted"]).mean())
            st.metric("Agreement with supplied labels", f"{acc:.1%}")

        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(out, use_container_width=True, height=340)
        with c2:
            st.write("**Predicted distribution**")
            st.bar_chart(out["predicted"].value_counts())
            low = int((out["confidence"] < 0.6).sum())
            st.caption(
                f"{low} of {len(out)} predictions fall below 0.60 confidence. "
                "In deployment these should route to human review rather than "
                "being actioned automatically."
            )

# --------------------------------------------------------------- explore
with tab_explore:
    df = _load(ref_path)
    names = config.LABEL_MAPS[LABEL]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Labelled epochs", f"{len(df):,}")
    c2.metric("Motion events", f"{df['event_id'].nunique():,}")
    c3.metric("Recording days", df["session_date"].nunique())
    c4.metric("Features", len(config.ALL_FEATURES))

    st.write("**Class balance**")
    st.bar_chart(df[LABEL].map(names).value_counts())

    st.write("**Signal trace around a labelled epoch**")
    cls = st.selectbox("Behaviour", [names[k] for k in sorted(names)])
    sub = df[df[LABEL].map(names) == cls]
    if len(sub):
        row = sub.iloc[st.slider("Example", 0, len(sub) - 1, 0)]
        trace = (
            [row[f"back_sumvector{i}"] for i in range(10, 0, -1)]
            + [row["sum_vectormagnitudes"]]
            + [row[f"forward_sumvector{i}"] for i in range(1, 11)]
        )
        st.line_chart(pd.DataFrame({"sum of vector magnitudes": trace}, index=range(-10, 11)))

# --------------------------------------------------------------- quality
with tab_quality:
    df = _load(ref_path)
    report = validation.run_all(df)
    st.subheader("Automated data-quality checks")
    st.dataframe(
        report.style.map(
            lambda v: {
                "pass": "background-color:#e6f4ea",
                "warn": "background-color:#fff4e5",
                "fail": "background-color:#fdecea",
            }.get(v, ""),
            subset=["severity"],
        ),
        use_container_width=True,
    )
    st.caption(
        "These checks run on every pipeline execution. A `fail` blocks modelling."
    )
