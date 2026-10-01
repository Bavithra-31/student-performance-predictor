"""Streamlit dashboard - Student Performance Predictor.

Run:  python -m streamlit run app.py
(Run `python train.py` first so that models/risk_model.joblib exists.)
"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from features import build_features

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

RISK_COLORS = {"Low": "#2e9e5b", "Medium": "#f0a30a", "High": "#d64545"}
MIN_WEEKS, MIN_TESTS = 3, 2


@st.cache_resource
def load_bundle():
    return joblib.load("models/risk_model.joblib")


@st.cache_data
def load_default():
    return pd.read_csv("data/new_students.csv")


def cols_of(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix) and any(ch.isdigit() for ch in c)]
    return sorted(cols, key=lambda c: int("".join(ch for ch in c if ch.isdigit())))


def validate_and_clean(raw):
    """Check the uploaded CSV. Returns (clean_df, warnings) or raises ValueError."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    problems = []
    if "student_id" not in df.columns:
        problems.append("`student_id`")
    att, tests = cols_of(df, "attendance_w"), cols_of(df, "test")
    if len(att) < MIN_WEEKS:
        problems.append(f"at least {MIN_WEEKS} attendance columns (`attendance_w1`, `attendance_w2`, ...)")
    if len(tests) < MIN_TESTS:
        problems.append(f"at least {MIN_TESTS} test columns (`test1`, `test2`, ...)")
    if "assignment_completion" not in df.columns:
        problems.append("`assignment_completion`")
    if problems:
        raise ValueError("Missing required columns: " + ", ".join(problems))

    warnings = []
    numeric = att + tests + ["assignment_completion"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    missing = int(df[numeric].isna().sum().sum())
    if missing:
        df[numeric] = df[numeric].fillna(df[numeric].median())
        warnings.append(f"{missing} missing or invalid values were replaced with the column median.")
    if df["student_id"].duplicated().any():
        warnings.append("Duplicate student IDs found. Student detail shows the first match.")
    return df, warnings


# ---------- Model check ----------
if not os.path.exists("models/risk_model.joblib"):
    st.error("Trained model not found. Please run `python train.py` first.")
    st.stop()
bundle = load_bundle()

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Settings")
uploaded = st.sidebar.file_uploader("Upload students CSV", type="csv")

with st.sidebar.expander("Required CSV format"):
    st.markdown(
        "- `student_id`\n"
        "- `attendance_w1`, `attendance_w2`, ... (weekly attendance %, at least 3 weeks)\n"
        "- `test1`, `test2`, ... (test scores, at least 2 tests)\n"
        "- `assignment_completion` (%)\n\n"
        "An `at_risk` column is not needed. Best results with 8 weeks and 5 tests, "
        "the same layout the model was trained on."
    )
    if os.path.exists("data/new_students.csv"):
        st.download_button("⬇️ Download sample template",
                           load_default().head(5).to_csv(index=False).encode(),
                           "students_template.csv", "text/csv")

threshold = st.sidebar.slider(
    "At-risk flag threshold", 0.1, 0.9, float(bundle["threshold"]), 0.05,
    help="A lower threshold flags more students (higher recall, more false alarms).")

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
    except Exception:
        st.error("The file could not be read as a CSV. Please check the file and try again.")
        st.stop()
    source_label = f"Uploaded file: {uploaded.name}"
elif os.path.exists("data/new_students.csv"):
    raw = load_default()
    source_label = "Sample data: data/new_students.csv"
else:
    st.warning("Please upload a students CSV file, or run `python generate_data.py` to create sample data.")
    st.stop()

try:
    raw, warnings = validate_and_clean(raw)
except ValueError as e:
    st.error(str(e))
    st.info("Open **Required CSV format** in the sidebar to see the expected columns and download a template.")
    st.stop()

# ---------- Predict ----------
X = build_features(raw)[bundle["columns"]]
df = raw.copy()
df["risk_probability"] = bundle["model"].predict_proba(X)[:, 1]
df["risk_level"] = pd.cut(df["risk_probability"], [-0.01, 0.3, 0.6, 1.0],
                          labels=["Low", "Medium", "High"]).astype(str)
df["flag_at_risk"] = df["risk_probability"] >= threshold

# ---------- Header + KPIs ----------
st.title("🎓 Student Performance Predictor")
st.caption("Early warning dashboard that flags at-risk students using attendance and test-score trends")
st.caption(f"📄 {source_label}")
for w in warnings:
    st.warning(w)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total students", len(df))
k2.metric("Flagged at-risk", int(df["flag_at_risk"].sum()))
k3.metric("High risk", int((df["risk_level"] == "High").sum()))
k4.metric("Average attendance %", f"{X['att_mean'].mean():.1f}")

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🧑‍🎓 Student detail", "📋 At-risk list"])

# ---------- Tab 1: Overview ----------
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        counts = df["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(counts.index, counts.values, color=[RISK_COLORS[k] for k in counts.index])
        for i, v in enumerate(counts.values):
            ax.text(i, v, str(v), ha="center", va="bottom")
        ax.set_title("Risk level distribution")
        ax.set_ylabel("Students")
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(df["risk_probability"], bins=20, color="#4a6fa5", edgecolor="white")
        ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold {threshold:.2f}")
        ax.set_title("Risk probability distribution")
        ax.set_xlabel("Risk probability")
        ax.legend()
        st.pyplot(fig)

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = df["flag_at_risk"].map({True: "#d64545", False: "#2e9e5b"})
        ax.scatter(X["att_mean"], X["score_mean"], c=colors, alpha=0.6, s=18)
        ax.set_xlabel("Average attendance %")
        ax.set_ylabel("Average test score")
        ax.set_title("Attendance vs score (red = flagged)")
        st.pyplot(fig)

    with c4:
        model = bundle["model"]
        final = model[-1] if hasattr(model, "steps") else model
        if hasattr(final, "feature_importances_"):
            imp = pd.Series(final.feature_importances_, index=bundle["columns"])
        else:
            imp = pd.Series(np.abs(final.coef_[0]), index=bundle["columns"])
        imp = imp.sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.barh(imp.index, imp.values, color="#4a6fa5")
        ax.set_title("What drives the risk? (feature importance)")
        st.pyplot(fig)

# ---------- Tab 2: Student detail ----------
with tab2:
    sid = st.selectbox("Select a student", df["student_id"])
    row = df[df["student_id"] == sid].iloc[0]
    att_cols, test_cols = cols_of(df, "attendance_w"), cols_of(df, "test")

    m1, m2 = st.columns(2)
    m1.metric("Risk probability", f"{row['risk_probability']*100:.1f}%")
    m2.metric("Risk level", row["risk_level"])

    d1, d2 = st.columns(2)
    with d1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(range(1, len(att_cols) + 1), row[att_cols].astype(float), marker="o",
                label=str(sid), color=RISK_COLORS[row["risk_level"]])
        ax.plot(range(1, len(att_cols) + 1), df[att_cols].mean(), linestyle="--",
                color="gray", label="Class average")
        ax.set_title("Weekly attendance %")
        ax.set_xlabel("Week")
        ax.legend()
        st.pyplot(fig)
    with d2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(range(1, len(test_cols) + 1), row[test_cols].astype(float), marker="o",
                label=str(sid), color=RISK_COLORS[row["risk_level"]])
        ax.plot(range(1, len(test_cols) + 1), df[test_cols].mean(), linestyle="--",
                color="gray", label="Class average")
        ax.set_title("Test scores")
        ax.set_xlabel("Test")
        ax.legend()
        st.pyplot(fig)

# ---------- Tab 3: At-risk list ----------
with tab3:
    out = (df.loc[df["flag_at_risk"], ["student_id", "risk_probability", "risk_level"]]
           .sort_values("risk_probability", ascending=False)
           .assign(risk_probability=lambda d: d["risk_probability"].round(3)))
    st.write(f"**{len(out)} students flagged** (threshold {threshold:.2f})")
    st.dataframe(out, hide_index=True)
    st.download_button("⬇️ Download CSV", out.to_csv(index=False).encode(),
                       "at_risk_students.csv", "text/csv")