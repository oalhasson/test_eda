# app.py
# BRFSS 2020-2024 EDA Dashboard
# Run with:  streamlit run app.py

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(page_title="BRFSS 2020-2024 EDA", layout="wide")
sns.set_style("whitegrid")

# ----------------------------------------------------------------------
# LABEL MAPS (BRFSS codebook)
# ----------------------------------------------------------------------
GENHLTH_MAP = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
EMPLOY_MAP  = {1: "Employed", 2: "Self-employed", 3: "Out of work 1y+",
               4: "Out of work <1y", 5: "Homemaker", 6: "Student",
               7: "Retired", 8: "Unable to work"}
SEX_MAP     = {1: "Male", 2: "Female"}
AGECAT_MAP  = {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
               6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
               11: "70-74", 12: "75-79", 13: "80+"}
EDUCAG_MAP  = {1: "No HS", 2: "HS grad", 3: "Some college", 4: "College grad"}
BMICAT_MAP  = {1: "Underweight", 2: "Normal", 3: "Overweight", 4: "Obese"}
MENT14D_MAP = {1: "0 days", 2: "1-13 days", 3: "14+ days"}

AGE_ORDER   = list(AGECAT_MAP.values())
EDU_ORDER   = list(EDUCAG_MAP.values())
BMI_ORDER   = list(BMICAT_MAP.values())
GH_ORDER    = list(GENHLTH_MAP.values())

W = "_LLCPWT_POOLED"

BINARY_COLS = ["GOOD_HEALTH", "HAS_DIABETES", "HAS_DEPRESSION", "HAS_HEART_ATTACK",
               "HAS_STROKE", "EVER_SMOKED", "HAS_EXERCISE", "GOT_FLUSHOT",
               "RECENT_CHECKUP", "HAS_PERSONAL_DOCTOR"]

DEFAULT_PATH = ("/root/.cache/kagglehub/datasets/ajenks/"
                "brfss-2020-2024-cleaned-and-weighted/versions/4/"
                "brfss_2020_2024_pooled_eda.parquet")

# ----------------------------------------------------------------------
# DATA LOADING (cached so it only reads once)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data(path):
    df = pd.read_parquet(path)
    df["genhlth_lbl"] = df["GENHLTH"].map(GENHLTH_MAP)
    df["educ_lbl"]    = df["_EDUCAG"].map(EDUCAG_MAP)
    df["employ_lbl"]  = df["EMPLOY1"].map(EMPLOY_MAP)
    df["sex_lbl"]     = df["_SEX"].map(SEX_MAP)
    df["age_lbl"]     = df["_AGEG5YR"].replace(14, np.nan).map(AGECAT_MAP)
    df["bmicat_lbl"]  = df["_BMI5CAT"].map(BMICAT_MAP)
    df["ment14d_lbl"] = df["_MENT14D"].map(MENT14D_MAP)
    df["fmd"] = np.where(df["MENTHLTH"].notna(),
                         (df["MENTHLTH"] >= 14).astype(float), np.nan)
    return df

# ----------------------------------------------------------------------
# WEIGHTED HELPERS
# ----------------------------------------------------------------------
def wmean(data, col, weight=W):
    d = data[[col, weight]].dropna()
    if len(d) == 0:
        return np.nan
    return np.average(d[col], weights=d[weight])

def wmean_by(data, group, col, weight=W):
    d = data.dropna(subset=[group, col, weight])
    if d.empty:
        return pd.Series(dtype=float)
    return (d.groupby(group)
             .apply(lambda g: np.average(g[col], weights=g[weight]))
             .rename(col))

def wprop_category(data, cat_col, order, weight=W):
    d = data.dropna(subset=[cat_col, weight])
    s = d.groupby(cat_col)[weight].sum()
    s = (s / s.sum() * 100)
    return s.reindex(order)

# ----------------------------------------------------------------------
# SIDEBAR — data path + filters
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")
path = st.sidebar.text_input("Parquet file path", value=DEFAULT_PATH)

try:
    df = load_data(path)
except Exception as e:
    st.error(f"Could not load data from:\n{path}\n\n{e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("🔎 Filters")

years = sorted(df["SURVEY_YEAR"].dropna().unique().astype(int))
sel_years = st.sidebar.multiselect("Survey year(s)", years, default=years)

sex_opts = ["Male", "Female"]
sel_sex = st.sidebar.multiselect("Sex", sex_opts, default=sex_opts)

# Apply filters
mask = df["SURVEY_YEAR"].isin(sel_years) & df["sex_lbl"].isin(sel_sex)
dff = df[mask]

st.sidebar.markdown(f"**Rows after filter:** {len(dff):,}")
st.sidebar.caption("All statistics are weighted using _LLCPWT_POOLED.")

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.title("🏥 BRFSS 2020–2024 — Health Survey EDA")
st.caption("CDC Behavioral Risk Factor Surveillance System • "
           f"{len(df):,} respondents • weighted analysis")

# ----------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------
tab_over, tab_flu, tab_mh, tab_access, tab_chronic, tab_corr = st.tabs(
    ["📊 Overview", "💉 Flu Shots", "🧠 Mental Health",
     "🩺 Access", "⚕️ Chronic & BMI", "🔗 Correlations"]
)

# ============================ OVERVIEW =================================
with tab_over:
    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Respondents (filtered)", f"{len(dff):,}")
    c2.metric("Years", f"{min(sel_years)}–{max(sel_years)}" if sel_years else "—")
    c3.metric("Columns", df.shape[1])

    st.markdown("#### Records per year")
    rec = dff["SURVEY_YEAR"].value_counts().sort_index()
    st.bar_chart(rec)

    st.markdown("#### Weighted national prevalence of key indicators")
    prev = pd.Series({c: round(wmean(dff, c) * 100, 1) for c in BINARY_COLS})
    prev = prev.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    prev.plot(kind="barh", color="steelblue", ax=ax)
    ax.set_xlabel("% of US adults")
    ax.set_title("Weighted Prevalence of Key Health Indicators")
    ax.invert_yaxis()
    st.pyplot(fig)
    st.dataframe(prev.rename("% of adults"))

# ============================ FLU SHOTS ===============================
with tab_flu:
    st.subheader("💉 Flu Shot Uptake")

    flu_year = dff.groupby("SURVEY_YEAR").apply(lambda g: wmean(g, "GOT_FLUSHOT")) * 100
    fig, ax = plt.subplots()
    flu_year.plot(marker="o", ax=ax, color="coral")
    ax.set_ylabel("% vaccinated"); ax.set_xlabel("Year")
    ax.set_title("Flu Shot Uptake Over Time (weighted)")
    ax.grid(True)
    st.pyplot(fig)

    col1, col2 = st.columns(2)

    with col1:
        flu_age = wmean_by(dff, "age_lbl", "GOT_FLUSHOT").reindex(AGE_ORDER) * 100
        fig, ax = plt.subplots()
        flu_age.plot(kind="bar", color="coral", ax=ax)
        ax.set_ylabel("% vaccinated"); ax.set_title("By Age Group")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    with col2:
        flu_edu = wmean_by(dff, "educ_lbl", "GOT_FLUSHOT").reindex(EDU_ORDER) * 100
        fig, ax = plt.subplots()
        flu_edu.plot(kind="bar", color="seagreen", ax=ax)
        ax.set_ylabel("% vaccinated"); ax.set_title("By Education")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    flu_emp = wmean_by(dff, "employ_lbl", "GOT_FLUSHOT").sort_values() * 100
    fig, ax = plt.subplots(figsize=(9, 4))
    flu_emp.plot(kind="barh", color="mediumpurple", ax=ax)
    ax.set_xlabel("% vaccinated"); ax.set_title("By Employment Status")
    st.pyplot(fig)

    st.info("💡 Flu shots typically rise steeply with age and education.")

# ============================ MENTAL HEALTH ===========================
with tab_mh:
    st.subheader("🧠 Mental Health (poor mental health days)")

    mh = dff[dff["MENTHLTH"].between(0, 30)]
    fig, ax = plt.subplots()
    ax.hist(mh["MENTHLTH"], bins=31, color="indianred")
    ax.set_xlabel("days (past 30)"); ax.set_ylabel("count")
    ax.set_title("Distribution of Poor Mental Health Days")
    st.pyplot(fig)

    fmd_year = dff.groupby("SURVEY_YEAR").apply(lambda g: wmean(g, "fmd")) * 100
    fig, ax = plt.subplots()
    fmd_year.plot(marker="o", color="darkred", ax=ax)
    ax.set_ylabel("% with 14+ bad days")
    ax.set_title("Frequent Mental Distress Over Time (weighted)")
    ax.grid(True)
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        fmd_age = wmean_by(dff, "age_lbl", "fmd").reindex(AGE_ORDER) * 100
        fig, ax = plt.subplots()
        fmd_age.plot(kind="bar", color="crimson", ax=ax)
        ax.set_ylabel("%"); ax.set_title("Frequent Distress by Age")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
    with col2:
        fmd_emp = wmean_by(dff, "employ_lbl", "fmd").sort_values() * 100
        fig, ax = plt.subplots()
        fmd_emp.plot(kind="barh", color="firebrick", ax=ax)
        ax.set_xlabel("%"); ax.set_title("Frequent Distress by Employment")
        st.pyplot(fig)

    st.info("💡 Mental distress is usually highest in young adults and people "
            "unable to work / unemployed — the opposite of physical health.")

# ============================ ACCESS ==================================
with tab_access:
    st.subheader("🩺 Healthcare Access & Checkups")

    col1, col2 = st.columns(2)
    with col1:
        chk_edu = wmean_by(dff, "educ_lbl", "RECENT_CHECKUP").reindex(EDU_ORDER) * 100
        fig, ax = plt.subplots()
        chk_edu.plot(kind="bar", color="teal", ax=ax)
        ax.set_ylabel("%"); ax.set_title("Recent Checkup by Education")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
    with col2:
        doc_year = dff.groupby("SURVEY_YEAR").apply(
            lambda g: wmean(g, "HAS_PERSONAL_DOCTOR")) * 100
        fig, ax = plt.subplots()
        doc_year.plot(marker="o", color="navy", ax=ax)
        ax.set_ylabel("%"); ax.set_title("Has Personal Doctor Over Time")
        ax.grid(True)
        st.pyplot(fig)

    chk_age = wmean_by(dff, "age_lbl", "RECENT_CHECKUP").reindex(AGE_ORDER) * 100
    fig, ax = plt.subplots(figsize=(9, 4))
    chk_age.plot(kind="bar", color="darkcyan", ax=ax)
    ax.set_ylabel("%"); ax.set_title("Recent Checkup by Age Group")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

# ============================ CHRONIC & BMI ===========================
with tab_chronic:
    st.subheader("⚕️ Chronic Conditions & BMI")

    dia_bmi = wmean_by(dff, "bmicat_lbl", "HAS_DIABETES").reindex(BMI_ORDER) * 100
    fig, ax = plt.subplots()
    dia_bmi.plot(kind="bar", color="chocolate", ax=ax)
    ax.set_ylabel("% with diabetes"); ax.set_title("Diabetes Rate by BMI Category")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    st.dataframe(dia_bmi.rename("% diabetes").round(1))

    st.markdown("#### Self-rated general health")
    gh = wprop_category(dff, "genhlth_lbl", GH_ORDER)
    fig, ax = plt.subplots()
    gh.plot(kind="bar", color="goldenrod", ax=ax)
    ax.set_ylabel("% of adults"); ax.set_title("Self-Rated General Health (weighted)")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

    gh_age = wmean_by(dff, "age_lbl", "GOOD_HEALTH").reindex(AGE_ORDER) * 100
    fig, ax = plt.subplots()
    gh_age.plot(marker="o", color="darkgreen", ax=ax)
    ax.set_ylabel("%"); ax.set_title("% Good/Excellent Health by Age")
    ax.tick_params(axis="x", rotation=45); ax.grid(True)
    st.pyplot(fig)

# ============================ CORRELATIONS ============================
with tab_corr:
    st.subheader("🔗 Correlation Between Health Indicators")

    corr_cols = ["GOOD_HEALTH", "HAS_DIABETES", "HAS_DEPRESSION", "HAS_HEART_ATTACK",
                 "HAS_STROKE", "EVER_SMOKED", "HAS_EXERCISE", "GOT_FLUSHOT",
                 "RECENT_CHECKUP", "_BMI5_SCALED", "MENTHLTH", "PHYSHLTH"]
    corr_cols = [c for c in corr_cols if c in dff.columns]
    corr = dff[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

    st.info("💡 Look for: MENTHLTH ↔ PHYSHLTH positive; HAS_EXERCISE negatively "
            "related to chronic conditions; BMI ↔ diabetes positive.")
