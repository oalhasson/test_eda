"""
BRFSS 2020-2024 Weighted EDA Dashboard
Single-file Streamlit app. All population statistics are weighted by _LLCPWT_POOLED.
"""
from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="BRFSS 2020-2024 EDA", layout="wide")
sns.set_theme(style="whitegrid")

DEFAULT_PATH = (
    "/root/.cache/kagglehub/datasets/ajenks/brfss-2020-2024-cleaned-and-weighted/"
    "versions/4/brfss_2020_2024_pooled_eda.parquet"
)
WEIGHT_COL = "_LLCPWT_POOLED"

BINARY_INDICATORS = [
    "GOOD_HEALTH", "HAS_DIABETES", "HAS_DEPRESSION", "HAS_HEART_ATTACK",
    "HAS_STROKE", "EVER_SMOKED", "HAS_EXERCISE", "GOT_FLUSHOT",
    "RECENT_CHECKUP", "HAS_PERSONAL_DOCTOR",
]

# Decode maps
GENHLTH_MAP = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
GENHLTH_ORDER = ["Excellent", "Very good", "Good", "Fair", "Poor"]

EDUCAG_MAP = {1: "No HS", 2: "HS grad", 3: "Some college", 4: "College grad"}
EDUCAG_ORDER = ["No HS", "HS grad", "Some college", "College grad"]

EMPLOY_MAP = {
    1: "Employed", 2: "Self-employed", 3: "Out of work 1y+",
    4: "Out of work <1y", 5: "Homemaker", 6: "Student",
    7: "Retired", 8: "Unable to work",
}
EMPLOY_ORDER = [
    "Employed", "Self-employed", "Out of work <1y", "Out of work 1y+",
    "Homemaker", "Student", "Retired", "Unable to work",
]

SEX_MAP = {1: "Male", 2: "Female"}

AGEG5YR_MAP = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
    6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
    11: "70-74", 12: "75-79", 13: "80+",
}
AGEG5YR_ORDER = [AGEG5YR_MAP[i] for i in range(1, 14)]

BMI5CAT_MAP = {1: "Underweight", 2: "Normal", 3: "Overweight", 4: "Obese"}
BMI5CAT_ORDER = ["Underweight", "Normal", "Overweight", "Obese"]

MENT14D_MAP = {1: "0 days", 2: "1-13 days", 3: "14+ days"}
MENT14D_ORDER = ["0 days", "1-13 days", "14+ days"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading parquet (~126 MB)...")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Decode → *_lbl columns; keep originals
    if "GENHLTH" in df:
        df["GENHLTH_lbl"] = df["GENHLTH"].map(GENHLTH_MAP)
    if "_EDUCAG" in df:
        df["EDUCAG_lbl"] = df["_EDUCAG"].map(EDUCAG_MAP)
    if "EMPLOY1" in df:
        df["EMPLOY_lbl"] = df["EMPLOY1"].map(EMPLOY_MAP)
    if "_SEX" in df:
        df["SEX_lbl"] = df["_SEX"].map(SEX_MAP)
    if "_AGEG5YR" in df:
        df["_AGEG5YR"] = df["_AGEG5YR"].where(df["_AGEG5YR"] != 14)  # 14 = unknown
        df["AGEG5YR_lbl"] = df["_AGEG5YR"].map(AGEG5YR_MAP)
    if "_BMI5CAT" in df:
        df["BMI5CAT_lbl"] = df["_BMI5CAT"].map(BMI5CAT_MAP)
    if "_MENT14D" in df:
        df["MENT14D_lbl"] = df["_MENT14D"].map(MENT14D_MAP)
    if "MENTHLTH" in df:
        df["FMD"] = (df["MENTHLTH"] >= 14).astype("float")
        df.loc[df["MENTHLTH"].isna(), "FMD"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Weighted helpers
# ---------------------------------------------------------------------------
def wmean(data: pd.DataFrame, col: str, weight: str = WEIGHT_COL) -> float:
    if col not in data or weight not in data or len(data) == 0:
        return np.nan
    v = data[col].to_numpy(dtype="float64", copy=False)
    w = data[weight].to_numpy(dtype="float64", copy=False)
    mask = ~np.isnan(v) & ~np.isnan(w) & (w > 0)
    if not mask.any():
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def wmean_by(data: pd.DataFrame, group: str, col: str,
             weight: str = WEIGHT_COL) -> pd.Series:
    if group not in data or col not in data:
        return pd.Series(dtype="float64")
    sub = data[[group, col, weight]].dropna(subset=[group])
    out = sub.groupby(group, dropna=True).apply(
        lambda g: wmean(g, col, weight), include_groups=False
    )
    return out.dropna()


def wprop_category(data: pd.DataFrame, cat_col: str,
                   order: Sequence[str] | None = None,
                   weight: str = WEIGHT_COL) -> pd.Series:
    if cat_col not in data or weight not in data:
        return pd.Series(dtype="float64")
    sub = data[[cat_col, weight]].dropna()
    if sub.empty:
        return pd.Series(dtype="float64")
    totals = sub.groupby(cat_col)[weight].sum()
    props = totals / totals.sum()
    if order is not None:
        props = props.reindex(order)
    return props


def reindex_safe(s: pd.Series, order: Iterable) -> pd.Series:
    return s.reindex([o for o in order if o in s.index])


# ---------------------------------------------------------------------------
# Sidebar / filters
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
data_path = st.sidebar.text_input("Parquet path", value=DEFAULT_PATH)

if not os.path.exists(data_path):
    st.error(f"File not found: {data_path}")
    st.stop()

df = load_data(data_path)

years_available = sorted(df["SURVEY_YEAR"].dropna().unique().astype(int).tolist()) \
    if "SURVEY_YEAR" in df else []
sel_years = st.sidebar.multiselect("Survey years", years_available,
                                   default=years_available)

sex_options = ["Male", "Female"]
sel_sex = st.sidebar.multiselect("Sex", sex_options, default=sex_options)

mask = pd.Series(True, index=df.index)
if sel_years and "SURVEY_YEAR" in df:
    mask &= df["SURVEY_YEAR"].isin(sel_years)
if sel_sex and "SEX_lbl" in df:
    mask &= df["SEX_lbl"].isin(sel_sex)
fdf = df.loc[mask]

st.sidebar.metric("Filtered rows", f"{len(fdf):,}")
st.sidebar.caption(f"All stats weighted by `{WEIGHT_COL}`.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🇺🇸 BRFSS 2020-2024 — Weighted Health EDA")
st.caption("CDC Behavioral Risk Factor Surveillance System pooled file. "
           "All population estimates use survey weights.")

tabs = st.tabs([
    "📊 Overview", "💉 Flu Shots", "🧠 Mental Health",
    "🩺 Access", "⚕️ Chronic & BMI", "🔗 Correlations", "🔑 Key Findings",
])


# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (filtered)", f"{len(fdf):,}")
    if "SURVEY_YEAR" in fdf and len(fdf):
        yr_min, yr_max = int(fdf["SURVEY_YEAR"].min()), int(fdf["SURVEY_YEAR"].max())
        c2.metric("Year range", f"{yr_min}–{yr_max}")
    c3.metric("Columns", f"{df.shape[1]}")

    st.subheader("Records per year")
    if "SURVEY_YEAR" in fdf:
        counts = fdf["SURVEY_YEAR"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(counts.index.astype(int), counts.values, color="#4C78A8")
        ax.set_xlabel("Year"); ax.set_ylabel("Unweighted respondents")
        st.pyplot(fig); plt.close(fig)

    st.subheader("Weighted prevalence — 10 core binary indicators")
    prev = {c: wmean(fdf, c) for c in BINARY_INDICATORS if c in fdf}
    prev_s = pd.Series(prev).dropna().sort_values()
    if not prev_s.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(prev_s.index, prev_s.values * 100, color="#4C78A8")
        ax.set_xlabel("Weighted % of adults"); ax.set_xlim(0, 100)
        for i, v in enumerate(prev_s.values):
            ax.text(v * 100 + 0.5, i, f"{v*100:.1f}%", va="center", fontsize=9)
        st.pyplot(fig); plt.close(fig)
        st.dataframe((prev_s * 100).round(2).rename("weighted_pct")
                     .to_frame().iloc[::-1])

        top = prev_s.idxmax(); bot = prev_s.idxmin()
        st.info(
            f"**Observed:** highest prevalence = `{top}` "
            f"({prev_s.max()*100:.1f}%); lowest = `{bot}` "
            f"({prev_s.min()*100:.1f}%). "
            f"Median prevalence across the 10 indicators is "
            f"{prev_s.median()*100:.1f}%."
        )


# ---------------------------------------------------------------------------
# Tab 2: Flu Shots
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Weighted flu-shot rate by year")
    if {"SURVEY_YEAR", "GOT_FLUSHOT"}.issubset(fdf.columns):
        trend = wmean_by(fdf, "SURVEY_YEAR", "GOT_FLUSHOT").sort_index()
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(trend.index.astype(int), trend.values * 100,
                marker="o", color="#E45756")
        ax.set_ylabel("Weighted % vaccinated"); ax.set_xlabel("Year")
        st.pyplot(fig); plt.close(fig)

        if len(trend) >= 2:
            delta = (trend.iloc[-1] - trend.iloc[0]) * 100
            direction = "increased" if delta > 0 else "decreased"
            st.info(
                f"**Observed:** flu vaccination {direction} from "
                f"{trend.iloc[0]*100:.1f}% in {int(trend.index.min())} to "
                f"{trend.iloc[-1]*100:.1f}% in {int(trend.index.max())} "
                f"(Δ = {delta:+.1f} pp)."
            )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**By age group**")
        if "AGEG5YR_lbl" in fdf:
            s = reindex_safe(wmean_by(fdf, "AGEG5YR_lbl", "GOT_FLUSHOT"),
                             AGEG5YR_ORDER) * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(s.index, s.values, color="#4C78A8")
            ax.set_ylabel("% vaccinated"); ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig); plt.close(fig)
            if not s.dropna().empty:
                st.caption(f"Range: {s.min():.1f}% → {s.max():.1f}% "
                           f"(peak in {s.idxmax()}).")
    with c2:
        st.markdown("**By education**")
        if "EDUCAG_lbl" in fdf:
            s = reindex_safe(wmean_by(fdf, "EDUCAG_lbl", "GOT_FLUSHOT"),
                             EDUCAG_ORDER) * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(s.index, s.values, color="#4C78A8")
            ax.set_ylabel("% vaccinated"); ax.tick_params(axis="x", rotation=20)
            st.pyplot(fig); plt.close(fig)
            if not s.dropna().empty:
                gap = s.max() - s.min()
                st.caption(f"Education gap: {gap:.1f} pp between "
                           f"`{s.idxmax()}` and `{s.idxmin()}`.")

    st.markdown("**By employment status**")
    if "EMPLOY_lbl" in fdf:
        s = reindex_safe(wmean_by(fdf, "EMPLOY_lbl", "GOT_FLUSHOT"),
                         EMPLOY_ORDER) * 100
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(s.index, s.values, color="#4C78A8")
        ax.set_xlabel("% vaccinated")
        st.pyplot(fig); plt.close(fig)
        if not s.dropna().empty:
            st.info(
                f"**Observed:** highest flu-shot rate among `{s.idxmax()}` "
                f"({s.max():.1f}%), lowest among `{s.idxmin()}` "
                f"({s.min():.1f}%)."
            )


# ---------------------------------------------------------------------------
# Tab 3: Mental Health
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Distribution of poor-mental-health days (MENTHLTH, 0–30)")
    if "MENTHLTH" in fdf:
        vals = fdf["MENTHLTH"].dropna()
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(vals, bins=31, color="#72B7B2", edgecolor="white")
        ax.set_xlabel("Days in past 30"); ax.set_ylabel("Respondents")
        st.pyplot(fig); plt.close(fig)
        pct_zero = (vals == 0).mean() * 100
        pct_fmd = (vals >= 14).mean() * 100
        st.caption(f"Unweighted: {pct_zero:.1f}% report 0 bad days; "
                   f"{pct_fmd:.1f}% report ≥14 (Frequent Mental Distress).")

    st.subheader("Frequent Mental Distress (≥14 days) — weighted trend")
    if {"SURVEY_YEAR", "FMD"}.issubset(fdf.columns):
        trend = wmean_by(fdf, "SURVEY_YEAR", "FMD").sort_index()
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(trend.index.astype(int), trend.values * 100,
                marker="o", color="#B279A2")
        ax.set_ylabel("% adults with FMD"); ax.set_xlabel("Year")
        st.pyplot(fig); plt.close(fig)
        if len(trend) >= 2:
            delta = (trend.iloc[-1] - trend.iloc[0]) * 100
            st.info(
                f"**Observed:** FMD prevalence moved from "
                f"{trend.iloc[0]*100:.1f}% ({int(trend.index.min())}) to "
                f"{trend.iloc[-1]*100:.1f}% ({int(trend.index.max())}) "
                f"(Δ = {delta:+.1f} pp)."
            )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**FMD by age group**")
        if "AGEG5YR_lbl" in fdf:
            s = reindex_safe(wmean_by(fdf, "AGEG5YR_lbl", "FMD"),
                             AGEG5YR_ORDER) * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(s.index, s.values, color="#B279A2")
            ax.set_ylabel("% with FMD"); ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig); plt.close(fig)
            if not s.dropna().empty:
                st.caption(f"Peak `{s.idxmax()}` ({s.max():.1f}%); "
                           f"lowest `{s.idxmin()}` ({s.min():.1f}%).")
    with c2:
        st.markdown("**FMD by employment**")
        if "EMPLOY_lbl" in fdf:
            s = reindex_safe(wmean_by(fdf, "EMPLOY_lbl", "FMD"),
                             EMPLOY_ORDER) * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(s.index, s.values, color="#B279A2")
            ax.set_xlabel("% with FMD")
            st.pyplot(fig); plt.close(fig)
            if not s.dropna().empty:
                st.caption(f"Highest among `{s.idxmax()}` "
                           f"({s.max():.1f}%).")


# ---------------------------------------------------------------------------
# Tab 4: Access to care
# ---------------------------------------------------------------------------
with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Recent checkup by education**")
        if "EDUCAG_lbl" in fdf:
            s = reindex_safe(wmean_by(fdf, "EDUCAG_lbl", "RECENT_CHECKUP"),
                             EDUCAG_ORDER) * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(s.index, s.values, color="#54A24B")
            ax.set_ylabel("% with recent checkup")
            ax.tick_params(axis="x", rotation=20)
            st.pyplot(fig); plt.close(fig)
            if not s.dropna().empty:
                st.caption(f"Gap: {s.max()-s.min():.1f} pp "
                           f"({s.idxmax()} vs {s.idxmin()}).")
    with c2:
        st.markdown("**Recent checkup by age**")
        if "AGEG5YR_lbl" in fdf:
            s = reindex_safe(wmean_by(fdf, "AGEG5YR_lbl", "RECENT_CHECKUP"),
                             AGEG5YR_ORDER) * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(s.index, s.values, color="#54A24B")
            ax.set_ylabel("% with recent checkup")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig); plt.close(fig)
            if not s.dropna().empty:
                st.caption(f"Peak `{s.idxmax()}` ({s.max():.1f}%); "
                           f"lowest `{s.idxmin()}` ({s.min():.1f}%).")

    st.subheader("Has a personal doctor — weighted trend")
    if {"SURVEY_YEAR", "HAS_PERSONAL_DOCTOR"}.issubset(fdf.columns):
        trend = wmean_by(fdf, "SURVEY_YEAR", "HAS_PERSONAL_DOCTOR").sort_index()
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(trend.index.astype(int), trend.values * 100,
                marker="o", color="#54A24B")
        ax.set_ylabel("% with personal doctor"); ax.set_xlabel("Year")
        st.pyplot(fig); plt.close(fig)
        if len(trend) >= 2:
            delta = (trend.iloc[-1] - trend.iloc[0]) * 100
            st.info(
                f"**Observed:** personal-doctor coverage went from "
                f"{trend.iloc[0]*100:.1f}% to {trend.iloc[-1]*100:.1f}% "
                f"(Δ = {delta:+.1f} pp across "
                f"{int(trend.index.min())}–{int(trend.index.max())})."
            )


# ---------------------------------------------------------------------------
# Tab 5: Chronic & BMI
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Diabetes prevalence by BMI category")
    if {"BMI5CAT_lbl", "HAS_DIABETES"}.issubset(fdf.columns):
        s = reindex_safe(wmean_by(fdf, "BMI5CAT_lbl", "HAS_DIABETES"),
                         BMI5CAT_ORDER) * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(s.index, s.values, color="#F58518")
        ax.set_ylabel("% with diabetes")
        st.pyplot(fig); plt.close(fig)
        if not s.dropna().empty:
            ratio = s.max() / s.min() if s.min() > 0 else np.nan
            st.info(
                f"**Observed:** diabetes prevalence ranges from "
                f"{s.min():.1f}% (`{s.idxmin()}`) to {s.max():.1f}% "
                f"(`{s.idxmax()}`) — a {ratio:.1f}× difference."
            )

    st.subheader("Self-rated general health (weighted)")
    if "GENHLTH_lbl" in fdf:
        s = wprop_category(fdf, "GENHLTH_lbl", GENHLTH_ORDER) * 100
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(s.index, s.values, color="#4C78A8")
        ax.set_ylabel("% of adults")
        st.pyplot(fig); plt.close(fig)
        if not s.dropna().empty:
            good = s.reindex(["Excellent", "Very good", "Good"]).sum()
            poor = s.reindex(["Fair", "Poor"]).sum()
            st.caption(f"Good-or-better: {good:.1f}%; Fair/Poor: {poor:.1f}%.")

    st.subheader("Self-reported good health by age")
    if {"AGEG5YR_lbl", "GOOD_HEALTH"}.issubset(fdf.columns):
        s = reindex_safe(wmean_by(fdf, "AGEG5YR_lbl", "GOOD_HEALTH"),
                         AGEG5YR_ORDER) * 100
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(s.index, s.values, marker="o", color="#4C78A8")
        ax.set_ylabel("% reporting good health")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig); plt.close(fig)
        if not s.dropna().empty:
            st.info(
                f"**Observed:** good-health share drops from "
                f"{s.iloc[0]:.1f}% ({s.index[0]}) to {s.iloc[-1]:.1f}% "
                f"({s.index[-1]}); peak at `{s.idxmax()}` "
                f"({s.max():.1f}%)."
            )


# ---------------------------------------------------------------------------
# Tab 6: Correlations
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Pearson correlations across health indicators")
    cols = [
        "GOOD_HEALTH", "HAS_DIABETES", "HAS_DEPRESSION", "HAS_HEART_ATTACK",
        "HAS_STROKE", "EVER_SMOKED", "HAS_EXERCISE", "GOT_FLUSHOT",
        "RECENT_CHECKUP", "_BMI5_SCALED", "MENTHLTH", "PHYSHLTH",
    ]
    cols = [c for c in cols if c in fdf.columns]
    if cols:
        # sample for speed if huge
        sub = fdf[cols]
        if len(sub) > 300_000:
            sub = sub.sample(300_000, random_state=0)
        corr = sub.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, vmin=-1, vmax=1, ax=ax,
                    annot_kws={"size": 8})
        st.pyplot(fig); plt.close(fig)

        # strongest off-diagonal pair
        c = corr.where(~np.eye(len(corr), dtype=bool))
        stacked = c.stack().dropna()
        if not stacked.empty:
            top_pair = stacked.abs().idxmax()
            top_val = stacked.loc[top_pair]
            st.info(
                f"**Observed:** strongest correlation is "
                f"`{top_pair[0]}` ↔ `{top_pair[1]}` (r = {top_val:+.2f}). "
                "Note: correlations among 0/1 indicators are bounded — "
                "interpret with care."
            )


# ---------------------------------------------------------------------------
# Tab 7: Key Findings (computed live)
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Key findings (computed from filtered data)")
    bullets: list[str] = []

    prev = pd.Series({c: wmean(fdf, c) for c in BINARY_INDICATORS
                      if c in fdf}).dropna()
    if not prev.empty:
        bullets.append(
            f"- Among 10 indicators, **{prev.idxmax()}** is most prevalent "
            f"({prev.max()*100:.1f}%) and **{prev.idxmin()}** is least "
            f"({prev.min()*100:.1f}%)."
        )

    if {"SURVEY_YEAR", "GOT_FLUSHOT"}.issubset(fdf.columns):
        t = wmean_by(fdf, "SURVEY_YEAR", "GOT_FLUSHOT").sort_index()
        if len(t) >= 2:
            bullets.append(
                f"- Flu vaccination: {t.iloc[0]*100:.1f}% "
                f"({int(t.index.min())}) → {t.iloc[-1]*100:.1f}% "
                f"({int(t.index.max())}) (Δ {(t.iloc[-1]-t.iloc[0])*100:+.1f} pp)."
            )

    if {"SURVEY_YEAR", "FMD"}.issubset(fdf.columns):
        t = wmean_by(fdf, "SURVEY_YEAR", "FMD").sort_index()
        if len(t) >= 2:
            bullets.append(
                f"- Frequent Mental Distress: {t.iloc[0]*100:.1f}% → "
                f"{t.iloc[-1]*100:.1f}% "
                f"(Δ {(t.iloc[-1]-t.iloc[0])*100:+.1f} pp)."
            )

    if {"BMI5CAT_lbl", "HAS_DIABETES"}.issubset(fdf.columns):
        s = reindex_safe(wmean_by(fdf, "BMI5CAT_lbl", "HAS_DIABETES"),
                         BMI5CAT_ORDER)
        if not s.dropna().empty and s.min() > 0:
            bullets.append(
                f"- Diabetes in Obese vs Normal BMI: "
                f"{s.get('Obese', np.nan)*100:.1f}% vs "
                f"{s.get('Normal', np.nan)*100:.1f}% "
                f"({s.get('Obese', np.nan)/s.get('Normal', np.nan):.1f}×)."
            )

    if {"EDUCAG_lbl", "RECENT_CHECKUP"}.issubset(fdf.columns):
        s = reindex_safe(wmean_by(fdf, "EDUCAG_lbl", "RECENT_CHECKUP"),
                         EDUCAG_ORDER) * 100
        if not s.dropna().empty:
            bullets.append(
                f"- Recent checkup education gap: "
                f"{s.max():.1f}% ({s.idxmax()}) vs "
                f"{s.min():.1f}% ({s.idxmin()}) — "
                f"{s.max()-s.min():+.1f} pp."
            )

    if {"AGEG5YR_lbl", "GOOD_HEALTH"}.issubset(fdf.columns):
        s = reindex_safe(wmean_by(fdf, "AGEG5YR_lbl", "GOOD_HEALTH"),
                         AGEG5YR_ORDER) * 100
        if not s.dropna().empty:
            bullets.append(
                f"- Self-rated good health: {s.iloc[0]:.1f}% "
                f"at {s.index[0]} → {s.iloc[-1]:.1f}% at {s.index[-1]}."
            )

    st.markdown("\n".join(bullets) if bullets
                else "_No findings — check filters or data path._")
    st.caption("All figures weighted by `_LLCPWT_POOLED`. "
               "Filters in sidebar affect every tab.")
