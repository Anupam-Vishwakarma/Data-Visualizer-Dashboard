"""
Pro Data Dashboard — Polished, fully commented end-to-end script
Author: Polished for Anupam
Purpose:
 - Full-feature Streamlit data dashboard:
   * Upload CSV/XLSX
   * Filters (categorical, numeric range, date)
   * Auto-generate many visual types (Histogram, Scatter, Bar, Pie, Box, Heatmap, Pairplot, Area, Line, Top-N)
   * Per-visual color customization, notes, PNG download, shareable config code
   * Insights tab (missing values, correlations, outliers, category dominance)
   * Summary tab with descriptive stats
   * ML & Clustering: baseline models (Linear/Logistic, RandomForest), KMeans, AutoML selection, feature importances, model download
   * PDF report generation (charts + insights)
 - Fixed bugs:
   * Removed duplicate heatmap branch
   * Added helper to convert matplotlib figures to PNG bytes
   * Robust handling for plotly->png (kaleido) absence
   * Cleaned figure lifecycle to avoid plot overwrites
Requirements:
 pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn fpdf2 kaleido openpyxl
"""

# -----------------------------
# Imports
# -----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import io
import os
import tempfile
import json
import base64
import pickle
import time
from datetime import datetime

# PDF library
from fpdf import FPDF

# sklearn for ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    silhouette_score
)
from sklearn.decomposition import PCA

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="📊 Pro Data Dashboard", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# -----------------------------
# Helper functions (robust & commented)
# -----------------------------
# -----------------------------

def make_onehot():
    """
    Compatibility wrapper for OneHotEncoder between scikit-learn versions.
    Newer scikit-learn uses sparse_output parameter; older versions use sparse.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def read_file(uploaded):
    """
    Read uploaded file (CSV or Excel). Returns a pandas DataFrame.
    """
    name = uploaded.name.lower()
    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded, engine="openpyxl")
    else:
        # let pandas infer separator; this works for most CSVs
        return pd.read_csv(uploaded)


def download_df_as_csv(df):
    """
    Return CSV bytes for a dataframe for Streamlit download_button.
    """
    return df.to_csv(index=False).encode("utf-8")


def small_insights(df):
    """
    Quick, auto-generated textual insights:
     - Missing values summary
     - Top correlations (if strong)
     - Category dominance for first few categorical columns
    Returns list of insight strings.
    """
    insights = []
    total_missing = int(df.isnull().sum().sum())
    if total_missing > 0:
        insights.append(f"⚠️ There are {total_missing} missing values across the dataset.")
    else:
        insights.append("✅ No missing values detected.")

    numeric = df.select_dtypes(include=[np.number])
    # strong numeric correlations
    if numeric.shape[1] >= 2:
        corr = numeric.corr().abs()
        corr_unstack = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        if not corr_unstack.empty:
            top_pair = corr_unstack.idxmax()
            top_val = corr_unstack.max()
            if top_val > 0.8:
                insights.append(f"🔗 Strong correlation ({top_val:.2f}) between `{top_pair[0]}` and `{top_pair[1]}`.")

    # category dominance (first 3 categorical columns)
    cats = df.select_dtypes(include=['object', 'category'])
    for c in cats.columns[:3]:
        vc = df[c].value_counts()
        if not vc.empty:
            top = vc.index[0]
            pct = vc.iloc[0] / len(df) * 100
            insights.append(f"🏷️ Column `{c}` dominated by `{top}` ({pct:.1f}%).")

    return insights


def metric_regression(y_true, y_pred):
    """
    Common regression metrics returned as dictionary.
    """
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAE": mean_absolute_error(y_true, y_pred)
    }


def metric_classification(y_true, y_pred, y_proba=None):
    """
    Common classification metrics. If y_proba supplied (n_samples x n_classes), compute ROC appropriately.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    roc = None
    if y_proba is not None:
        try:
            # binary
            if len(np.unique(y_true)) == 2:
                roc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                roc = roc_auc_score(y_true, y_proba, multi_class='ovo', average='weighted')
        except Exception:
            roc = None
    return {"Accuracy": acc, "F1": f1, "ROC-AUC": roc}


def prepare_preprocessor(X):
    """
    Create a ColumnTransformer preprocessor:
     - numeric columns: median impute + StandardScaler
     - categorical columns: most_frequent impute + OneHotEncoder
    Returns (preprocessor, numeric_cols, cat_cols)
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot())
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipeline, numeric_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor, numeric_cols, cat_cols


def limit_dataframe_size(df, max_rows=50000):
    """
    If dataset is too large for ML operations, sample it and warn the user.
    """
    if len(df) > max_rows:
        st.warning(f"Dataset has {len(df):,} rows — sampling {max_rows} rows for faster ML.")
        return df.sample(max_rows, random_state=42).reset_index(drop=True)
    return df


def plotly_fig_to_png_bytes(fig):
    """
    Convert plotly figure to png bytes using plotly.io.to_image (kaleido).
    Returns bytes or None if conversion fails (e.g., kaleido missing).
    """
    try:
        return pio.to_image(fig, format="png", scale=2)
    except Exception:
        return None


def fig_to_png_bytes_matplotlib(fig):
    """
    Convert a matplotlib figure to PNG bytes.
    This is safe and works even if kaleido is not present.
    """
    buf = io.BytesIO()
    # fig may be a matplotlib Figure object OR a seaborn PairGrid (which has .fig)
    try:
        # seaborn pairplot returns a PairGrid; handle both cases
        if hasattr(fig, "fig"):
            fig.figure.savefig(buf, format="png", bbox_inches="tight")
        else:
            fig.savefig(buf, format="png", bbox_inches="tight")
    except Exception:
        # fallback: try plt.savefig
        plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# -----------------------------
# Theme & session state initialization
# -----------------------------
if "theme" not in st.session_state:
    st.session_state["theme"] = "Light"

def apply_theme():
    """
    Very small theme CSS toggle — keeps UI readable in light/dark.
    (This is minimal; Streamlit's native theming is preferred for production.)
    """
    if st.session_state["theme"] == "Dark":
        st.markdown("""
        <style>
        body { background-color: #121212; color: #e6e6e6; }
        .stApp { background-color: #121212; color: #e6e6e6; }
        .stContainer { border-color: #333; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        body { background-color: white; color: black; }
        .stContainer { border-color: #ddd; }
        </style>
        """, unsafe_allow_html=True)

apply_theme()

# -----------------------------
# Header (title, AI suggest button, theme toggle)
# -----------------------------
# st.title("📊 Pro Data Dashboard")

# col_header, col_theme = st.columns([4, 1])
# with col_header:
#     st.header("Explore Your Data: Visuals | Insights | Summary | ML & Clustering")
#     # Small AI suggestion button — shows recommended plots based on detected columns
#     if st.button("🤖 AI Suggest Charts", key="ai_suggest"):
#         with st.expander("AI Suggestions for Charts (Based on dataset)"):
#             st.write("- Histogram: numeric columns (distribution)")
#             st.write("- Scatter: two numeric columns (relationship)")
#             st.write("- Box: numeric column (outliers & quartiles)")
#             st.write("- Bar / Pie: categorical distributions")
#             st.write("- Heatmap: correlation between numeric columns")
#             st.write("- Pairplot: pairwise relationships for numeric columns")
# ============================================================
# Theme Toggle (Light / Dark)
# ============================================================
if "theme" not in st.session_state:
    st.session_state["theme"] = "Light"

col1, col2 = st.columns([8, 1])
with col1:
    st.title("📊 Pro Data Dashboard")
with col2:
    # Theme toggle button
    if st.button("🌙" if st.session_state["theme"] == "Light" else "☀️", key="theme_toggle"):
        st.session_state["theme"] = (
            "Dark" if st.session_state["theme"] == "Light" else "Light"
        )
        st.rerun()  # ✅ Updated for Streamlit v1.37+


# -----------------------------
# Sidebar: Upload + Filters
# -----------------------------
st.sidebar.header("📁 Upload & Filters")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xls", "xlsx"], key="uploader")

# If no file uploaded, ask user to upload and stop further execution
if not uploaded_file:
    st.info("Upload a CSV or Excel file from the sidebar to start the dashboard.")
    st.stop()

# -----------------------------
# Read and process the uploaded file
# -----------------------------
try:
    df = read_file(uploaded_file)
    # Keep a pristine copy for ML/Summary (so filters don't permanently remove rows)
    df_original = df.copy()
    st.success(f"Loaded `{uploaded_file.name}` — {df.shape[0]} rows × {df.shape[1]} cols")

    # Automatic type detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Datetime detection — try parsing strings to datetime if any column looks like datetime
    datetime_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
    # If none detected but some object columns look like dates, try to coerce
    if not datetime_cols:
        for c in df.columns:
            if df[c].dtype == object:
                try:
                    parsed = pd.to_datetime(df[c], errors='coerce')
                    # If a reasonable fraction parsed to datetimes, treat column as datetime
                    if parsed.notnull().sum() / max(1, len(parsed)) > 0.6:
                        df[c] = parsed
                        datetime_cols.append(c)
                except Exception:
                    pass

    # -----------------------------
    # Sidebar Filters (PowerBI-like)
    # -----------------------------
    st.sidebar.header("🔎 Filters")

    # Categorical filter
    selected_cat_col = st.sidebar.selectbox("Categorical Column (filter)", [None] + categorical_cols, key="s_cat")
    if selected_cat_col:
        unique_vals = df[selected_cat_col].dropna().unique().tolist()
        # Limit choices shown by default to first 5 to avoid huge lists
        chosen_vals = st.sidebar.multiselect(f"Values for `{selected_cat_col}`", unique_vals, default=unique_vals[:5], key="s_cat_vals")
        if chosen_vals:
            df = df[df[selected_cat_col].isin(chosen_vals)]

    # Numeric range filter
    selected_num_col = st.sidebar.selectbox("Numeric Column (range filter)", [None] + numeric_cols, key="s_num")
    if selected_num_col:
        min_val = float(df_original[selected_num_col].min())
        max_val = float(df_original[selected_num_col].max())
        lo, hi = st.sidebar.slider(f"Range for `{selected_num_col}`", min_val, max_val, (min_val, max_val), key="s_num_range")
        df = df[(df[selected_num_col] >= lo) & (df[selected_num_col] <= hi)]

    # Date filter (if any datetime columns)
    if datetime_cols:
        date_col = st.sidebar.selectbox("Date column filter (optional)", [None] + datetime_cols, key="s_date")
        if date_col:
            # Default start/end picks
            start_date = st.sidebar.date_input("Start date", value=pd.to_datetime(df_original[date_col].min()).date(), key="s_start")
            end_date = st.sidebar.date_input("End date", value=pd.to_datetime(df_original[date_col].max()).date(), key="s_end")
            df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]

except Exception as e:
    st.error(f"Failed to read or process file: {e}")
    st.stop()

# -----------------------------
# Tabs setup: Visuals | Insights | Summary | ML & Clustering
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Visuals", "🧠 Insights", "📋 Summary", "🤖 ML & Clustering"])

# -----------------------------
# TAB 1: VISUALS
# -----------------------------
with tab1:
    st.subheader("Auto-Generated Visuals (One per row, vertical layout)")
    # Prepare a list of visuals to auto-generate based on available columns
    visuals = []

    # Histograms for first up-to-3 numeric columns
    for col in numeric_cols[:3]:
        visuals.append({"type": "Histogram", "x": col})

    # Scatter(s)
    if len(numeric_cols) >= 2:
        # create up to 2 scatter combos
        for i in range(min(2, len(numeric_cols)-1)):
            visuals.append({"type": "Scatter", "x": numeric_cols[i], "y": numeric_cols[i+1]})

    # Bar(s): categorical vs numeric (first two cats)
    if categorical_cols and numeric_cols:
        for cat in categorical_cols[:2]:
            visuals.append({"type": "Bar", "x": cat, "y": numeric_cols[0]})

    # Pie(s)
    for cat in categorical_cols[:2]:
        visuals.append({"type": "Pie", "cat": cat})

    # Box plots for numeric
    for col in numeric_cols[:2]:
        visuals.append({"type": "Box", "y": col})

    # Heatmap (correlations) — single
    if len(numeric_cols) >= 2:
        visuals.append({"type": "Heatmap"})

    # Pairplot
    if len(numeric_cols) >= 2:
        visuals.append({"type": "Pairplot"})

    # Top N categories
    for cat in categorical_cols[:2]:
        visuals.append({"type": "Top N", "cat": cat})

    # Area chart if at least 2 numeric columns
    if len(numeric_cols) >= 2:
        visuals.append({"type": "Area", "y": numeric_cols[:2]})

    # Line if datetime + numeric
    if datetime_cols and numeric_cols:
        visuals.append({"type": "Line", "x": datetime_cols[0], "y": numeric_cols[0]})

    # Render each visual in its own container (one per row)
    for idx, vis in enumerate(visuals):
        # Separator + border container
        st.markdown("---")
        st.markdown(f"<div style='border: 2px solid #ddd; padding: 12px; border-radius: 8px; margin-bottom: 16px;'>", unsafe_allow_html=True)

        # Color picker stored in session_state for persistence
        default_color = st.session_state.get(f"color_{vis.get('type')}_{idx}", "#2E86AB")
        # Color picker - Streamlit automatically manages color state using unique keys
        vis_color = st.color_picker(
            f"🎨 Color for {vis['type']} {idx+1}",
            "#2E86AB",
            key=f"color_{vis['type']}_{idx}"
        )


        # Allow users to change axes / params where applicable
        fig = None  # Will hold a plotly figure for PNG export; matplotlib figs handled separately

        try:
            if vis["type"] == "Histogram":
                # X-axis selection
                xcol = st.selectbox(f"X-axis (Numeric) for Histogram {idx+1}", numeric_cols, index=numeric_cols.index(vis["x"]) if vis["x"] in numeric_cols else 0, key=f"x_hist_{idx}")
                fig = px.histogram(df, x=xcol, nbins=30)
                fig.update_traces(marker_color=vis_color)

            elif vis["type"] == "Scatter":
                xcol = st.selectbox(f"X-axis for Scatter {idx+1}", numeric_cols, index=numeric_cols.index(vis["x"]) if vis["x"] in numeric_cols else 0, key=f"x_scatter_{idx}")
                ycol = st.selectbox(f"Y-axis for Scatter {idx+1}", numeric_cols, index=numeric_cols.index(vis["y"]) if vis["y"] in numeric_cols else (1 if len(numeric_cols) > 1 else 0), key=f"y_scatter_{idx}")
                fig = px.scatter(df, x=xcol, y=ycol)
                # color not built-in here; use marker color via update_traces if single series
                fig.update_traces(marker=dict(color=vis_color))

            elif vis["type"] == "Bar":
                xcol = st.selectbox(f"Category (X) for Bar {idx+1}", categorical_cols, index=categorical_cols.index(vis["x"]) if vis["x"] in categorical_cols else 0, key=f"x_bar_{idx}")
                ycol = st.selectbox(f"Value (Y) for Bar {idx+1}", numeric_cols, index=numeric_cols.index(vis["y"]) if vis["y"] in numeric_cols else 0, key=f"y_bar_{idx}")
                agg_df = df.groupby(xcol)[ycol].mean().reset_index()
                fig = px.bar(agg_df, x=xcol, y=ycol)
                fig.update_traces(marker_color=vis_color)

            elif vis["type"] == "Pie":
                cat = st.selectbox(f"Category for Pie {idx+1}", categorical_cols, index=categorical_cols.index(vis["cat"]) if vis["cat"] in categorical_cols else 0, key=f"cat_pie_{idx}")
                vc = df[cat].value_counts().nlargest(6)
                fig = px.pie(values=vc.values, names=vc.index)

            elif vis["type"] == "Box":
                ycol = st.selectbox(f"Y-axis (Numeric) for Box {idx+1}", numeric_cols, index=numeric_cols.index(vis["y"]) if vis["y"] in numeric_cols else 0, key=f"y_box_{idx}")
                fig = px.box(df, y=ycol)
                # color not necessary; box has default styles

            elif vis["type"] == "Heatmap":
                # Use matplotlib + seaborn to show correlation heatmap
                fig_mat, ax = plt.subplots(figsize=(8, 6))
                # compute correlation only on current numeric columns
                corr_df = df[numeric_cols].corr()
                sns.heatmap(corr_df, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig_mat)
                # Save matplotlib fig bytes for PDF later if needed
                fig = None  # We already displayed matplotlib fig; set fig None to skip plotly handling

            elif vis["type"] == "Pairplot":
                # Seaborn pairplot can be heavy; sample if dataset large
                sample_df = df[numeric_cols[:3]].dropna()
                if len(sample_df) > 500:
                    sample_df = sample_df.sample(500, random_state=42)
                pairgrid = sns.pairplot(sample_df)
                # pairgrid has attribute .fig or .figure depending on seaborn version
                try:
                    st.pyplot(pairgrid.fig)
                except Exception:
                    # fallback: try drawing current figure
                    st.pyplot()
                fig = None

            elif vis["type"] == "Top N":
                cat = st.selectbox(f"Category for Top N {idx+1}", categorical_cols, index=categorical_cols.index(vis["cat"]) if vis["cat"] in categorical_cols else 0, key=f"cat_topn_{idx}")
                vc = df[cat].value_counts().nlargest(10)
                fig = px.bar(x=vc.index, y=vc.values, labels={'x': cat, 'y': 'count'})
                fig.update_traces(marker_color=vis_color)

            elif vis["type"] == "Area":
                # Allow user to choose multiple numeric columns for stacked area
                y_choices = st.multiselect(f"Y-axes (Numeric) for Area {idx+1}", numeric_cols, default=list(vis["y"]) if isinstance(vis["y"], list) else vis["y"], key=f"y_area_{idx}")
                if y_choices:
                    # Plotly area expects wide-form data; it's okay to pass dataframe with y columns
                    fig = px.area(df, y=y_choices)
                    # Plotly will automatically choose colors; we don't override here

            elif vis["type"] == "Line":
                # X can be datetime or categorical; allow None to plot simple line vs index
                x_options = [None] + datetime_cols + categorical_cols
                if vis.get("x") in x_options:
                    default_index = x_options.index(vis["x"])
                else:
                    default_index = 0
                xcol = st.selectbox(f"X-axis for Line {idx+1}", x_options, index=default_index, key=f"x_line_{idx}")
                ycol = st.selectbox(f"Y-axis (Numeric) for Line {idx+1}", numeric_cols, index=numeric_cols.index(vis["y"]) if vis["y"] in numeric_cols else 0, key=f"y_line_{idx}")
                if xcol:
                    fig = px.line(df, x=xcol, y=ycol)
                else:
                    fig = px.line(df, y=ycol)
                fig.update_traces(line=dict(color=vis_color))

            # Display Plotly figure and provide PNG download + share code
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{vis['type']}_{idx}_{time.time()}")

                # Convert to PNG bytes for download (if kaleido available)
                img_bytes = plotly_fig_to_png_bytes(fig)
                if img_bytes:
                    st.download_button("⬇️ Download PNG", data=img_bytes, file_name=f"{vis['type']}_{idx}.png", key=f"download_png_{idx}_{time.time()}")

                # Shareable config: encode the visual's config to base64 so another user can paste it (simple approach)
                try:
                    # include visible parameters only (not the full dataframe)
                    share_obj = {
                        "type": vis["type"],
                        "params": {k: v for k, v in vis.items() if k != "type"},
                        "color": vis_color
                    }
                    share_code = base64.urlsafe_b64encode(json.dumps(share_obj).encode()).decode()
                    st.code(f"Share Code: {share_code}", language="text")
                    st.caption("Copy this code to share this visual configuration.")
                except Exception:
                    pass

            # Comments / notes saved in session_state so notes persist on reruns
            comment_key = f"comment_{vis['type']}_{idx}"
            if comment_key not in st.session_state:
                st.session_state[comment_key] = ""
            st.text_area(
                f"💬 Add Notes / Comments for {vis['type']} {idx+1}",
                key=comment_key,
                placeholder="Write observations or insights here...",
            )
        except Exception as e:
            st.error(f"Error rendering {vis['type']} (index {idx}): {e}")

        # close border container
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# TAB 2: INSIGHTS
# -----------------------------
with tab2:
    st.subheader("🧠 Smart Dataset Insights")
    # Quick insights (missing values, strong correlations, category dominance)
    insights = small_insights(df_original)
    for ins in insights:
        st.write(ins)

    # Outlier detection using IQR method — show counts per numeric column
    st.markdown("### 📊 Outlier Detection (IQR Method)")
    if numeric_cols:
        outlier_report = []
        for col in numeric_cols:
            q1, q3 = df_original[col].quantile(0.25), df_original[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            count = ((df_original[col] < lower) | (df_original[col] > upper)).sum()
            outlier_report.append((col, int(count)))
        st.dataframe(pd.DataFrame(outlier_report, columns=["Column", "Outlier Count"]))
    else:
        st.info("No numeric columns available for outlier analysis.")

    # Category dominance table
    st.markdown("### 🏷️ Category Dominance")
    if categorical_cols:
        cat_report = []
        for col in categorical_cols:
            vc = df_original[col].value_counts()
            if not vc.empty:
                top_cat, pct = vc.index[0], vc.iloc[0]/len(df_original)*100
                cat_report.append((col, top_cat, round(pct, 1)))
        st.dataframe(pd.DataFrame(cat_report, columns=["Column", "Top Category", "% Share"]))
    else:
        st.info("No categorical columns found.")

# -----------------------------
# TAB 3: SUMMARY
# -----------------------------
with tab3:
    st.subheader("📋 Dataset Summary")
    # Show describe() for everything (includes counts, unique, top, freq for categorical)
    try:
        desc = df_original.describe(include='all').transpose()
        st.write(desc)
    except Exception as e:
        st.write("Could not compute full describe():", e)

    # Top numeric correlations (if exist)
    if len(numeric_cols) >= 2:
        st.subheader("🔗 Top Numeric Correlations")
        corr = df_original[numeric_cols].corr().abs()
        corr_un = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        if not corr_un.empty:
            top_corr = corr_un.sort_values(ascending=False).head(10)
            st.write(top_corr)
        else:
            st.write("No correlations found.")
    else:
        st.info("Need at least two numeric columns to compute correlations.")

# -----------------------------
# TAB 4: ML & CLUSTERING
# -----------------------------
with tab4:
    st.header("🤖 Quick ML Studio — Train baseline models & clustering")

    # Work with a bounded-size dataset for ML
    df_ml_full = limit_dataframe_size(df_original, max_rows=50000)

    st.markdown("### 1) Select target (label) for supervised tasks")
    target = st.selectbox("Target column", options=[None] + df_ml_full.columns.tolist(), key="ml_target")
    if target:
        n_unique = df_ml_full[target].nunique(dropna=True)
        # Heuristic: numeric target with many unique values -> regression
        is_numeric_target = pd.api.types.is_numeric_dtype(df_ml_full[target])
        if is_numeric_target and n_unique > 20:
            task = "regression"
        else:
            task = "classification"
        st.info(f"Auto-detected task: **{task}** (target unique values: {n_unique})")

        # Features selection (default: all except target, but limited to first 10)
        features = st.multiselect("Features (leave empty to use all except target)", options=[c for c in df_ml_full.columns if c != target], default=[c for c in df_ml_full.columns if c != target][:10], key="ml_features")
        if not features:
            st.warning("Select at least one feature to proceed with ML.")
        else:
            X = df_ml_full[features].copy()
            y = df_ml_full[target].copy()

            # Drop rows where target is null
            non_null_idx = ~y.isnull()
            X = X.loc[non_null_idx].reset_index(drop=True)
            y = y.loc[non_null_idx].reset_index(drop=True)

            # Preprocessing pipelines
            st.markdown("### 2) Preprocessing & train/test split")
            test_size = st.slider("Test set size (fraction)", 0.05, 0.5, 0.2, step=0.05, key="ml_test_size")
            random_state = 42
            preprocessor, num_cols, cat_cols = prepare_preprocessor(X)

            # Stratify for classification (if categories are few)
            stratify = None
            if task == "classification" and y.nunique() <= 10:
                stratify = y

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
            st.write(f"Train shape: {X_train.shape} — Test shape: {X_test.shape}")

            # Choose which baseline models to run
            st.markdown("### 3) Models to run (baseline)")
            run_lr = st.checkbox("Run Linear/Logistic model", value=True, key="run_lr")
            run_rf = st.checkbox("Run RandomForest model", value=True, key="run_rf")

            models_results = {}

            # Linear / Logistic
            if run_lr:
                if task == "regression":
                    pipe_lr = Pipeline([("pre", preprocessor), ("model", LinearRegression())])
                    pipe_lr.fit(X_train, y_train)
                    preds = pipe_lr.predict(X_test)
                    metrics = metric_regression(y_test, preds)
                    models_results["LinearRegression"] = {"pipeline": pipe_lr, "metrics": metrics}
                else:
                    pipe_lr = Pipeline([("pre", preprocessor), ("model", LogisticRegression(max_iter=300))])
                    try:
                        pipe_lr.fit(X_train, y_train)
                        preds = pipe_lr.predict(X_test)
                        proba = None
                        try:
                            proba = pipe_lr.predict_proba(X_test)
                        except Exception:
                            proba = None
                        metrics = metric_classification(y_test, preds, y_proba=proba)
                        models_results["LogisticRegression"] = {"pipeline": pipe_lr, "metrics": metrics}
                    except Exception as e:
                        st.warning(f"LogisticRegression training failed: {e}")

            # RandomForest
            if run_rf:
                if task == "regression":
                    pipe_rf = Pipeline([("pre", preprocessor), ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
                    pipe_rf.fit(X_train, y_train)
                    preds = pipe_rf.predict(X_test)
                    metrics = metric_regression(y_test, preds)
                    models_results["RandomForestRegressor"] = {"pipeline": pipe_rf, "metrics": metrics}
                else:
                    pipe_rf = Pipeline([("pre", preprocessor), ("model", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))])
                    pipe_rf.fit(X_train, y_train)
                    preds = pipe_rf.predict(X_test)
                    proba = None
                    try:
                        proba = pipe_rf.predict_proba(X_test)
                    except Exception:
                        proba = None
                    metrics = metric_classification(y_test, preds, y_proba=proba)
                    models_results["RandomForestClassifier"] = {"pipeline": pipe_rf, "metrics": metrics}

            # Display model results (metrics + optional feature importances)
            st.markdown("### 4) Model results")
            if not models_results:
                st.info("No models ran. Select at least one model checkbox.")
            else:
                for name, res in models_results.items():
                    st.markdown(f"#### {name}")
                    m = res["metrics"]
                    for k, v in m.items():
                        if v is None:
                            st.write(f"- {k}: N/A")
                        elif isinstance(v, float):
                            st.write(f"- {k}: **{v:.4f}**")
                        else:
                            st.write(f"- {k}: **{v}**")

                    # If model has feature_importances_ (RandomForest), map them to feature names
                    try:
                        model_obj = res["pipeline"].named_steps["model"]
                        if hasattr(model_obj, "feature_importances_"):
                            st.markdown("**Top feature importances (RandomForest)**")
                            try:
                                pre = res["pipeline"].named_steps["pre"]
                                num_names = num_cols
                                cat_names = []
                                if cat_cols:
                                    # get names from one-hot encoder
                                    try:
                                        ohe = pre.named_transformers_['cat'].named_steps['onehot']
                                        cat_ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
                                        cat_names = cat_ohe_names
                                    except Exception:
                                        # fallback: approximate by using cat input cols
                                        cat_names = cat_cols
                                feature_names = list(num_names) + list(cat_names)
                                importances = model_obj.feature_importances_
                                fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
                                fi_df = fi_df.sort_values("importance", ascending=False).head(20)
                                st.dataframe(fi_df.reset_index(drop=True))
                                # small horizontal bar chart
                                fig = px.bar(fi_df, x="importance", y="feature", orientation="h")
                                st.plotly_chart(fig, use_container_width=True, key=f"fi_{name}_{time.time()}")
                            except Exception as e:
                                st.write("Feature importance mapping failed:", e)
                    except Exception:
                        pass

                    # Confusion matrix for classification
                    if task == "classification":
                        try:
                            cm = confusion_matrix(y_test, res["pipeline"].predict(X_test))
                            st.write("Confusion matrix:")
                            fig, ax = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                            st.pyplot(fig)
                        except Exception:
                            pass

                        # ROC Curve if binary
                        try:
                            proba = None
                            if hasattr(res["pipeline"], "predict_proba"):
                                try:
                                    proba = res["pipeline"].predict_proba(X_test)
                                except Exception:
                                    proba = None
                            if proba is not None and y_test.nunique() == 2:
                                from sklearn.metrics import roc_curve, auc
                                fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
                                roc_auc = auc(fpr, tpr)
                                st.write(f"ROC AUC: {roc_auc:.4f}")
                                fig, ax = plt.subplots()
                                ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
                                ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
                                ax.set_xlabel("False Positive Rate")
                                ax.set_ylabel("True Positive Rate")
                                ax.legend()
                                st.pyplot(fig)
                        except Exception:
                            pass

                    # Model download (pickle)
                    try:
                        buf = io.BytesIO()
                        pickle.dump(res["pipeline"], buf)
                        buf.seek(0)
                        st.download_button(f"⬇️ Download `{name}` model (.pkl)", data=buf, file_name=f"{name}.pkl", key=f"download_model_{name}_{time.time()}")
                    except Exception as e:
                        st.write("Model download failed:", e)

            # -----------------------------
            # Clustering section (KMeans)
            # -----------------------------
            st.markdown("---")
            st.subheader("🔎 Optional: Clustering (KMeans)")
            run_cluster = st.checkbox("Run KMeans clustering on selected numeric features", value=False, key="run_kmeans")
            if run_cluster:
                cluster_features = st.multiselect("Choose numeric features for clustering", options=X.select_dtypes(include=[np.number]).columns.tolist(), default=X.select_dtypes(include=[np.number]).columns.tolist()[:3], key="cluster_features")
                if len(cluster_features) < 1:
                    st.warning("Select at least one numeric feature.")
                else:
                    X_cluster = X[cluster_features].dropna()
                    # sampling for visual clarity if very large
                    if len(X_cluster) > 5000:
                        X_cluster = X_cluster.sample(5000, random_state=42)
                        st.info("Sampling 5000 rows for clustering plot.")
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_cluster)
                    n_clusters = st.slider("Number of clusters (k)", 2, 12, 3, key="kmeans_k")
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = kmeans.fit_predict(X_scaled)
                    try:
                        sil = silhouette_score(X_scaled, labels)
                        st.write(f"Silhouette score: {sil:.4f}")
                    except Exception:
                        pass

                    # If dims > 2, use PCA to project to 2D for visualization
                    if X_scaled.shape[1] > 2:
                        pca = PCA(n_components=2)
                        proj = pca.fit_transform(X_scaled)
                        plot_df = pd.DataFrame({"x": proj[:, 0], "y": proj[:, 1], "cluster": labels.astype(str)})
                        fig = px.scatter(plot_df, x="x", y="y", color="cluster", title="KMeans clusters (PCA projection)")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        plot_df = pd.DataFrame(X_cluster.reset_index(drop=True))
                        plot_df["cluster"] = labels.astype(str)
                        if X_cluster.shape[1] == 2:
                            cols_plot = X_cluster.columns.tolist()
                            fig = px.scatter(plot_df, x=cols_plot[0], y=cols_plot[1], color="cluster", title="KMeans clusters")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write("Not enough dimensions for a clear 2D clustering plot.")

            # -----------------------------
            # AutoML & PDF Report
            # -----------------------------
            st.markdown("---")
            st.subheader("🚀 AutoML Recommend + Report Generation")
            run_automl = st.checkbox("Run AutoML recommendation & generate report", key="run_automl")
            if run_automl:
                st.info("Running a simple AutoML: train a small set of candidate models and pick best based on quick metric.")
                if task == "regression":
                    candidates = {
                        "LinearRegression": LinearRegression(),
                        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42)
                    }
                else:
                    candidates = {
                        "LogisticRegression": LogisticRegression(max_iter=300),
                        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42)
                    }

                best_score = -np.inf
                best_name = None
                best_pipe = None
                for name, mdl in candidates.items():
                    pipe = Pipeline([("pre", preprocessor), ("model", mdl)])
                    try:
                        pipe.fit(X_train, y_train)
                        preds = pipe.predict(X_test)
                        if task == "regression":
                            score = r2_score(y_test, preds)
                        else:
                            score = accuracy_score(y_test, preds)
                        st.write(f"{name} → Score: {score:.4f}")
                        if score > best_score:
                            best_score = score
                            best_name = name
                            best_pipe = pipe
                    except Exception as e:
                        st.write(f"{name} failed: {e}")

                if best_pipe is not None:
                    st.success(f"✅ Best Model: {best_name} (Score: {best_score:.4f})")
                    # Provide model download
                    buf = io.BytesIO()
                    try:
                        pickle.dump(best_pipe, buf)
                        buf.seek(0)
                        st.download_button("⬇️ Download Best Model (.pkl)", data=buf, file_name=f"best_model_{best_name}.pkl", key=f"download_best_{time.time()}")
                    except Exception as e:
                        st.write("Failed to prepare model download:", e)

                    # Now generate PDF report: insights + a few charts (if available)
                    st.subheader("📄 Generate PDF Report (insights + charts)")
                    # Create FPDF and write textual information and embed chart images
                    try:
                        pdf = FPDF()
                        pdf.set_auto_page_break(auto=True, margin=15)
                        pdf.add_page()
                        pdf.set_font("Arial", "B", 16)
                        pdf.cell(0, 10, "Pro Data Dashboard Report", ln=True, align="C")
                        pdf.ln(6)
                        pdf.set_font("Arial", "", 11)
                        pdf.cell(0, 8, f"Dataset: {uploaded_file.name}", ln=True)
                        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
                        pdf.ln(6)
                        pdf.set_font("Arial", "B", 12)
                        pdf.cell(0, 8, "Top Insights:", ln=True)
                        pdf.set_font("Arial", "", 11)
                        for insight in small_insights(df):
                            pdf.multi_cell(0, 6, f"- {insight}")
                        pdf.ln(4)

                        images_added = 0

                        # Add histogram if numeric columns exist
                        if numeric_cols:
                            try:
                                fig_hist = px.histogram(df, x=numeric_cols[0], nbins=30)
                                img_bytes = plotly_fig_to_png_bytes(fig_hist)
                                if img_bytes:
                                    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                                    tmpf.write(img_bytes)
                                    tmpf.flush()
                                    tmpf.close()
                                    pdf.add_page()
                                    pdf.set_font("Arial", "B", 12)
                                    pdf.cell(0, 8, f"Histogram: {numeric_cols[0]}", ln=True)
                                    pdf.image(tmpf.name, w=170)
                                    images_added += 1
                                    os.unlink(tmpf.name)
                            except Exception as e:
                                st.write("Could not add histogram to PDF:", e)

                        # Add top categorical bar if categorical columns exist
                        if categorical_cols:
                            try:
                                top_cat = categorical_cols[0]
                                vc = df[top_cat].value_counts().nlargest(10)
                                fig_bar = px.bar(x=vc.index, y=vc.values, labels={'x': top_cat, 'y': 'count'})
                                img_bytes = plotly_fig_to_png_bytes(fig_bar)
                                if img_bytes:
                                    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                                    tmpf.write(img_bytes)
                                    tmpf.flush()
                                    tmpf.close()
                                    pdf.add_page()
                                    pdf.set_font("Arial", "B", 12)
                                    pdf.cell(0, 8, f"Top categories: {top_cat}", ln=True)
                                    pdf.image(tmpf.name, w=170)
                                    images_added += 1
                                    os.unlink(tmpf.name)
                            except Exception as e:
                                st.write("Could not add categorical bar to PDF:", e)

                        # Add correlation heatmap if multiple numeric columns
                        if len(numeric_cols) >= 2:
                            try:
                                fig_mat, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                                png_bytes = fig_to_png_bytes_matplotlib(fig_mat)
                                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                                tmpf.write(png_bytes)
                                tmpf.flush()
                                tmpf.close()
                                pdf.add_page()
                                pdf.set_font("Arial", "B", 12)
                                pdf.cell(0, 8, "Correlation heatmap", ln=True)
                                pdf.image(tmpf.name, w=170)
                                images_added += 1
                                os.unlink(tmpf.name)
                            except Exception as e:
                                st.write("Could not add heatmap to PDF:", e)

                        # Add feature importances if available from best model
                        try:
                            model_obj = best_pipe.named_steps["model"]
                            if hasattr(model_obj, "feature_importances_"):
                                try:
                                    transformers = best_pipe.named_steps['pre'].transformers_
                                    num_names = []
                                    cat_names = []
                                    for t in transformers:
                                        if t[0] == 'num':
                                            num_names = t[2]
                                        if t[0] == 'cat':
                                            cat_input_cols = t[2]
                                            try:
                                                ohe = t[1].named_steps['onehot']
                                                cat_names = list(ohe.get_feature_names_out(cat_input_cols))
                                            except Exception:
                                                cat_names = list(cat_input_cols)
                                    feature_names = list(num_names) + list(cat_names)
                                    fi = model_obj.feature_importances_
                                    fi_df = pd.DataFrame({"feature": feature_names, "importance": fi}).sort_values("importance", ascending=False).head(20)
                                    fig = px.bar(fi_df, x="importance", y="feature", orientation="h", title="Top 20 Feature Importances")
                                    img_bytes = plotly_fig_to_png_bytes(fig)
                                    if img_bytes:
                                        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                                        tmpf.write(img_bytes)
                                        tmpf.flush()
                                        tmpf.close()
                                        pdf.add_page()
                                        pdf.set_font("Arial", "B", 12)
                                        pdf.cell(0, 8, "Feature importances (best model)", ln=True)
                                        pdf.image(tmpf.name, w=170)
                                        images_added += 1
                                        os.unlink(tmpf.name)
                                except Exception as e:
                                    st.write("Could not compute feature importances for PDF:", e)
                        except Exception:
                            pass

                        if images_added == 0:
                            pdf.add_page()
                            pdf.set_font("Arial", "I", 11)
                            pdf.cell(0, 8, "No charts were embedded in the PDF (kaleido or other converters might be missing).", ln=True)

                        # Output PDF bytes and provide download button
                        pdf_out = pdf.output(dest='S').encode('latin-1')
                        st.download_button("⬇️ Download PDF Report (charts + insights)", data=pdf_out, file_name=f"dashboard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", key=f"download_pdf_{time.time()}")
                        st.success("PDF report prepared.")
                    except Exception as e:
                        st.write("PDF generation failed:", e)
                else:
                    st.warning("AutoML couldn't find a working model among candidates.")

    else:
        st.info("Select a target column to run ML models.")

# -----------------------------
# End of script
# -----------------------------
st.success("✅ Dashboard loaded. Use the tabs to explore visuals, insights, summary, and ML.")
