"""
Pro Data Dashboard Edition (Streamlit)
Features:
 - Upload CSV/Excel
 - Auto-detect numeric & categorical columns
 - Multiple charts (Line, Bar, Pie, Scatter, Box, Histogram, Area, Heatmap)
 - Sidebar slicers (categorical multi-select + numeric range)
 - Color picker, theme switch, stat cards
 - Generate all charts, download filtered data, save/load settings
 - Simple rule-based dataset summary (insights)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
import io
from datetime import datetime

st.set_page_config(page_title="📊 Pro Data Dashboard", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Utility functions
# -------------------------
def read_file(uploaded):
    if uploaded.name.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded, engine="openpyxl")
    else:
        return pd.read_csv(uploaded)

def download_df_as_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def small_insights(df):
    insights = []
    # missing
    total_missing = int(df.isnull().sum().sum())
    if total_missing > 0:
        insights.append(f"⚠️ There are {total_missing} missing values across the dataset.")
    else:
        insights.append("✅ No missing values detected.")
    # high correlations
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] >= 2:
        corr = numeric.corr().abs()
        # find top correlated pair
        corr_unstack = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        if not corr_unstack.empty:
            top_pair = corr_unstack.idxmax()
            top_val = corr_unstack.max()
            if top_val > 0.8:
                insights.append(f"🔗 Strong correlation ({top_val:.2f}) between `{top_pair[0]}` and `{top_pair[1]}`.")
    # top categories
    cats = df.select_dtypes(include=['object', 'category'])
    for c in cats.columns[:3]:
        vc = df[c].value_counts()
        if not vc.empty:
            top = vc.index[0]
            pct = vc.iloc[0] / len(df) * 100
            insights.append(f"🏷️ Column `{c}` dominated by `{top}` ({pct:.1f}%).")
    return insights

def save_settings(settings: dict, filename="dashboard_settings.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

def load_settings(filename="dashboard_settings.json"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def plotly_to_png(fig):
    # return PNG bytes of a plotly figure
    buf = io.BytesIO()
    fig.write_image(buf, format="png", scale=2)
    buf.seek(0)
    return buf

# -------------------------
# Sidebar - controls & upload
# -------------------------
st.sidebar.header("🔧 Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xls", "xlsx"])

# theme / color
st.sidebar.markdown("### 🎨 Visual Settings")
chart_color = st.sidebar.color_picker("Pick main chart color", "#2E86AB")
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], index=0)

# settings save / load
st.sidebar.markdown("---")
if st.sidebar.button("Save Dashboard Settings"):
    s = {
        "chart_color": chart_color,
        "theme": theme
    }
    save_settings(s)
    st.sidebar.success("Settings saved.")

if st.sidebar.button("Load Dashboard Settings"):
    s = load_settings()
    if s:
        # note: can't programmatically set color_picker or selectbox easily; show message
        st.sidebar.info("Settings loaded (manually apply if needed).")
    else:
        st.sidebar.warning("No saved settings found.")

# -------------------------
# Main App
# -------------------------
st.title("📊 Pro Data Dashboard Edition")
st.markdown("Upload a dataset to automatically explore and visualize. Use the sidebar filters and color picker to customize charts.")

if uploaded_file:
    try:
        df = read_file(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    # basic cleaning/copy
    df_original = df.copy()
    st.success(f"Loaded `{uploaded_file.name}` — {df.shape[0]} rows × {df.shape[1]} cols")

    # top stat cards
    col1, col2, col3, col4, col5 = st.columns([1.2,1.2,1.2,1.2,1.2])
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing values", int(df.isnull().sum().sum()))
    col4.metric("Numeric cols", len(df.select_dtypes(include=[np.number]).columns))
    col5.metric("Categorical cols", len(df.select_dtypes(include=['object','category']).columns))

    # show quick insights
    with st.expander("🧠 Quick Insights"):
        ins = small_insights(df)
        for i in ins:
            st.write(i)

    # identify cols
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    # Filter UI
    st.sidebar.header("🔎 Filters (Slicers)")
    # categorical slicer
    selected_cat_col = st.sidebar.selectbox("Categorical Column (filter)", [None] + categorical_cols)
    if selected_cat_col:
        unique_vals = df[selected_cat_col].dropna().unique().tolist()
        chosen_vals = st.sidebar.multiselect(f"Values for `{selected_cat_col}`", unique_vals, default=unique_vals[:5])
        if chosen_vals:
            df = df[df[selected_cat_col].isin(chosen_vals)]

    # numeric slicer
    selected_num_col = st.sidebar.selectbox("Numeric Column (range filter)", [None] + numeric_cols)
    if selected_num_col:
        min_val = float(df_original[selected_num_col].min())
        max_val = float(df_original[selected_num_col].max())
        lo, hi = st.sidebar.slider(f"Range for `{selected_num_col}`", min_val, max_val, (min_val, max_val))
        df = df[(df[selected_num_col] >= lo) & (df[selected_num_col] <= hi)]

    # date filter (if any)
    if datetime_cols:
        date_col = st.sidebar.selectbox("Date column filter (optional)", [None] + datetime_cols)
        if date_col:
            start_date = st.sidebar.date_input("Start date")
            end_date = st.sidebar.date_input("End date")
            df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]

    # main layout: left = charts, right = controls + preview
    left, right = st.columns((3,1))

    # right column: preview + download + data table toggle
    with right:
        st.subheader("📁 Data Preview & Export")
        if st.checkbox("Show filtered data", value=False):
            st.dataframe(df.head(200))
        csv_bytes = download_df_as_csv(df)
        st.download_button("⬇️ Download Filtered CSV", csv_bytes, file_name=f"filtered_{uploaded_file.name}")

        # basic aggregation quick stats
        if numeric_cols:
            st.markdown("### Aggregations (Selected numeric)")
            sel = st.selectbox("Choose numeric for quick aggregations", numeric_cols, key="agg_select")
            st.write(df[sel].agg(['mean','median','std','min','max']).to_frame(name=sel))

    # chart controls in left top
    with left:
        st.subheader("🎯 Chart Builder")
        chart_type = st.selectbox("Select chart type",
                                  ["Auto Recommendation","Line","Bar","Pie","Scatter","Box","Histogram","Area","Heatmap","Pairplot","Top N Categories"])
        # choose columns depending on type
        if chart_type == "Line":
            if not numeric_cols:
                st.warning("No numeric columns available.")
            else:
                y = st.selectbox("Y-axis (numeric)", numeric_cols, key="line_y")
                x = st.selectbox("X-axis (optional)", [None] + (categorical_cols + datetime_cols), key="line_x")
                if st.button("Generate Line Chart"):
                    if x:
                        fig = px.line(df, x=x, y=y, color=None)
                    else:
                        fig = px.line(df, y=y)
                    fig.update_traces(line_color=chart_color)
                    st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar":
            if not categorical_cols or not numeric_cols:
                st.warning("Need at least one categorical and one numeric column.")
            else:
                x = st.selectbox("Category (X)", categorical_cols, key="bar_x")
                y = st.selectbox("Value (Y)", numeric_cols, key="bar_y")
                agg = st.selectbox("Aggregation", ["sum","mean","median","count"], index=1)
                if st.button("Generate Bar Chart"):
                    agg_df = df.groupby(x)[y].agg(agg).reset_index().sort_values(by=y, ascending=False)
                    fig = px.bar(agg_df, x=x, y=y)
                    fig.update_traces(marker_color=chart_color)
                    st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Pie":
            if not categorical_cols:
                st.warning("Need a categorical column.")
            else:
                cat = st.selectbox("Category for Pie", categorical_cols, key="pie_cat")
                top_n = st.slider("Top N categories (others grouped)", 2, min(20, max(2, len(df[cat].unique()))), 6)
                if st.button("Generate Pie Chart"):
                    vc = df[cat].value_counts()
                    top = vc.nlargest(top_n)
                    others = vc.iloc[top_n:].sum()
                    labels = list(top.index) + (["Others"] if others>0 else [])
                    values = list(top.values) + ([others] if others>0 else [])
                    fig = px.pie(values=values, names=labels)
                    st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter":
            if len(numeric_cols) < 2:
                st.warning("At least two numeric columns needed.")
            else:
                x = st.selectbox("X-axis (numeric)", numeric_cols, key="scatter_x")
                y = st.selectbox("Y-axis (numeric)", numeric_cols, key="scatter_y")
                color_col = st.selectbox("Color (optional, categorical)", [None] + categorical_cols, key="scatter_color")
                if st.button("Generate Scatter Plot"):
                    fig = px.scatter(df, x=x, y=y, color=color_col)
                    st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Box":
            if not numeric_cols:
                st.warning("Need numeric columns.")
            else:
                col = st.selectbox("Numeric for Boxplot", numeric_cols, key="box_col")
                if st.button("Generate Box Plot"):
                    fig = px.box(df, y=col)
                    st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Histogram":
            if not numeric_cols:
                st.warning("Need numeric columns.")
            else:
                col = st.selectbox("Numeric for Histogram", numeric_cols, key="hist_col")
                bins = st.slider("Bins", 5, 100, 20)
                if st.button("Generate Histogram"):
                    fig = px.histogram(df, x=col, nbins=bins)
                    fig.update_traces(marker_color=chart_color)
                    st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Area":
            if not numeric_cols:
                st.warning("Need numeric columns.")
            else:
                cols = st.multiselect("Numeric columns for Area (stacked)", numeric_cols, default=numeric_cols[:2])
                if st.button("Generate Area Chart"):
                    if cols:
                        fig = px.area(df, y=cols)
                        st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Heatmap":
            if len(numeric_cols) < 2:
                st.warning("Need at least two numeric columns.")
            else:
                if st.button("Generate Heatmap"):
                    fig, ax = plt.subplots(figsize=(10,6))
                    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)

        elif chart_type == "Pairplot":
            if len(numeric_cols) < 2:
                st.warning("Need at least two numeric columns.")
            else:
                sel = st.multiselect("Select numeric columns for pairplot", numeric_cols, default=numeric_cols[:3])
                if st.button("Generate Pairplot"):
                    fig = sns.pairplot(df[sel].dropna().sample(min(500, len(df))))
                    st.pyplot(fig)

        elif chart_type == "Top N Categories":
            if not categorical_cols:
                st.warning("No categorical columns.")
            else:
                cat = st.selectbox("Category column", categorical_cols, key="topn_cat")
                topn = st.slider("Top N", 2, 30, 10)
                if st.button("Show Top N"):
                    vc = df[cat].value_counts().nlargest(topn)
                    fig = px.bar(x=vc.index, y=vc.values, labels={'x':cat,'y':'count'})
                    st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Auto Recommendation":
            # pick best numeric and categorical automatically
            st.info("Auto-chart: Showing histogram of top numeric column and a bar of top categorical column (if any).")
            if numeric_cols:
                top_num = numeric_cols[0]
                fig = px.histogram(df, x=top_num)
                st.plotly_chart(fig, use_container_width=True)
            if categorical_cols:
                top_cat = categorical_cols[0]
                vc = df[top_cat].value_counts().nlargest(10)
                fig = px.bar(x=vc.index, y=vc.values)
                st.plotly_chart(fig, use_container_width=True)

        # Generate All Charts button
        st.markdown("---")
        if st.button("🔁 Generate All Recommended Charts"):
            # generate a set of charts
            with st.spinner("Generating charts..."):
                # Show 2x2 grid of common visuals
                cols = st.columns(2)
                if numeric_cols:
                    # Histogram of first numeric
                    fig1 = px.histogram(df, x=numeric_cols[0], nbins=30)
                    cols[0].plotly_chart(fig1, use_container_width=True)
                if categorical_cols:
                    vc = df[categorical_cols[0]].value_counts().nlargest(10)
                    fig2 = px.bar(x=vc.index, y=vc.values)
                    cols[1].plotly_chart(fig2, use_container_width=True)
                if len(numeric_cols) >= 2:
                    fig3 = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
                    cols[0].plotly_chart(fig3, use_container_width=True)
                if len(numeric_cols) >= 1:
                    fig4 = px.box(df, y=numeric_cols[0])
                    cols[1].plotly_chart(fig4, use_container_width=True)
            st.success("✅ All charts generated.")

    # bottom: dataset summary and correlations
    st.markdown("---")
    st.subheader("📋 Dataset Summary")
    st.write(df.describe(include='all').transpose())

    # Quick correlation table
    if len(numeric_cols) >= 2:
        st.subheader("🔗 Top Correlations")
        corr = df[numeric_cols].corr().abs()
        corr_un = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        if not corr_un.empty:
            top_corr = corr_un.sort_values(ascending=False).head(10)
            st.write(top_corr)
        else:
            st.write("No correlations found.")

else:
    st.info("Upload a CSV or Excel file from the sidebar to start the Pro Dashboard.")
import streamlit as st
import numpy as np
def show_summary_correlation(df):
    st.markdown("---")
    st.subheader("📋 Dataset Summary")
    st.write(df.describe(include='all').transpose())

    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) >= 2:
        st.subheader("🔗 Top Correlations")
        corr = df[numeric_cols].corr().abs()
        corr_un = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        if not corr_un.empty:
            top_corr = corr_un.sort_values(ascending=False).head(10)
            st.write(top_corr)
        else:
            st.write("No correlations found.")

