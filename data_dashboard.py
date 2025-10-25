"""
📊 Pro Data Dashboard — Phase 1 + Phase 2 + PHASE 3 (AutoML & Clustering)
- Upload CSV/Excel
- Charts, Filters, Insights
- ML tab: Auto train baseline models, show metrics, download model
- Clustering: KMeans
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
import io
import pickle
from datetime import datetime

# sklearn imports for Phase 3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score, f1_score, roc_auc_score, confusion_matrix,
                             silhouette_score)

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
    total_missing = int(df.isnull().sum().sum())
    if total_missing > 0:
        insights.append(f"⚠️ There are {total_missing} missing values across the dataset.")
    else:
        insights.append("✅ No missing values detected.")
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] >= 2:
        corr = numeric.corr().abs()
        corr_unstack = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        if not corr_unstack.empty:
            top_pair = corr_unstack.idxmax()
            top_val = corr_unstack.max()
            if top_val > 0.8:
                insights.append(f"🔗 Strong correlation ({top_val:.2f}) between `{top_pair[0]}` and `{top_pair[1]}`.")
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

def metric_regression(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return {"R2": r2, "RMSE": rmse, "MAE": mae}

def metric_classification(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    roc = None
    if y_proba is not None:
        try:
            # If multiclass, roc_auc_score needs multi_class param; handle binary primarily
            if len(np.unique(y_true)) == 2:
                roc = roc_auc_score(y_true, y_proba[:,1])
            else:
                roc = roc_auc_score(y_true, y_proba, multi_class='ovo', average='weighted')
        except Exception:
            roc = None
    return {"Accuracy": acc, "F1": f1, "ROC-AUC": roc}

def prepare_preprocessor(X):
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    # numeric pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    # categorical pipeline
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))

    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, cat_cols)
    ], remainder='drop')
    return preprocessor, numeric_cols, cat_cols

def limit_dataframe_size(df, max_rows=50000):
    if len(df) > max_rows:
        st.warning(f"Dataset has {len(df):,} rows — sampling {max_rows} rows for faster ML.")
        return df.sample(max_rows, random_state=42).reset_index(drop=True)
    return df

# -------------------------
# Sidebar - controls & upload
# -------------------------
st.sidebar.header("🔧 Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xls", "xlsx"])

st.sidebar.markdown("### 🎨 Visual Settings")
chart_color = st.sidebar.color_picker("Pick main chart color", "#2E86AB")
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], index=0)

st.sidebar.markdown("---")
if st.sidebar.button("Save Dashboard Settings"):
    s = {"chart_color": chart_color, "theme": theme}
    save_settings(s)
    st.sidebar.success("Settings saved.")

if st.sidebar.button("Load Dashboard Settings"):
    s = load_settings()
    if s:
        st.sidebar.info("Settings loaded (apply manually).")
    else:
        st.sidebar.warning("No saved settings found.")

# -------------------------
# Main Layout
# -------------------------
st.title("📊 Pro Data Dashboard Edition")
st.markdown("Upload a dataset to explore, visualize and run quick ML experiments.")

if uploaded_file:
    try:
        df = read_file(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    # safe copy
    df_original = df.copy()
    st.success(f"Loaded `{uploaded_file.name}` — {df.shape[0]} rows × {df.shape[1]} cols")

    # basic column detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    # Tabs (add ML tab)
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "🧠 Insights", "📋 Summary", "🤖 ML (Phase 3)"])

    # --------------------------
    # TAB 1: Dashboard (same as before)
    # --------------------------
    with tab1:
        col1, col2, col3, col4, col5 = st.columns([1.2]*5)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing values", int(df.isnull().sum().sum()))
        col4.metric("Numeric cols", len(numeric_cols))
        col5.metric("Categorical cols", len(categorical_cols))

        with st.expander("🧠 Quick Insights"):
            for i in small_insights(df):
                st.write(i)

        # Filters in sidebar (same pattern, using original df_original for ranges)
        st.sidebar.header("🔎 Filters (Slicers)")
        selected_cat_col = st.sidebar.selectbox("Categorical Column (filter)", [None] + categorical_cols, key="s_cat")
        if selected_cat_col:
            unique_vals = df[selected_cat_col].dropna().unique().tolist()
            chosen_vals = st.sidebar.multiselect(f"Values for `{selected_cat_col}`", unique_vals, default=unique_vals[:5], key="s_cat_vals")
            if chosen_vals:
                df = df[df[selected_cat_col].isin(chosen_vals)]

        selected_num_col = st.sidebar.selectbox("Numeric Column (range filter)", [None] + numeric_cols, key="s_num")
        if selected_num_col:
            min_val = float(df_original[selected_num_col].min())
            max_val = float(df_original[selected_num_col].max())
            lo, hi = st.sidebar.slider(f"Range for `{selected_num_col}`", min_val, max_val, (min_val, max_val), key="s_num_range")
            df = df[(df[selected_num_col] >= lo) & (df[selected_num_col] <= hi)]

        if datetime_cols:
            date_col = st.sidebar.selectbox("Date column filter (optional)", [None] + datetime_cols, key="s_date")
            if date_col:
                start_date = st.sidebar.date_input("Start date", key="s_start")
                end_date = st.sidebar.date_input("End date", key="s_end")
                df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]

        left, right = st.columns((3,1))
        with right:
            st.subheader("📁 Data Preview & Export")
            if st.checkbox("Show filtered data", value=False, key="show_filtered"):
                st.dataframe(df.head(200))
            csv_bytes = download_df_as_csv(df)
            st.download_button("⬇️ Download Filtered CSV", csv_bytes, file_name=f"filtered_{uploaded_file.name}", key="download_filtered")

            if numeric_cols:
                st.markdown("### Aggregations (Numeric)")
                sel = st.selectbox("Choose numeric column", numeric_cols, key="agg_select_main")
                st.write(df[sel].agg(['mean','median','std','min','max']).to_frame(name=sel))

        with left:
            st.subheader("🎯 Chart Builder")
            chart_type = st.selectbox("Select chart type",
                                      ["Auto Recommendation","Line","Bar","Pie","Scatter","Box","Histogram","Area","Heatmap","Pairplot","Top N Categories"],
                                      key="chart_type_main")

            # implement charts with unique keys (same approach as Phase 2)
            if chart_type == "Line":
                if numeric_cols:
                    y = st.selectbox("Y-axis (numeric)", numeric_cols, key="line_y_main")
                    x = st.selectbox("X-axis (optional)", [None] + categorical_cols + datetime_cols, key="line_x_main")
                    if st.button("Generate Line Chart", key="btn_line"):
                        fig = px.line(df, x=x, y=y) if x else px.line(df, y=y)
                        fig.update_traces(line_color=chart_color)
                        st.plotly_chart(fig, use_container_width=True, key="line_chart_main")

            elif chart_type == "Bar":
                if categorical_cols and numeric_cols:
                    x = st.selectbox("Category (X)", categorical_cols, key="bar_x_main")
                    y = st.selectbox("Value (Y)", numeric_cols, key="bar_y_main")
                    agg = st.selectbox("Aggregation", ["sum","mean","median","count"], index=1, key="bar_agg_main")
                    if st.button("Generate Bar Chart", key="btn_bar"):
                        agg_df = df.groupby(x)[y].agg(agg).reset_index().sort_values(by=y, ascending=False)
                        fig = px.bar(agg_df, x=x, y=y)
                        fig.update_traces(marker_color=chart_color)
                        st.plotly_chart(fig, use_container_width=True, key="bar_chart_main")

            elif chart_type == "Auto Recommendation":
                st.info("Auto-chart: histogram + top categorical bar")
                if numeric_cols:
                    fig = px.histogram(df, x=numeric_cols[0])
                    st.plotly_chart(fig, use_container_width=True, key="auto_hist_main")
                if categorical_cols:
                    vc = df[categorical_cols[0]].value_counts().nlargest(10)
                    fig = px.bar(x=vc.index, y=vc.values)
                    st.plotly_chart(fig, use_container_width=True, key="auto_bar_main")

            # Generate All (with unique keys)
            st.markdown("---")
            if st.button("🔁 Generate All Recommended Charts", key="btn_gen_all"):
                with st.spinner("Generating charts..."):
                    cols = st.columns(2)
                    if numeric_cols:
                        fig1 = px.histogram(df, x=numeric_cols[0], nbins=30)
                        cols[0].plotly_chart(fig1, use_container_width=True, key="gen_all_hist")
                    if categorical_cols:
                        vc = df[categorical_cols[0]].value_counts().nlargest(10)
                        fig2 = px.bar(x=vc.index, y=vc.values)
                        cols[1].plotly_chart(fig2, use_container_width=True, key="gen_all_bar")
                    if len(numeric_cols) >= 2:
                        fig3 = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
                        cols[0].plotly_chart(fig3, use_container_width=True, key="gen_all_scatter")
                    if len(numeric_cols) >= 1:
                        fig4 = px.box(df, y=numeric_cols[0])
                        cols[1].plotly_chart(fig4, use_container_width=True, key="gen_all_box")
                st.success("✅ All charts generated successfully!")

    # --------------------------
    # TAB 2: Insights (Phase 2)
    # --------------------------
    with tab2:
        st.subheader("🧠 Smart Dataset Insights")
        ins = small_insights(df_original)
        for i in ins:
            st.write(i)

        # Outlier detection quick
        st.markdown("### 📊 Outlier detection (IQR method)")
        outlier_report = {}
        for col in numeric_cols:
            q1, q3 = df_original[col].quantile(0.25), df_original[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            count = ((df_original[col] < lower) | (df_original[col] > upper)).sum()
            outlier_report[col] = int(count)
        if outlier_report:
            st.dataframe(pd.DataFrame(outlier_report.items(), columns=["Column", "Outlier Count"]))
        else:
            st.info("No numeric columns to check for outliers.")

        # Category dominance
        st.markdown("### 🏷️ Category dominance")
        cat_report = []
        for col in categorical_cols:
            vc = df_original[col].value_counts()
            if not vc.empty:
                top_cat, pct = vc.index[0], vc.iloc[0]/len(df_original)*100
                cat_report.append((col, top_cat, round(pct,1)))
        if cat_report:
            st.dataframe(pd.DataFrame(cat_report, columns=["Column","Top Category","% Share"]))
        else:
            st.info("No categorical columns found.")

    # --------------------------
    # TAB 3: Summary
    # --------------------------
    with tab3:
        st.subheader("📋 Dataset Summary")
        st.write(df_original.describe(include='all').transpose())
        if len(numeric_cols) >= 2:
            st.subheader("🔗 Top Correlations")
            corr = df_original[numeric_cols].corr().abs()
            corr_un = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
            if not corr_un.empty:
                top_corr = corr_un.sort_values(ascending=False).head(10)
                st.write(top_corr)
            else:
                st.write("No correlations found.")

    # --------------------------
    # TAB 4: ML (Phase 3)
    # --------------------------
    with tab4:
        st.header("🤖 Quick ML Studio — Train baseline models")
        st.markdown("Choose target column and run baseline regression/classification models. Use modest dataset sizes for speed.")

        # Limit dataset size for ML steps
        df_ml_full = limit_dataframe_size(df_original, max_rows=50000)

        st.markdown("### 1) Select target (label)")
        target = st.selectbox("Target column", options=[None] + df_ml_full.columns.tolist(), key="ml_target")
        if target:
            # Determine task type heuristic
            n_unique = df_ml_full[target].nunique(dropna=True)
            is_numeric_target = pd.api.types.is_numeric_dtype(df_ml_full[target])
            # Heuristic: numeric with many unique values -> regression, otherwise classification
            if is_numeric_target and n_unique > 20:
                task = "regression"
            else:
                task = "classification"
            st.info(f"Auto-detected task: **{task}** (target unique values: {n_unique})")

            # Feature selection (exclude target and datetime)
            features = st.multiselect("Features (leave empty to use all except target)", options=[c for c in df_ml_full.columns if c != target], default=[c for c in df_ml_full.columns if c != target][:10], key="ml_features")
            if not features:
                st.warning("Select at least one feature.")
            else:
                X = df_ml_full[features].copy()
                y = df_ml_full[target].copy()

                # Basic drop rows where target is null
                non_null_idx = ~y.isnull()
                X = X.loc[non_null_idx].reset_index(drop=True)
                y = y.loc[non_null_idx].reset_index(drop=True)

                st.markdown("### 2) Preprocessing & split")
                test_size = st.slider("Test set size (fraction)", 0.05, 0.5, 0.2, step=0.05, key="ml_test_size")
                random_state = 42

                preprocessor, num_cols, cat_cols = prepare_preprocessor(X)

                # Train/test split (stratify if classification and not too many classes)
                stratify = None
                if task == "classification":
                    if y.nunique() <= 10:
                        try:
                            stratify = y
                        except Exception:
                            stratify = None

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

                st.write(f"Train shape: {X_train.shape} — Test shape: {X_test.shape}")

                # Choose models to evaluate
                st.markdown("### 3) Models to run (baseline)")
                run_lr = st.checkbox("Run Linear/Logistic model", value=True, key="run_lr")
                run_rf = st.checkbox("Run RandomForest model", value=True, key="run_rf")

                # We'll build pipelines
                models_results = {}

                if run_lr:
                    if task == "regression":
                        pipe_lr = Pipeline([("pre", preprocessor), ("model", LinearRegression())])
                        pipe_lr.fit(X_train, y_train)
                        preds = pipe_lr.predict(X_test)
                        metrics = metric_regression(y_test, preds)
                        models_results["LinearRegression"] = {"pipeline": pipe_lr, "metrics": metrics}
                    else:
                        # classification (use max_iter increased)
                        pipe_lr = Pipeline([("pre", preprocessor), ("model", LogisticRegression(max_iter=200))])
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

                if run_rf:
                    if task == "regression":
                        pipe_rf = Pipeline([("pre", preprocessor), ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
                        pipe_rf.fit(X_train, y_train)
                        preds = pipe_rf.predict(X_test)
                        metrics = metric_regression(y_test, preds)
                        # compute feature importances (map back if onehot used)
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

                # Display results
                st.markdown("### 4) Model results")
                if not models_results:
                    st.info("No models ran. Select at least one model checkbox.")
                else:
                    for name, res in models_results.items():
                        st.markdown(f"#### {name}")
                        m = res["metrics"]
                        # show metrics as key: value
                        for k, v in m.items():
                            if v is None:
                                st.write(f"- {k}: N/A")
                            elif isinstance(v, float):
                                st.write(f"- {k}: **{v:.4f}**")
                            else:
                                st.write(f"- {k}: **{v}**")

                        # If regression and model is forest -> show feature importances
                        model_obj = res["pipeline"].named_steps["model"]
                        if hasattr(model_obj, "feature_importances_"):
                            st.markdown("**Top feature importances (RandomForest)**")
                            # Map feature importances back — need to get transformed feature names
                            try:
                                # build transform to get column names from preprocessor
                                pre = res["pipeline"].named_steps["pre"]
                                # numeric names
                                num_names = num_cols
                                # categorical onehot names
                                cat_names = []
                                if cat_cols:
                                    ohe = pre.named_transformers_['cat'].named_steps['onehot']
                                    cat_ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
                                    cat_names = cat_ohe_names
                                feature_names = num_names + cat_names
                                importances = model_obj.feature_importances_
                                fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
                                fi_df = fi_df.sort_values("importance", ascending=False).head(20)
                                st.dataframe(fi_df.reset_index(drop=True))
                                # bar chart
                                fig = px.bar(fi_df, x="importance", y="feature", orientation="h")
                                st.plotly_chart(fig, use_container_width=True, key=f"fi_{name}")
                            except Exception as e:
                                st.write("Feature importance mapping failed:", e)

                        # If classification — confusion matrix and ROC (if proba)
                        if task == "classification":
                            try:
                                cm = confusion_matrix(y_test, res["pipeline"].predict(X_test))
                                st.write("Confusion matrix:")
                                fig, ax = plt.subplots()
                                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                                st.pyplot(fig, key=f"cm_{name}")
                            except Exception:
                                pass
                            # ROC curve if proba available & binary
                            try:
                                proba = None
                                if hasattr(res["pipeline"], "predict_proba"):
                                    proba = res["pipeline"].predict_proba(X_test)
                                if proba is not None and y_test.nunique() == 2:
                                    from sklearn.metrics import roc_curve, auc
                                    fpr, tpr, _ = roc_curve(y_test, proba[:,1])
                                    roc_auc = auc(fpr, tpr)
                                    st.write(f"ROC AUC: {roc_auc:.4f}")
                                    fig, ax = plt.subplots()
                                    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
                                    ax.plot([0,1], [0,1], linestyle="--", color="grey")
                                    ax.set_xlabel("False Positive Rate")
                                    ax.set_ylabel("True Positive Rate")
                                    ax.legend()
                                    st.pyplot(fig, key=f"roc_{name}")
                            except Exception:
                                pass

                        # Download trained pipeline
                        buf = io.BytesIO()
                        try:
                            pickle.dump(res["pipeline"], buf)
                            buf.seek(0)
                            st.download_button(f"⬇️ Download `{name}` model (.pkl)", data=buf, file_name=f"{name}.pkl", key=f"download_model_{name}")
                        except Exception as e:
                            st.write("Model download failed:", e)

                # --------------------
                # Clustering (optional)
                # --------------------
                st.markdown("---")
                st.subheader("🔎 Optional: Clustering (KMeans)")
                run_cluster = st.checkbox("Run KMeans clustering on selected numeric features", value=False, key="run_kmeans")
                if run_cluster:
                    cluster_features = st.multiselect("Choose numeric features for clustering", options=X.select_dtypes(include=[np.number]).columns.tolist(), default=X.select_dtypes(include=[np.number]).columns.tolist()[:3], key="cluster_features")
                    if len(cluster_features) < 1:
                        st.warning("Select at least one numeric feature.")
                    else:
                        # sample for plotting if large
                        X_cluster = X[cluster_features].dropna()
                        if len(X_cluster) > 5000:
                            X_cluster = X_cluster.sample(5000, random_state=42)
                            st.info("Sampling 5000 rows for clustering plot.")
                        # scale
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
                        # 2D plot via PCA if >2 dims
                        if X_scaled.shape[1] > 2:
                            from sklearn.decomposition import PCA
                            pca = PCA(n_components=2)
                            proj = pca.fit_transform(X_scaled)
                            plot_df = pd.DataFrame({"x": proj[:,0], "y": proj[:,1], "cluster": labels})
                            fig = px.scatter(plot_df, x="x", y="y", color="cluster", title="KMeans clusters (PCA projection)")
                            st.plotly_chart(fig, use_container_width=True, key="kmeans_plot")
                        else:
                            plot_df = pd.DataFrame(X_cluster.reset_index(drop=True))
                            plot_df["cluster"] = labels
                            if X_cluster.shape[1] == 2:
                                cols_plot = X_cluster.columns.tolist()
                                fig = px.scatter(plot_df, x=cols_plot[0], y=cols_plot[1], color="cluster", title="KMeans clusters")
                                st.plotly_chart(fig, use_container_width=True, key="kmeans_plot_2d")
                            else:
                                st.write("Not enough dims for easy plotting.")

else:
    st.info("Upload a CSV or Excel file from the sidebar to start the Pro Dashboard.")
