import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🧠 Streamlit Page Config
st.set_page_config(page_title="📊 Data Dashboard Pro", layout="wide")

# 🎨 App Title
st.title("📊 Advanced Data Dashboard v2.0")
st.markdown("Upload your CSV and explore insights with stunning visuals ✨")

# 📤 File Upload
uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File Uploaded Successfully!")

    # 🧾 Dataset Preview
    with st.expander("🔍 Preview Dataset"):
        st.dataframe(df.head())

    # 🧩 Basic Info Cards
    st.subheader("📋 Dataset Overview")
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Numeric Columns", len(df.select_dtypes(include=['int64', 'float64']).columns))
    col5.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))

    # 🎨 Color & Theme Picker
    color_theme = st.sidebar.color_picker("🎨 Pick Chart Color", "#2E86AB")
    sns.set_palette([color_theme])

    # 🧮 Select Columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # 🔍 Filters Section (Slicers)
    st.sidebar.header("🔎 Data Filters")
    selected_cat = st.sidebar.selectbox("Select categorical column for filter", [None] + categorical_cols)
    if selected_cat:
        cat_values = df[selected_cat].unique().tolist()
        selected_values = st.sidebar.multiselect("Select values", cat_values, default=cat_values)
        df = df[df[selected_cat].isin(selected_values)]

    selected_num = st.sidebar.selectbox("Select numeric column for range filter", [None] + numeric_cols)
    if selected_num:
        min_val, max_val = float(df[selected_num].min()), float(df[selected_num].max())
        range_vals = st.sidebar.slider("Select Range", min_val, max_val, (min_val, max_val))
        df = df[(df[selected_num] >= range_vals[0]) & (df[selected_num] <= range_vals[1])]

    # 📊 Chart Selection Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Line Chart", "📊 Bar Chart", "🥧 Pie Chart", "📉 Scatter Plot",
        "📦 Box Plot", "📚 Histogram", "🗺️ Heatmap", "📋 Summary Stats"
    ])

    # Line Chart
    with tab1:
        st.subheader("📈 Line Chart")
        y_axis = st.selectbox("Select numeric column for Y-axis", numeric_cols, key="line_y")
        plt.figure(figsize=(10,5))
        plt.plot(df[y_axis], color=color_theme)
        plt.title(f"Line Chart of {y_axis}")
        st.pyplot(plt)

    # Bar Chart
    with tab2:
        st.subheader("📊 Bar Chart")
        x_axis = st.selectbox("Select category for X-axis", categorical_cols, key="bar_x")
        y_axis = st.selectbox("Select numeric for Y-axis", numeric_cols, key="bar_y")
        plt.figure(figsize=(10,5))
        sns.barplot(x=df[x_axis], y=df[y_axis], color=color_theme)
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Pie Chart
    with tab3:
        st.subheader("🥧 Pie Chart")
        cat_col = st.selectbox("Select categorical column", categorical_cols, key="pie_cat")
        pie_data = df[cat_col].value_counts()
        plt.figure(figsize=(6,6))
        plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
        plt.title(f"Pie Chart of {cat_col}")
        st.pyplot(plt)

    # Scatter Plot
    with tab4:
        st.subheader("📉 Scatter Plot")
        x_axis = st.selectbox("Select X-axis", numeric_cols, key="scatter_x")
        y_axis = st.selectbox("Select Y-axis", numeric_cols, key="scatter_y")
        plt.figure(figsize=(8,5))
        sns.scatterplot(x=df[x_axis], y=df[y_axis], color=color_theme)
        plt.title(f"Scatter Plot: {x_axis} vs {y_axis}")
        st.pyplot(plt)

    # Box Plot
    with tab5:
        st.subheader("📦 Box Plot")
        num_col = st.selectbox("Select numeric column", numeric_cols, key="box_num")
        plt.figure(figsize=(6,5))
        sns.boxplot(y=df[num_col], color=color_theme)
        plt.title(f"Boxplot of {num_col}")
        st.pyplot(plt)

    # Histogram
    with tab6:
        st.subheader("📚 Histogram")
        num_col = st.selectbox("Select column for histogram", numeric_cols, key="hist_num")
        plt.figure(figsize=(8,5))
        plt.hist(df[num_col], bins=20, color=color_theme, edgecolor='black')
        plt.title(f"Distribution of {num_col}")
        st.pyplot(plt)

    # Heatmap
    with tab7:
        st.subheader("🗺️ Correlation Heatmap")
        plt.figure(figsize=(10,6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)

    # Summary
    with tab8:
        st.subheader("📋 Statistical Summary")
        st.write(df.describe())

else:
    st.info("📤 Upload a CSV file from sidebar to start exploring.")
