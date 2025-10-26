# modules/insights.py
"""
Insights Module for Data Visualizer Dashboard
---------------------------------------------
This module generates textual insights for a given dataset and chart type.

Functions:
---------
1. generate_insights(df, chart_type): Returns insights HTML for the dashboard.
"""

import pandas as pd
from dash import html

def generate_insights(df: pd.DataFrame, chart_type: str) -> html.Div:
    """
    Generate textual insights for the given dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset for which insights are to be generated.
    chart_type : str
        Type of chart selected by user (optional for context).

    Returns:
    --------
    insights_div : html.Div
        Dash HTML Div containing textual insights.
    """
    if df.empty:
        return html.Div("Dataset is empty. No insights available.", className="text-danger")
    
    insights = []

    # Basic info
    insights.append(html.H6("📊 Dataset Overview"))
    insights.append(html.P(f"Number of rows: {df.shape[0]}"))
    insights.append(html.P(f"Number of columns: {df.shape[1]}"))

    # Column-wise insights
    for col in df.columns:
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        missing_count = df[col].isna().sum()
        info_text = f"Column '{col}' | Type: {dtype} | Unique: {unique_count} | Missing: {missing_count}"

        # Numeric specific stats
        if pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            median = df[col].median()
            min_val = df[col].min()
            max_val = df[col].max()
            info_text += f" | Min: {min_val}, Max: {max_val}, Mean: {mean:.2f}, Median: {median:.2f}"
        
        # Categorical specific stats
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            top = df[col].mode().values[0] if not df[col].mode().empty else "N/A"
            info_text += f" | Most Frequent: {top}"

        insights.append(html.P(info_text))
    
    # Summary suggestion based on chart type
    if chart_type:
        insights.append(html.Hr())
        insights.append(html.H6("💡 Chart-specific Suggestion"))
        if chart_type in ["bar", "pie", "donut"]:
            insights.append(html.P("Categorical analysis recommended."))
        elif chart_type in ["line", "scatter", "area", "bubble"]:
            insights.append(html.P("Trend analysis for numeric data recommended."))
        elif chart_type in ["box", "hist", "heatmap"]:
            insights.append(html.P("Distribution analysis for numeric columns recommended."))
    
    return html.Div(insights, className="p-3 border rounded bg-light")
