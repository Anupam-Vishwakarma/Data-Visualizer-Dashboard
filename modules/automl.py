# modules/automl.py
"""
AutoML Module for Data Visualizer Dashboard
-------------------------------------------
This module analyzes the given dataset and provides automatic recommendations 
for the most suitable visualization type based on the dataset's characteristics.

Functions:
---------
1. suggest_visual(df): Returns a recommended chart type (string) for a given DataFrame.
"""

import pandas as pd

def suggest_visual(df: pd.DataFrame) -> str:
    """
    Analyze the dataset and suggest an appropriate visualization type.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset for which visualization recommendation is needed.

    Returns:
    --------
    recommendation : str
        Suggested chart type as a string.
        Options: 'bar', 'line', 'pie', 'scatter', 'box', 'hist', 'area', 'donut', 'heatmap', 'bubble'
    """
    # Basic checks
    if df.empty:
        return "No data available to suggest a chart."
    
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    # Recommendation logic
    if len(num_cols) == 0 and len(cat_cols) > 0:
        # Only categorical columns
        if len(cat_cols) == 1:
            return "pie"  # single categorical column → Pie Chart
        else:
            return "bar"  # multiple categorical columns → Bar Chart
    elif len(num_cols) == 1 and len(cat_cols) == 0:
        # Single numeric column → Histogram or Box Plot
        return "hist"
    elif len(num_cols) == 2 and len(cat_cols) == 0:
        # Two numeric columns → Scatter Plot
        return "scatter"
    elif len(num_cols) >= 2 and len(cat_cols) == 0:
        # Multiple numeric columns → Line Chart
        return "line"
    elif len(cat_cols) >= 1 and len(num_cols) >= 1:
        # Mixed columns → Bar Chart or Box Plot
        return "bar"
    else:
        # Fallback
        return "bar"
