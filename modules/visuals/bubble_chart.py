from dash import dcc, html
import plotly.express as px
import pandas as pd

def create_bubble_chart(df: pd.DataFrame, x_col: str = None, y_col: str = None, size_col: str = None, color_col: str = None):
    """
    Generate a Bubble Chart using Plotly Express.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to plot.
    x_col : str, optional
        Column name for x-axis (default: first numeric)
    y_col : str, optional
        Column name for y-axis (default: second numeric)
    size_col : str, optional
        Column name for bubble size (default: third numeric)
    color_col : str, optional
        Column name for color grouping

    Returns:
    --------
    html.Div
        Dash HTML Div containing the Bubble chart
    """
    if df.empty:
        return html.Div("Dataset is empty. Cannot generate bubble chart.", className="text-danger")

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) < 3:
        return html.Div("Need at least 3 numeric columns for bubble chart.", className="text-danger")

    if x_col is None or y_col is None or size_col is None:
        x_col, y_col, size_col = numeric_cols[:3]

    try:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            size=size_col,
            color=color_col,
            title=f"Bubble Chart: {y_col} vs {x_col} (size={size_col})",
            template="plotly_white"
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="closest",
            plot_bgcolor="white",
            paper_bgcolor="white",
            title_x=0.5
        )
        return html.Div(dcc.Graph(figure=fig), className="chart-container shadow p-3 mb-5 bg-white rounded")
    except Exception as e:
        return html.Div(f"Error generating bubble chart: {e}", className="text-danger")
