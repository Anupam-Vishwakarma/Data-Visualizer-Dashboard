# modules/visuals/area_chart.py


from dash import dcc, html
import plotly.express as px
import pandas as pd

def create_area_chart(df: pd.DataFrame, x_col: str = None, y_col: str = None, color_col: str = None):
    """
    Generate an Area Chart using Plotly Express.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to plot.
    x_col : str, optional
        Column name for x-axis (default: first column if numeric not specified)
    y_col : str, optional
        Column name for y-axis (default: first numeric column)
    color_col : str, optional
        Column name for color grouping

    Returns:
    --------
    html.Div
        Dash HTML Div containing the Plotly Area chart
    """
    if df.empty:
        return html.Div("Dataset is empty. Cannot generate chart.", className="text-danger")

    # Auto-select columns if not provided
    numeric_cols = df.select_dtypes(include="number").columns
    if not numeric_cols.any():
        return html.Div("No numeric columns available for area chart.", className="text-danger")

    if x_col is None:
        x_col = df.columns[0]  # fallback: first column
    if y_col is None:
        y_col = numeric_cols[0]  # first numeric column as y-axis

    try:
        fig = px.area(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=f"Area Chart: {y_col} vs {x_col}",
            template="plotly_white"
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
            title_x=0.5  # Center the title
        )
        return html.Div(dcc.Graph(figure=fig), className="chart-container shadow p-3 mb-5 bg-white rounded")
    except Exception as e:
        return html.Div(f"Error generating area chart: {e}", className="text-danger")
