from dash import dcc, html
import plotly.express as px
import pandas as pd

def create_box_plot(df: pd.DataFrame, x_col: str = None, y_col: str = None, color_col: str = None):
    """
    Generate a Box Plot using Plotly Express.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to plot.
    x_col : str, optional
        Column name for x-axis (default: None)
    y_col : str, optional
        Column name for y-axis (default: first numeric column)
    color_col : str, optional
        Column name for color grouping

    Returns:
    --------
    html.Div
        Dash HTML Div containing the Plotly Box chart
    """
    if df.empty:
        return html.Div("Dataset is empty. Cannot generate box plot.", className="text-danger")

    numeric_cols = df.select_dtypes(include="number").columns
    if not numeric_cols.any():
        return html.Div("No numeric columns available for box plot.", className="text-danger")

    if y_col is None:
        y_col = numeric_cols[0]

    try:
        fig = px.box(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=f"Box Plot: {y_col}" if not x_col else f"Box Plot: {y_col} vs {x_col}",
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
        return html.Div(f"Error generating box plot: {e}", className="text-danger")
