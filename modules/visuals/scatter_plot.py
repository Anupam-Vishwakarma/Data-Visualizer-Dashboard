from dash import dcc, html
import plotly.express as px

def create_scatter_plot(df, x_col=None, y_col=None, color_col=None):
    numeric_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    if len(numeric_cols) < 2:
        return html.Div("Need at least 2 numeric columns for scatter plot.", className="text-danger")

    if x_col is None or y_col is None:
        x_col, y_col = numeric_cols[:2]

    if color_col is None and len(cat_cols) > 0:
        color_col = cat_cols[0]

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=f"Scatter Plot: {y_col} vs {x_col}",
        template="plotly_white"
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    return html.Div(dcc.Graph(figure=fig), className="chart-container shadow p-3 mb-5 bg-white rounded")
