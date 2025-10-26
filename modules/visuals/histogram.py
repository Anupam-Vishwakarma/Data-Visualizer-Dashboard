from dash import dcc, html
import plotly.express as px

def create_histogram(df, x_col=None, color_col=None):
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) < 1:
        return html.Div("No numeric columns for histogram.", className="text-danger")

    if x_col is None:
        x_col = numeric_cols[0]

    # Default color column (optional)
    if color_col is None:
        cat_cols = df.select_dtypes(include="object").columns
        if len(cat_cols) > 0:
            color_col = cat_cols[0]

    fig = px.histogram(
        df,
        x=x_col,
        color=color_col,
        title=f"Histogram: {x_col}",
        template="plotly_white"
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    return html.Div(dcc.Graph(figure=fig), className="chart-container shadow p-3 mb-5 bg-white rounded")
