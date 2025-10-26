from dash import dcc, html
import plotly.express as px

def create_line_chart(df, x_col=None, y_col=None, color_col=None):
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) < 1:
        return html.Div("No numeric columns available for line chart.", className="text-danger")

    if y_col is None:
        y_col = numeric_cols[0]

    if x_col is None:
        # Prefer categorical or datetime for x-axis if available
        cat_cols = df.select_dtypes(include=["object", "category", "datetime"]).columns
        x_col = cat_cols[0] if len(cat_cols) > 0 else df.columns[0]

    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=f"Line Chart: {y_col} vs {x_col}",
        template="plotly_white"
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    return html.Div(dcc.Graph(figure=fig), className="chart-container shadow p-3 mb-5 bg-white rounded")
