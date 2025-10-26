from dash import dcc, html
import plotly.express as px

def create_heatmap(df, x_col=None, y_col=None, z_col=None):
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) < 2:
        return html.Div("Need at least 2 numeric columns for heatmap.", className="text-danger")

    # Default selection
    if x_col is None or y_col is None:
        x_col, y_col = numeric_cols[:2]
    if z_col is None:
        z_col = numeric_cols[0]  # fallback to first numeric column

    fig = px.density_heatmap(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        title=f"Heatmap: {z_col} by {x_col} & {y_col}",
        template="plotly_white"
    )
    fig.update_layout(margin=dict(l=20,r=20,t=40,b=20),
                      plot_bgcolor="white",
                      paper_bgcolor="white")
    return html.Div(dcc.Graph(figure=fig), className="chart-container shadow p-3 mb-5 bg-white rounded")
