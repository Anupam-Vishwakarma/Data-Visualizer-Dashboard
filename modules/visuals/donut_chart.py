# modules/visuals/donut_chart.py


from dash import dcc, html
import plotly.express as px

def create_donut_chart(df, names_col=None, values_col=None):
    if names_col is None or values_col is None:
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) < 1:
            return html.Div("No numeric columns for donut chart.", className="text-danger")
        names_col = df.columns[0]
        values_col = numeric_cols[0]
    fig = px.pie(df, names=names_col, values=values_col, hole=0.4, title=f"Donut Chart: {values_col} by {names_col}", template="plotly_white")
    fig.update_layout(margin=dict(l=20,r=20,t=40,b=20), plot_bgcolor="white", paper_bgcolor="white")
    return html.Div(dcc.Graph(figure=fig), className="chart-container shadow p-3 mb-5 bg-white rounded")
