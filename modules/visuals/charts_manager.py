# modules/visuals/charts_manager.py

# ------------------ Chart Manager ------------------ #
# Central chart handler for the dashboard.
# Provides functions for generating Dash charts and raw Plotly figures.

from .bar_chart import create_bar_chart
from .line_chart import create_line_chart
from .scatter_plot import create_scatter_plot
from .pie_chart import create_pie_chart
from .box_plot import create_box_plot
from .histogram import create_histogram
from .area_chart import create_area_chart
from .donut_chart import create_donut_chart
from .heatmap import create_heatmap
from .bubble_chart import create_bubble_chart
from dash import html

# Mapping chart_type string to the actual function
chart_mapping = {
    "bar": create_bar_chart,
    "line": create_line_chart,
    "scatter": create_scatter_plot,
    "pie": create_pie_chart,
    "box": create_box_plot,
    "hist": create_histogram,
    "area": create_area_chart,
    "donut": create_donut_chart,
    "heatmap": create_heatmap,
    "bubble": create_bubble_chart
}


def generate_chart(chart_type, df, **kwargs):
    """
    Returns a Dash HTML Div containing the chart
    """
    chart_func = chart_mapping.get(chart_type.lower())
    if not chart_func:
        return html.Div(f"Chart type '{chart_type}' not supported.", className="text-danger")
    
    try:
        return chart_func(df, **kwargs)
    except Exception as e:
        return html.Div(f"Error generating chart '{chart_type}': {e}", className="text-danger")


def get_plotly_figure(chart_type, df, **kwargs):
    """
    Returns the raw Plotly figure object for PDF or other non-Dash usage.
    """
    chart_func = chart_mapping.get(chart_type.lower())
    if not chart_func:
        return None
    
    try:
        chart_div = chart_func(df, **kwargs)
        # chart_div is html.Div -> children[0] is dcc.Graph -> figure
        return chart_div.children[0].figure
    except Exception as e:
        print(f"Error generating figure for PDF: {e}")
        return None
