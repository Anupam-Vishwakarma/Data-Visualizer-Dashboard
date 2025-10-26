# modules/report.py
"""
Report Module for Data Visualizer Dashboard
-------------------------------------------
Generates PDF reports including:
- Dataset overview
- Column-wise summary
- Chart image
- Insights

Uses FPDF for PDF generation and Plotly for chart rendering.
"""

import io
import tempfile
import pandas as pd
import re
from fpdf import FPDF
import plotly.io as pio
from modules.visuals.charts_manager import get_plotly_figure
from modules.insights import generate_insights

def remove_emojis(text):
    """Remove emojis and non-standard characters from text."""
    return re.sub(r'[^\x00-\x7F]+', '', text)

def safe_multicell(pdf, text, line_height=5):
    """Safely write text to PDF, split long lines to prevent FPDF errors."""
    text = remove_emojis(text)
    max_width = pdf.w - 2 * pdf.l_margin
    # Split text into lines that roughly fit page width
    lines = []
    for paragraph in text.split('\n'):
        words = paragraph.split(' ')
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            # Approximate width in units
            if pdf.get_string_width(test_line) < max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
    for line in lines:
        pdf.multi_cell(0, line_height, line.strip())

def generate_report(df: pd.DataFrame, chart_type: str, **kwargs):
    """
    Generate a PDF report for a given dataset and chart type.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to include in the report
    chart_type : str
        Type of chart selected
    **kwargs : dict
        Additional parameters for chart generation (x_col, y_col, color_col, etc.)

    Returns:
    --------
    dict : For Dash dcc.Download
        {"content": <pdf bytes>, "filename": "report.pdf"}
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---------------- Header ----------------
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, remove_emojis("Data Visualizer Dashboard Report"), ln=True, align="C")
    pdf.ln(10)

    # ---------------- Dataset Overview ----------------
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Dataset Overview:", ln=True)
    pdf.set_font("Arial", "", 10)
    safe_multicell(pdf, f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    pdf.ln(5)

    # ---------------- Column-wise Summary ----------------
    for col in df.columns:
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        missing_count = df[col].isna().sum()
        col_text = f"Column '{col}' | Type: {dtype} | Unique: {unique_count} | Missing: {missing_count}"
        if pd.api.types.is_numeric_dtype(df[col]):
            col_text += f" | Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}"
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            top = df[col].mode().values[0] if not df[col].mode().empty else "N/A"
            col_text += f" | Most Frequent: {top}"
        safe_multicell(pdf, col_text)
    pdf.ln(5)

    # ---------------- Chart Image ----------------
    try:
        fig = get_plotly_figure(chart_type, df, **kwargs)
        if fig:
            img_bytes = pio.to_image(fig, format="png", width=700, height=400, scale=2)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                tmpfile.write(img_bytes)
                tmpfile.flush()
                pdf.add_page()
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"{chart_type.title()} Visualization", ln=True)
                pdf.ln(5)
                pdf.image(tmpfile.name, x=15, w=pdf.w - 30)
    except Exception as e:
        safe_multicell(pdf, f"Chart could not be rendered. Error: {e}")

    # ---------------- Insights ----------------
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Insights:", ln=True)
    pdf.set_font("Arial", "", 10)
    try:
        insights_div = generate_insights(df, chart_type)
        for child in getattr(insights_div, "children", []):
            if isinstance(child, list):
                for sub in child:
                    safe_multicell(pdf, str(sub))
            else:
                safe_multicell(pdf, str(child))
    except Exception:
        safe_multicell(pdf, "No insights available.")

    # ---------------- Output ----------------
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    return {"content": pdf_output.getvalue(), "filename": "report.pdf"}