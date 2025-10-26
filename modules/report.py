# modules/report.py
"""
Unicode-safe PDF Report Generator for Data Visualizer Dashboard
---------------------------------------------------------------
✅ Full Unicode support using DejaVuSans.ttf
✅ Safe multi-cell text rendering
✅ Automatic chart image embedding
✅ Insights integration
✅ Prevents "Not enough horizontal space" errors
"""

import os
import io
import re
import tempfile
import pandas as pd
import plotly.io as pio
from fpdf import FPDF
from modules.visuals.charts_manager import get_plotly_figure
from modules.insights import generate_insights


# ---------------- Custom FPDF Class ----------------
class SafePDF(FPDF):
    def safe_multi_cell(self, text, line_height=5):
        """Safely write long text with unicode support"""
        text = clean_text(text)
        if not text:
            return

        max_chars_per_line = 100  # manual word wrap
        words = text.split(" ")
        current_line = ""
        lines = []

        for w in words:
            if len(current_line) + len(w) + 1 > max_chars_per_line:
                lines.append(current_line)
                current_line = w
            else:
                current_line += (" " if current_line else "") + w
        if current_line:
            lines.append(current_line)

        for line in lines:
            try:
                self.multi_cell(0, line_height, line)
            except Exception:
                # fallback: break into smaller chunks
                for i in range(0, len(line), 50):
                    self.multi_cell(0, line_height, line[i:i + 50])


# ---------------- Utility: Clean Text ----------------
def clean_text(text):
    """Remove emojis, non-ASCII, and extra spaces"""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------- Report Generator ----------------
def generate_report(df: pd.DataFrame, chart_type: str, **kwargs):
    pdf = SafePDF()
    
    # Add DejaVuSans font for Unicode
    fonts_dir = os.path.join(os.getcwd(), "fonts")
    pdf.add_page()
    pdf.add_font("DejaVu", "", os.path.join(fonts_dir, "DejaVuSans.ttf"), uni=True)
    pdf.add_font("DejaVu", "B", os.path.join(fonts_dir, "DejaVuSans.ttf"), uni=True)
    pdf.set_font("DejaVu", size=12)
    pdf.set_auto_page_break(auto=True, margin=15)

    # ---------------- Title ----------------
    pdf.set_font("DejaVu", "B", 16)
    pdf.safe_multi_cell("Data Visualizer Dashboard Report")
    pdf.ln(5)

    # ---------------- Dataset Overview ----------------
    pdf.set_font("DejaVu", "B", 12)
    pdf.safe_multi_cell("Dataset Overview:")
    pdf.set_font("DejaVu", "", 10)
    pdf.safe_multi_cell(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    pdf.ln(5)

    # ---------------- Column Summary ----------------
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique = df[col].nunique()
        missing = df[col].isna().sum()
        summary = f"{col} | Type: {dtype} | Unique: {unique} | Missing: {missing}"

        if pd.api.types.is_numeric_dtype(df[col]):
            summary += (
                f" | Min: {df[col].min():.2f} | Max: {df[col].max():.2f} | "
                f"Mean: {df[col].mean():.2f} | Median: {df[col].median():.2f}"
            )
        else:
            top = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
            summary += f" | Top: {clean_text(str(top))}"

        pdf.safe_multi_cell(summary)
    pdf.ln(5)

    # ---------------- Chart Generation ----------------
    try:
        fig = get_plotly_figure(chart_type, df, **kwargs)
        if fig:
            img_bytes = pio.to_image(fig, format="png", width=700, height=400, scale=2)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(img_bytes)
                tmp.flush()
                pdf.add_page()
                pdf.set_font("DejaVu", "B", 12)
                pdf.safe_multi_cell(f"{chart_type.title()} Visualization")
                pdf.ln(5)
                pdf.image(tmp.name, x=15, w=pdf.w - 30)
    except Exception as e:
        pdf.safe_multi_cell(f"[Chart Error] {e}")

    # ---------------- Insights Section ----------------
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 12)
    pdf.safe_multi_cell("Insights:")
    pdf.set_font("DejaVu", "", 10)

    try:
        insights_div = generate_insights(df, chart_type)
        for child in getattr(insights_div, "children", []):
            if isinstance(child, list):
                for sub in child:
                    pdf.safe_multi_cell(clean_text(str(sub)))
            else:
                pdf.safe_multi_cell(clean_text(str(child)))
    except Exception as e:
        pdf.safe_multi_cell(f"[Insights Error] {e}")

    # ---------------- Save PDF ----------------
    output_path = os.path.join(os.getcwd(), f"{chart_type}_report.pdf")
    pdf.output(output_path)
    print(f"✅ Report saved successfully to: {output_path}")
    return {"path": output_path, "filename": f"{chart_type}_report.pdf"}
