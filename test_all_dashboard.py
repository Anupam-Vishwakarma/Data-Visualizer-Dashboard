# test_all_dashboard.py
"""
Test all charts and generate reports for multiple datasets
without triggering terminal-based rendering errors.
"""

import os
import pandas as pd
from modules.report import generate_report
from modules.visuals.charts_manager import get_plotly_figure
import plotly.io as pio

# Ensure kaleido is installed: pip install kaleido

# ---------------- Sample Datasets ----------------
datasets = {
    "Dataset 1": pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [5, 6, 7, 8]
    }),
    "Dataset 2": pd.DataFrame({
        "X": [10, 20, 30, 40, 50],
        "Y": [15, 25, 35, 45, 55],
        "Category": ["A", "B", "A", "B", "C"]
    }),
    "Dataset 3": pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie", "David"],
        "Score": [90, 85, 95, 88]
    })
}

# ---------------- Chart Types ----------------
chart_types = [
    "bar", "line", "pie", "scatter", "box", "hist",
    "area", "donut", "heatmap", "bubble"
]

# ---------------- Test Loop ----------------
for dataset_name, df in datasets.items():
    print(f"\n=== Testing {dataset_name} ===")
    
    for chart_type in chart_types:
        report_filename = f"{chart_type}_report.pdf"
        try:
            # Generate figure using Plotly
            fig = get_plotly_figure(chart_type, df)
            
            if fig is None:
                raise ValueError("Figure generation returned None")
            
            # Export figure to PNG bytes for PDF
            img_bytes = pio.to_image(fig, format="png", width=700, height=400, scale=2)
            
            # Save the image temporarily
            temp_img_path = os.path.join(os.getcwd(), f"{chart_type}_temp.png")
            with open(temp_img_path, "wb") as f:
                f.write(img_bytes)
            
            # Generate PDF using report.py
            generate_report(df, chart_type)
            
            # Clean up temp image
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            
            print(f"[✅] {chart_type.title()} chart for {dataset_name} - Report generated successfully")
        
        except Exception as e:
            print(f"[❌] Failed for {dataset_name}, Chart {chart_type}: {e}")

print("\nAll tests completed.")
