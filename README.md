# Pro Data Dashboard

A polished, end-to-end Streamlit-based data dashboard for exploratory data analysis (EDA), visualization, insights, summary statistics, machine learning (ML) modeling, and clustering. Upload your CSV or Excel files and get instant visuals, filters, and ML predictions without coding.

## Features

- **Data Upload & Processing**: Supports CSV and Excel (.xls/.xlsx) files. Automatic column type detection (numeric, categorical, datetime).
- **Interactive Filters**: Apply categorical, numeric range, and date filters to refine your dataset.
- **Auto-Generated Visuals**: One-click generation of multiple chart types including:
  - Histograms, Scatter plots, Bar charts, Pie charts, Box plots, Heatmaps, Pairplots, Area charts, Line charts, and Top-N category bars.
  - Customizable colors, axis selections, and user notes for each visual.
  - PNG downloads and shareable configuration codes for visuals.
- **Insights Tab**: Automated textual insights on missing values, correlations, outliers, and category dominance.
- **Summary Tab**: Descriptive statistics, top correlations, and dataset overviews.
- **ML & Clustering Tab**:
  - Baseline models: Linear/Logistic Regression and RandomForest for regression/classification.
  - Feature importances, confusion matrices, ROC curves, and model downloads (.pkl files).
  - KMeans clustering with silhouette scores and PCA projections for visualization.
  - AutoML: Simple model selection and best-model recommendation.
- **PDF Report Generation**: Export insights and charts into a downloadable PDF report.
- **Theme Toggle**: Switch between light and dark modes for better readability.
- **Robust Handling**: Works with large datasets (up to 50,000 rows for ML), sampling for performance, and fallback for missing dependencies (e.g., kaleido for PNG exports).

## Installation

1. **Clone or Download the Repository**:
   - Download the `data_dashboard.py` script and place it in your project directory.

2. **Install Dependencies**:
   - Ensure you have Python 3.7+ installed.
   - Run the following command to install required packages:
     ```
     pip install -r requirements.txt
     ```

3. **Run the App**:
   - Execute the script:
     ```
     streamlit run data_dashboard.py
     ```
   - Open the provided local URL in your browser (e.g., http://localhost:8501).

## Usage

1. **Upload Data**: Use the sidebar to upload a CSV or Excel file.
2. **Apply Filters**: Select categorical, numeric, or date filters to subset your data.
3. **Explore Tabs**:
   - **Visuals**: View and customize auto-generated charts. Add notes, download PNGs, or share configs.
   - **Insights**: Review automated insights and outlier/category reports.
   - **Summary**: Check descriptive stats and correlations.
   - **ML & Clustering**: Choose a target, run models, perform clustering, and generate reports.
4. **Generate Reports**: In the ML tab, enable AutoML to get a best model and download a PDF with insights and charts.
5. **Theme**: Toggle between light/dark modes using the button in the header.

### Tips
- For large datasets, the app automatically samples rows for ML and visualizations to ensure performance.
- If kaleido (for PNG exports) is missing, matplotlib fallbacks are used where possible.
- Models are saved as pickle files for reuse in other projects.

## Requirements

- Python 3.7+
- See `requirements.txt` for a full list of dependencies.

## Dependencies Overview

- **streamlit**: For the web app interface.
- **pandas**: Data manipulation and analysis.
- **numpy**: Numerical computations.
- **matplotlib**: Plotting (used for heatmaps and fallbacks).
- **seaborn**: Statistical visualizations (e.g., pairplots, heatmaps).
- **plotly**: Interactive charts (e.g., histograms, scatters).
- **scikit-learn**: ML models, preprocessing, and metrics.
- **fpdf2**: PDF report generation.
- **kaleido**: For converting Plotly charts to PNG (optional; app handles absence gracefully).
- **openpyxl**: Reading Excel files.

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. Ensure code follows PEP 8 standards and includes comments for clarity.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

Created by Anupam. Based on an end-to-end Streamlit script for data dashboards.

## Troubleshooting

- **File Upload Issues**: Ensure your CSV/Excel has headers and is not corrupted.
- **ML Errors**: For classification, ensure the target has reasonable unique values. Large datasets are sampled.
- **PDF/Chart Export Fails**: Install kaleido via `pip install kaleido` for full PNG support.
- **Performance**: If the app is slow, reduce dataset size or disable heavy visuals like pairplots.

For issues, check the console for error messages or raise an issue in the repository.