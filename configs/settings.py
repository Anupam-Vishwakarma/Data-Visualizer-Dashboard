"""
config/settings.py

Application Settings for Data Visualizer Dashboard
--------------------------------------------------

This file contains global constants and configuration parameters
used throughout the dashboard application.
"""

# Application title shown in browser tab and navbar
APP_TITLE = "📊 Data Visualizer Dashboard"

# Default port for running the app (optional)
APP_PORT = 8050

# Default theme (can be overridden by theme.json)
DEFAULT_THEME = "light"

# Path configurations
DATA_FOLDER = "data"          # Folder where datasets are stored
ASSETS_FOLDER = "assets"      # Folder for images, CSS, JS files
STORAGE_FOLDER = "storage"    # Folder for user preferences, comments, etc.
CONFIG_FOLDER = "configs"     # Folder for configuration files

# Authentication settings
MAX_LOGIN_ATTEMPTS = 5        # Max attempts before temporary lock
OTP_EXPIRY_MINUTES = 5        # OTP validity in minutes

# Report settings
REPORT_FOLDER = "reports"     # Folder where PDF reports will be saved

# Visualization defaults
DEFAULT_CHART_COLOR = "#1f77b4"  # Default color for charts
DEFAULT_CHART_WIDTH = 700
DEFAULT_CHART_HEIGHT = 400
